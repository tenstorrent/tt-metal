# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Route adapter tensors through a model's own weight-preparation pipeline.

A LoRA delta has to arrive in the same layout as the base weight it modifies. tt_dit models
routinely fuse and permute weights on the way in -- ``MiniMaxH3Attention`` merges q/k/v into one
``to_qkv``, interleaves heads across the TP factor *and* permutes Q/K rotary channels;
``ColParallelLinear`` packs SwiGLU gate/up into interleaved tile pairs. Every one of those
transforms already exists, in ``Module._prepare_torch_state``.

Rather than mirror them per model (the Wan loader carries a hand-written copy of one), run the
adapter's tensors down the same recursion ``Module._load_torch_state_dict_inner`` uses, and stop
one step short of loading. What comes back is each destination ``Parameter`` paired with the
tensor in its final layout.

This works for LoRA ``B`` factors as well as for full-rank deltas because every transform in that
pipeline acts on the *output* dimension and treats the input dimension as opaque rows. A ``B`` of
shape ``[out, rank]`` therefore transforms exactly as a weight of shape ``[out, in]`` does. ``A``
lives on the input side and needs no routing at all.

Two contracts the caller must honour:

* A module whose ``_prepare_torch_state`` fuses several children (``to_q``/``to_k``/``to_v``) reads
  all of them unconditionally. Supply a fused group whole or not at all.
* The tensors handed in are deltas, so the routed result is a delta. Any transform that is not
  linear in the tensor would break that -- none in tt_dit are, they are all reshapes, permutes and
  concatenations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..utils.substate import pop_substate

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import torch

    from ..layers.module import Module, Parameter


@dataclass(frozen=True)
class RoutedTensor:
    """One adapter tensor resolved to its destination."""

    path: str
    """Dotted path of the destination parameter, e.g. ``transformer_blocks.0.attn.to_qkv.weight``."""
    module: Module
    """The module owning the parameter -- the bind target for a LoRA factor."""
    param_name: str
    """Parameter name within ``module``, ``"weight"`` or ``"bias"``."""
    param: Parameter
    value: torch.Tensor
    """The tensor after every ``_prepare_torch_state`` on the path from the root."""


def route(root: Module, state: Mapping[str, torch.Tensor]) -> tuple[list[RoutedTensor], list[str]]:
    """Resolve ``state`` against ``root``'s weight-preparation pipeline without loading anything.

    Returns the resolved tensors and the keys that reached a leaf with no parameter to land on.
    Unresolved keys are the caller's error to report: for an adapter they mean a target the model
    does not have, which is a coverage bug rather than something to skip.
    """
    routed: list[RoutedTensor] = []
    unresolved: list[str] = []
    _route_inner(root, dict(state), prefix="", routed=routed, unresolved=unresolved)
    return routed, unresolved


def _route_inner(
    module: Module,
    state: dict[str, torch.Tensor],
    *,
    prefix: str,
    routed: list[RoutedTensor],
    unresolved: list[str],
) -> None:
    try:
        module._prepare_torch_state(state)  # noqa: SLF001
    except KeyError as err:
        # A fusing hook reads all of its sources unconditionally, so a group supplied in part
        # surfaces here as a bare KeyError naming a sibling the caller never mentioned.
        msg = (
            f"preparing '{prefix or '<root>'}' ({type(module).__name__}) needs {err}, which this "
            f"adapter did not supply alongside {sorted(state)}; fused targets must be routed whole"
        )
        raise KeyError(msg) from err

    for name, child in module.named_children():
        child_state = pop_substate(state, name)
        if child_state:
            _route_inner(child, child_state, prefix=f"{prefix}{name}.", routed=routed, unresolved=unresolved)

    for name, parameter in module.named_parameters():
        if name in state:
            routed.append(
                RoutedTensor(
                    path=f"{prefix}{name}",
                    module=module,
                    param_name=name,
                    param=parameter,
                    value=state.pop(name),
                )
            )

    unresolved.extend(f"{prefix}{name}" for name in state)


def named_modules(root: Module, prefix: str = "") -> Iterator[tuple[str, Module]]:
    """Every module under ``root``, paired with its dotted path. Root itself is ``""``."""
    yield prefix.rstrip("."), root
    for name, child in root.named_children():
        yield from named_modules(child, f"{prefix}{name}.")
