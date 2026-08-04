# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Weights without bytes.

``attention_ltx.py:379`` returns ``None`` from ``_compute_gate`` when the gate
weight is unloaded, so a weightless dry run silently loses the exact finding
phase 5 reported (roadmap blocker 37). Every parameter therefore has to be
*loaded* -- just not with data.

Loading goes through the real ``Parameter.load_torch_tensor``, not by assigning
``_data``, which buys three things beyond honesty:

* ``utils/tensor.from_torch`` builds the real mesh mapper, so the distribution the
  shim records is the one tt_dit asked for rather than one restated here;
* ``Parameter._check_data`` compares the shim's per-device shape against
  ``Parameter.local_shape``, computed independently by tt_dit -- a free check on
  the shard math that blocker 36 says is load-bearing;
* dtype and layout mismatches surface here instead of as a wrong byte count.

On ``_prepare_torch_state`` (``_interleave_heads``, swiglu permutation): it does
not run here, because a state dict needs checkpoint keys -- but its *shape* is
already captured. The transpose it applies is baked into ``Parameter.total_shape``,
and the swiglu/interleave reorders are shape-preserving (``prepare_for_fused_swiglu``
maps ``[.., 2N] -> [.., 2N]``), so ``_check_data`` above validates the final
per-device shape on every parameter regardless. What preprocessing changes that
the load path does *not* reconstruct is a fused weight's column *ordering*; the
analyzer reasons about weight shape and value identity, not column order, so this
does not affect findings. Pinning the interleave down belongs to on-device
conformance (blockers 12, phase 11). Checkpoint-derived branch *flags* (blocker 38)
are handled in :mod:`.checkpoint`.
"""

from __future__ import annotations

from typing import List, Tuple

from ..ir import ACT, PARAM
from .context import CTX
from .hostenv import host_tensor, torch_dtype_for
from .install import assert_installed


class param_scope:
    """Mint PARAM symbols named after the parameter path while this is open."""

    def __init__(self, base: str):
        self.base = base

    def __enter__(self):
        CTX.loading_weights = True
        CTX.entry_base = self.base
        return self

    def __exit__(self, *exc):
        CTX.loading_weights = False
        CTX.entry_base = None
        CTX.entry_kind = ACT
        return False


def load_meta_weights(module, prefix: str = "", _top: bool = True) -> int:
    """Load every parameter in ``module`` from meta tensors. Returns the count."""
    assert_installed()
    count = 0
    for name, parameter in module.named_parameters():
        path = prefix + name
        with param_scope(path.replace(".", "_")):
            parameter.load_torch_tensor(host_tensor(parameter.total_shape, torch_dtype_for(parameter.dtype)))
        count += 1
    for name, child in module.named_children():
        count += load_meta_weights(child, prefix + name + ".", _top=False)
    if _top:
        module._mark_loaded()  # noqa: SLF001  -- `is_loaded()` gates real forward paths
    return count


def parameter_symbols() -> List[Tuple[str, tuple]]:
    """The PARAM symbols the run created, for reporting."""
    graph = CTX.require_graph()
    return [(s.id, s.shape) for s in graph.symbols.values() if s.kind == PARAM]
