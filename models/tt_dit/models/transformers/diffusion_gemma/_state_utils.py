# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""State-dict helpers specific to DiffusionGemma's encoder/decoder layout.

The MoE wrapper consumes its state at construction time (the demos/gemma4 ``MoEBlock``
loads weights eagerly), so per-layer ``router.*`` and ``experts.*`` substates have to
be plucked from the larger checkpoint *before* building the model. This helper
centralizes that pattern (was previously copy-pasted across tests + pipeline).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch


def per_layer_moe_substates(
    state: "Mapping[str, torch.Tensor]",
    *,
    num_layers: int,
    prefix: str = "",
) -> list[dict]:
    """Extract per-layer MoE substates from a flat HF state-dict.

    Args:
        state:       Mapping of flat parameter names to tensors.
        num_layers:  Number of layers expected. Result list has this length;
                     entries are empty dicts when no MoE keys are present
                     (useful for randomly-initialized test paths).
        prefix:      Prefix above ``layers.{i}.``. For an isolated layer-state
                     it's ``""``; for a full model it's e.g. ``"model.decoder."``.

    Returns:
        A list of length ``num_layers`` where ``result[i]`` is a flat dict like
        ``{"router.proj.weight": ..., "experts.gate_up_proj": ...}`` — exactly the
        format ``DiffusionGemmaMoE`` accepts as its ``state_dict`` argument.
    """
    substates: list[dict] = [{} for _ in range(num_layers)]
    full_prefix = f"{prefix}layers."
    for k, v in state.items():
        if not k.startswith(full_prefix):
            continue
        rest = k[len(full_prefix) :]
        try:
            i_str, sub = rest.split(".", 1)
            i = int(i_str)
        except ValueError:
            continue
        if i >= num_layers:
            continue
        if sub.startswith("router.") or sub.startswith("experts."):
            substates[i][sub] = v
    return substates
