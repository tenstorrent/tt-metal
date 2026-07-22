# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-1 PCC test for MiniMax-M3 clamped "swigluoai" SwiGLU vs a hand-written torch reference.

M3 uses the gpt-oss SwiGLU variant (hidden_act="swigluoai", swiglu_alpha=1.702, swiglu_limit=7.0):
    gate = clamp(gate, max=limit); up = clamp(up, -limit, limit)
    out  = (up + 1) * (gate * sigmoid(alpha * gate))
vs a plain SiLU SwiGLU (silu(gate) * up). Reference anchor: transformers
modeling_gpt_oss.py:119-122.

Inputs are scaled past ±limit so the clamp path is actually exercised. Depends ONLY on torch
(no HuggingFace / checkpoint), random inputs — runs on a single Wormhole/Blackhole card.
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.tt.moe.activation import apply_swiglu

from ..test_factory import parametrize_mesh_with_fabric


def _torch_swiglu(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """gpt-oss clamped swigluoai reference (fp32)."""
    gate = gate.float().clamp(max=limit)
    up = up.float().clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(alpha * gate)
    return (up + 1.0) * glu


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("alpha, limit", [(1.702, 7.0)], ids=["a1.702_l7"])
@pytest.mark.parametrize(
    "m, width",
    [
        (128, 3072),  # expert / shared intermediate_size
        (128, 12288),  # dense_intermediate_size (layers 0-2)
        (32, 3072),  # single tile of tokens
    ],
    ids=["i3072", "i12288", "m32"],
)
def test_swiglu_vs_ref(mesh_device, device_params, alpha, limit, m, width, reset_seeds):
    """apply_swiglu (clamped swigluoai) vs torch reference, random inputs spanning past ±limit."""
    # Scale past the ±7 clamp so both clamp branches are exercised.
    gate = torch.randn(1, 1, m, width) * 3.0
    up = torch.randn(1, 1, m, width) * 3.0

    ref = _torch_swiglu(gate, up, alpha, limit)

    config = SimpleNamespace(swiglu_limit=limit, alpha=alpha)

    def _to_tt(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    out_tt = apply_swiglu(_to_tt(gate), _to_tt(up), config)
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).reshape(1, 1, m, width)

    passing, pcc = comp_pcc(ref, out, 0.99)
    logger.info(f"swiglu m={m} width={width} alpha={alpha} limit={limit}: {pcc}")
    assert passing, f"PCC fail (width={width}): {pcc}"
