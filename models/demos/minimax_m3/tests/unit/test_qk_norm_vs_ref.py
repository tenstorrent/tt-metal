# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-1 PCC test for MiniMax-M3 per-head QK-norm vs a hand-written torch reference.

M3 uses qk_norm_type="per_head": after the head split (Q -> [1, 64, S, 128], K -> [1, 4, S, 128]),
RMSNorm runs independently per (head, token) over head_dim with a single [head_dim] gain shared
across heads, applied BEFORE RoPE on Q and K only. The gain uses the same gemma (1+w) RMSNorm as
the layernorms (use_gemma_norm=true), folded into the weight at load (weights.py). Anchor:
transformers minimax_m3_vl.

Builds the qk-norm gain exactly as weights.py does (gemma fold + (1,1,head_dim/32,32) ROW_MAJOR,
replicated) and applies the production apply_qk_norm_per_head op. Torch-only, random weights — single card.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.tt.attention.operations import apply_qk_norm_per_head

from ..test_factory import parametrize_mesh_with_fabric


def _torch_qk_norm_per_head(x: torch.Tensor, weight: torch.Tensor, eps: float, gemma: bool) -> torch.Tensor:
    """Per-head RMSNorm over head_dim (last dim); weight [head_dim] broadcasts across heads."""
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(var + eps)
    scale = (1.0 + weight.float()) if gemma else weight.float()
    return normed * scale


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("gemma", [True, False], ids=["gemma", "plain"])
@pytest.mark.parametrize(
    "n_heads, seq_len, head_dim",
    [
        (64, 128, 128),  # Q: num_attention_heads
        (4, 128, 128),  # K: num_key_value_heads
    ],
    ids=["q64", "k4"],
)
def test_qk_norm_per_head_vs_ref(mesh_device, device_params, gemma, n_heads, seq_len, head_dim, reset_seeds):
    """apply_qk_norm_per_head vs torch per-head reference, random weights."""
    eps = 1e-6
    x = torch.randn(1, n_heads, seq_len, head_dim)
    weight = torch.randn(head_dim)

    ref = _torch_qk_norm_per_head(x, weight, eps, gemma)

    # Build the gain exactly as weights.py: gemma fold, then (1, 1, head_dim/32, 32) ROW_MAJOR.
    w = weight.float() + 1.0 if gemma else weight.float()
    w = w.reshape(1, 1, -1, ttnn.TILE_SIZE)

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    w_tt = ttnn.from_torch(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    out_tt = apply_qk_norm_per_head(x_tt, w_tt, eps)
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).reshape(1, n_heads, seq_len, head_dim)

    passing, pcc = comp_pcc(ref, out, 0.99)
    logger.info(f"qk_norm_per_head gemma={gemma} n_heads={n_heads} head_dim={head_dim}: {pcc}")
    assert passing, f"PCC fail (gemma={gemma}, n_heads={n_heads}): {pcc}"
