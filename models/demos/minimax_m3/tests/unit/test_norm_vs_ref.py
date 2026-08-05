# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-1 PCC test for MiniMax-M3 RMSNorm vs a hand-written torch reference.

M3 uses Gemma-style RMSNorm: out = x_normed * (1 + weight) (config use_gemma_norm=true),
vs a plain RMSNorm: out = x_normed * weight. The tt RMSNorm class folds the +1 into the
weight at load time; this test verifies both modes against the torch reference.

Depends ONLY on torch (no HuggingFace / AutoConfig / checkpoint), random weights — runs on a
single Wormhole/Blackhole card. This is the oracle pattern for M3: self-authored torch
reference + identical random weights, since M3 ships no HF modeling code.
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.tt.rms_norm import RMSNorm

from ..test_factory import parametrize_mesh_with_fabric


def _torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, gemma: bool) -> torch.Tensor:
    """Reference RMSNorm. Gemma form (anchor: transformers modeling_gemma) applies (1 + w);
    plain form applies w. Normalization is done in fp32, matching HF."""
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    scale = (1.0 + weight.float()) if gemma else weight.float()
    return normed * scale


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("gemma", [True, False], ids=["gemma", "plain"])
@pytest.mark.parametrize(
    "m, width",
    [
        (128, 6144),  # hidden_size (decoder layernorm / final norm)
        (128, 128),  # head_dim width (per-head QK-norm geometry)
        (32, 6144),  # single tile of tokens
    ],
    ids=["h6144", "h128", "m32"],
)
def test_rms_norm_vs_ref(mesh_device, device_params, gemma, m, width, reset_seeds):
    """tt RMSNorm class (incl. the gemma (1+w) weight fold) vs torch reference, random weights."""
    eps = 1e-6
    x = torch.randn(1, 1, m, width)
    weight = torch.randn(width)

    ref = _torch_rms_norm(x, weight, eps, gemma)

    hf_config = SimpleNamespace(rms_norm_eps=eps, use_gemma_norm=gemma)
    norm = RMSNorm(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict={"weight": weight},
        tensor_cache_path=None,
    )

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    out_tt = norm(x_tt)
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).reshape(1, 1, m, width)

    passing, pcc = comp_pcc(ref, out, 0.99)
    logger.info(f"rms_norm gemma={gemma} m={m} width={width}: {pcc}")
    assert passing, f"PCC fail (gemma={gemma}, width={width}): {pcc}"
