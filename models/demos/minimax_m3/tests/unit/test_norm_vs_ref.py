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
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.rms_norm import RMSNorm
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

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


# (8, 4) + linear_fabric: the production mesh (SP=8 rows, TP=4 cols) on FABRIC_1D — the plain-MESH
# single-galaxy descriptor has no wrap links, so FABRIC_1D_RING is not openable here.
@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("m, width", [(640, 6144), (128, 6144)], ids=["m640", "m128"])
def test_rms_norm_distributed_interleaved(mesh_device, device_params, m, width, reset_seeds, tmp_path):
    """Distributed (3-op) RMSNorm on a DRAM-INTERLEAVED emb/tp-sharded activation — the sharded-residual
    path (M3_SHARDED_RESIDUAL=1).

    Covers the two things that path adds beyond the single-pass norm:
      1. ``rms_norm_pre_all_gather`` -> stats all-gather -> ``rms_norm_post_all_gather`` with
         ``program_config=None`` (the interleaved kernels; M3's residual is DRAM-interleaved, and the
         pre-existing distributed branch only handled L1-sharded input);
      2. the TP-sharded gain being recovered from the REPLICATED weight cache entry in cache-only mode
         (empty state_dict) — which is how the production model builds it, since the tilized cache holds
         only the replicated layout.

    Checked against the fp32 torch reference AND against the single-pass norm on the same weight, so a
    regression in either the collective or the gain sharding shows up as a PCC drop rather than a shape
    error somewhere downstream.
    """
    tp = mesh_device.shape[1]
    if tp <= 1:
        pytest.skip("distributed norm needs tp > 1")
    assert width % (tp * ttnn.TILE_SIZE) == 0, f"width {width} must split tile-aligned across tp={tp}"

    eps = 1e-6
    x = torch.randn(1, 1, m, width)
    weight = torch.randn(width)
    ref = _torch_rms_norm(x, weight, eps, gemma=True)

    hf_config = SimpleNamespace(rms_norm_eps=eps, use_gemma_norm=True)
    mesh_config = MeshConfig(tuple(mesh_device.shape), tp=tp)
    # Linear topology: valid on a ring-configured fabric (it just leaves the wrap link unused) and it is
    # what the single-galaxy FABRIC_1D model run uses.
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    # Populate the replicated weight cache the way the real weight-cache build does...
    cache_dir = str(tmp_path / "norm")
    RMSNorm(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict={"weight": weight},
        tensor_cache_path=cache_dir,
        mesh_config=mesh_config,
        ccl_manager=ccl,
        is_distributed=False,
    )
    # ...then build BOTH norms cache-only (state_dict={}), as the production model does.
    single = RMSNorm(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict={},
        tensor_cache_path=cache_dir,
        mesh_config=mesh_config,
        ccl_manager=ccl,
        is_distributed=False,
    )
    dist = RMSNorm(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict={},
        tensor_cache_path=cache_dir,
        mesh_config=mesh_config,
        ccl_manager=ccl,
        is_distributed=True,
    )

    # Sharded input: emb across the TP cols, replicated across the rows. DRAM-interleaved on purpose.
    x_sharded = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, -1)),
    )
    out_dist = dist(x_sharded)
    assert out_dist.shape[-1] == width // tp, f"distributed norm must stay emb/tp, got {out_dist.shape}"
    # Compose the per-column shards back to full width (row 0's columns).
    shards = ttnn.get_device_tensors(out_dist)
    dist_full = torch.cat([ttnn.to_torch(shards[c]).reshape(1, 1, m, width // tp).float() for c in range(tp)], dim=-1)

    x_full = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    single_full = ttnn.to_torch(ttnn.get_device_tensors(single(x_full))[0]).reshape(1, 1, m, width).float()

    ok_ref, pcc_ref = comp_pcc(ref, dist_full, 0.99)
    ok_single, pcc_single = comp_pcc(single_full, dist_full, 0.99)
    logger.info(f"distributed rms_norm m={m} width={width} tp={tp}: vs_ref={pcc_ref} vs_single_pass={pcc_single}")
    assert ok_ref, f"distributed norm vs fp32 reference PCC fail: {pcc_ref}"
    assert ok_single, f"distributed norm vs single-pass norm PCC fail: {pcc_single}"
