# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi prefill RMSNorm: all mesh shards and repeated shared-resource reuse."""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    assert_requested_tp_wrap_was_realized,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import clear_tt_ccl_cache
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm


@pytest.mark.skipif(not is_blackhole(), reason="Kimi fused prefill targets Blackhole")
@pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4")
@pytest.mark.parametrize(
    "mesh_device,device_params,topology",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=7168, l1_small_size=768),
            ttnn.Topology.Ring,
            id="torus-xy",
        ),
        pytest.param(
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
            id="linear-1d",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_kimi_fused_rmsnorm(mesh_device, device_params, topology):
    assert_requested_tp_wrap_was_realized(mesh_device)
    clear_tt_ccl_cache()
    torch.manual_seed(50932)
    inputs = [
        torch.randn(1, 1, 5120, 7168).to(torch.bfloat16),
        (torch.randn(1, 1, 5120, 7168) + 20).to(torch.bfloat16),
    ]
    weights = [(torch.randn(7168) * 0.2 + 1).to(torch.bfloat16) for _ in inputs]
    xs = [
        ttnn.from_torch(
            x,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(8, 4), dims=(2, 3)),
        )
        for x in inputs
    ]
    norms = [
        TtDistributedRmsNorm(
            mesh_device,
            emb_dim=7168,
            epsilon=1e-5,
            torch_weight=weight,
            cluster_axis=1,
            num_links=2,
            topology=topology,
            use_fused=True,
        )
        for weight in weights
    ]
    # Hold outputs until the whole mixed sequence is enqueued. This catches stale
    # stats, missed semaphore resets and cache-hit weight-address mistakes.
    outputs = [norms[i % 2](xs[i % 2]) for i in range(16)]
    composer = ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(2, 3))
    try:
        for i, output in enumerate(outputs):
            x = inputs[i % 2].float()
            reference = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * weights[i % 2].float()
            actual = ttnn.to_torch(output, mesh_composer=composer).float()
            assert actual.shape == reference.shape
            assert torch.isfinite(actual).all()
            relative_error = ((actual - reference).square().mean() / reference.square().mean()).sqrt().item()
            assert relative_error < 0.01, (i, relative_error)
            ttnn.deallocate(output)
    finally:
        clear_tt_ccl_cache()
