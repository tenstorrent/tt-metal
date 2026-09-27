# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measure cached host dispatch for GLM-5.2's two distributed norm layouts."""

import statistics
import time

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=6144),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(0)
def test_glm52_fused_rmsnorm_host(mesh_device, device_params):
    """Use production local shape: 640 rows x 1536 hidden elements on each chip."""
    torch.manual_seed(42)
    x = ttnn.from_torch(
        torch.randn((1, 1, 5120, 6144), dtype=torch.bfloat16),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(8, 4), dims=(2, 3)),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tp_topology = per_axis_topology(device_params["fabric_config"])[1]

    for output_name, output_memcfg in (
        ("attn_l1", ttnn.L1_MEMORY_CONFIG),
        ("ffn_dram", ttnn.DRAM_MEMORY_CONFIG),
    ):
        norm = TtDistributedRmsNorm(
            mesh_device=mesh_device,
            emb_dim=6144,
            epsilon=1e-5,
            torch_weight=torch.ones(6144, dtype=torch.bfloat16),
            cluster_axis=1,
            num_links=2,
            topology=tp_topology,
            use_fused=True,
            output_memcfg=output_memcfg,
        )
        original_op = ttnn.experimental.dit_fused_distributed_rmsnorm
        original_resources = norm.tt_ccl.get_fused_rmsnorm_resources
        op_times = []
        resource_times = []

        def timed_op(*args, **kwargs):
            start = time.perf_counter_ns()
            result = original_op(*args, **kwargs)
            op_times.append(time.perf_counter_ns() - start)
            return result

        def timed_resources(*args, **kwargs):
            start = time.perf_counter_ns()
            result = original_resources(*args, **kwargs)
            resource_times.append(time.perf_counter_ns() - start)
            return result

        norm.tt_ccl.get_fused_rmsnorm_resources = timed_resources
        ttnn.experimental.dit_fused_distributed_rmsnorm = timed_op
        try:
            # ABBA order reduces drift from sequential device activity.
            for fused in (False, True, True, False):
                norm.set_fused_enabled(fused)
                for _ in range(10):
                    out = norm(x)
                    ttnn.synchronize_device(mesh_device)
                    ttnn.deallocate(out)
                samples = []
                for _ in range(40):
                    start = time.perf_counter_ns()
                    out = norm(x)
                    samples.append(time.perf_counter_ns() - start)
                    ttnn.synchronize_device(mesh_device)
                    ttnn.deallocate(out)
                print(
                    f"HOST_NORM {output_name} {'fused' if fused else 'baseline'} "
                    f"isolated_p50_us={statistics.median(samples) / 1000:.1f} "
                    f"isolated_min_us={min(samples) / 1000:.1f}",
                    flush=True,
                )

            print(
                f"HOST_NORM {output_name} fused_op_p50_us={statistics.median(op_times) / 1000:.1f} "
                f"resource_lookup_p50_us={statistics.median(resource_times) / 1000:.1f}",
                flush=True,
            )

            for fused in (False, True, True, False):
                norm.set_fused_enabled(fused)
                dispatch_ms = []
                completion_ms = []
                for _ in range(3):
                    start = time.perf_counter_ns()
                    for _ in range(156):
                        out = norm(x)
                        ttnn.deallocate(out)
                    dispatched = time.perf_counter_ns()
                    ttnn.synchronize_device(mesh_device)
                    completed = time.perf_counter_ns()
                    dispatch_ms.append((dispatched - start) / 1e6)
                    completion_ms.append((completed - start) / 1e6)
                print(
                    f"HOST_NORM {output_name} {'fused' if fused else 'baseline'} "
                    f"batch156_dispatch_p50_ms={statistics.median(dispatch_ms):.1f} "
                    f"batch156_complete_p50_ms={statistics.median(completion_ms):.1f}",
                    flush=True,
                )
        finally:
            ttnn.experimental.dit_fused_distributed_rmsnorm = original_op
            norm.tt_ccl.get_fused_rmsnorm_resources = original_resources
