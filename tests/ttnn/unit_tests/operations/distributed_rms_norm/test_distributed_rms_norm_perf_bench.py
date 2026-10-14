# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-time bench for the feature_spec LOOSE_CASES configs (bf16 x + bf16 gamma, cluster_axis=1, Linear,
num_links=2, HiFi4 + fp32_dest_acc_en). Measurement only — run under `scripts/run_safe_pytest.sh --profile`
and read DEVICE KERNEL DURATION [ns] (one row per device per call) from the ops_perf_results CSV.
The op is called twice per shape (first call compiles); take the second call's rows.
"""

import pytest
import torch
import ttnn

from ttnn.operations.distributed_rms_norm import distributed_rms_norm


def _system_mesh_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


FABRIC_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "fabric_router_config": _router_config(),
    "reliability_mode": ttnn.FabricReliabilityMode.STRICT_INIT,
}


def _to_mesh(stacked, mesh_device, *, dtype, layout):
    rows = stacked.shape[0]
    global_tensor = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        global_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )


@pytest.mark.parametrize("mesh_device", [_system_mesh_shape()], indirect=True)
@pytest.mark.parametrize("device_params", [FABRIC_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "shape", [(1, 1, 640, 1792), (1, 1, 32, 1792), (1, 1, 2048, 1280)], ids=lambda s: "x".join(map(str, s))
)
def test_perf_bench(mesh_device, shape):
    if tuple(mesh_device.shape)[1] < 2:
        pytest.skip("cluster_axis 1 has size 1")
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(0)
    x = torch.randn((rows, cols, *shape)).to(torch.bfloat16)
    g = torch.randn((rows, cols, 1, 1, 1, shape[-1])).to(torch.bfloat16)
    tx = _to_mesh(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tg = _to_mesh(g, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    for _ in range(2):
        distributed_rms_norm(
            tx,
            cluster_axis=1,
            gamma=tg,
            topology=ttnn.Topology.Linear,
            num_links=2,
            compute_kernel_config=cfg,
        )
    ttnn.synchronize_device(mesh_device)
