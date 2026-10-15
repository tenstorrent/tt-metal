# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf harness for matmul_reduce_scatter (run under scripts/run_safe_pytest.sh --profile).

N back-to-back calls of a LOOSE_CASES geometry; read DEVICE KERNEL DURATION of the later calls from the profiler CSV
(the first call carries the cross-chip launch skew at the ready fence). MMRS_CASES picks the cases (comma list),
MMRS_CALLS the calls per case, MMRS_LINKS num_links (default 2).

The mesh opens with the production fabric router config (max packet payload 14 KiB + 64 B on Blackhole, 7 KiB + 64 B
on Wormhole), as the golden suite and the models do -- every perf target assumes it. MMRS_PAYLOAD=<bytes> overrides
it (e.g. 4352, the router default), MMRS_PAYLOAD=default keeps the router default config."""

import os

import pytest
import torch
import ttnn

from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

CASES = {
    "focus": ((1, 1, 640, 2048), (2048, 7168), 1, -1, ttnn.bfloat8_b, False),
    "glm": ((1, 1, 640, 4096), (4096, 6144), 1, -1, ttnn.bfloat8_b, False),
    "mimo": ((1, 1, 2048, 2048), (2048, 4096), 1, -2, ttnn.bfloat8_b, True),
    "smallk": ((1, 1, 640, 512), (512, 7168), 1, -1, ttnn.bfloat8_b, False),  # link-bound small K
    "r2": ((1, 1, 2048, 4096), (4096, 4096), 1, -1, ttnn.bfloat8_b, True),  # regime R2 (both operands streamed)
}


def _router_params():
    payload = os.environ.get("MMRS_PAYLOAD")
    if payload == "default":
        return {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (
        int(payload) if payload else (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    )
    return {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "fabric_router_config": cfg}


def _system_mesh_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


def _to_mesh(stacked, mesh_device, dtype):
    rows = stacked.shape[0]
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        glob,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )


@pytest.mark.parametrize("mesh_device", [_system_mesh_shape()], indirect=True)
@pytest.mark.parametrize("device_params", [_router_params()], indirect=True)
@pytest.mark.parametrize("case", os.environ.get("MMRS_CASES", "focus").split(","))
def test_perf(mesh_device, case):
    a_shape, w_shape, axis, sd, wdt, fp32 = CASES[case]
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(0)
    a = _to_mesh(torch.randn((rows, cols, *a_shape)).to(torch.bfloat16), mesh_device, ttnn.bfloat16)
    w = _to_mesh((torch.randn((rows, cols, *w_shape)) * w_shape[0] ** -0.5), mesh_device, wdt)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32)
    for _ in range(int(os.environ.get("MMRS_CALLS", "5"))):
        matmul_reduce_scatter(
            a,
            w,
            cluster_axis=axis,
            scatter_dim=sd,
            num_links=int(os.environ.get("MMRS_LINKS", "2")),
            compute_kernel_config=cfg,
        )
    ttnn.synchronize_device(mesh_device)
