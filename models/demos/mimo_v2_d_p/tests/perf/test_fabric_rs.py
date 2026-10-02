# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""fabric_reduce_scatter (ttnn/ttnn/operations/examples/fabric_reduce_scatter) vs ttnn.reduce_scatter(dim=2) on both
mesh axes at the MiMo shapes: exactness against the torch sum (bf16 partials; the sum of G bf16 values rounds), and
the wall time per call of back-to-back calls (MIMO_RS_CALLS, default 20) after a warm-up.

MIMO_RS_ROWS: rows per chip of the partial (default 4096 on axis 0, the 4x2 send-back; the chip's S on axis 1)."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS, mesh_id

N = int(os.environ.get("MIMO_RS_CALLS", "20"))


def _device_us(dev, f, reps=3):
    """In-process device time (TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
    TT_METAL_PROFILER_CPP_POST_PROCESS=1): per call, each chip's summed program kernel time, the slowest chip; median.
    """
    if not os.environ.get("TT_METAL_PROFILER_MID_RUN_DUMP"):
        return None
    ts = []
    for _ in range(reps):
        ttnn.ReadDeviceProfiler(dev)
        f()
        ttnn.ReadDeviceProfiler(dev)
        data = ttnn.get_latest_programs_perf_data()
        ts.append(
            max(
                sum(p.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration for p in progs) / 1e3
                for progs in data.values()
            )
        )
    return sorted(ts)[len(ts) // 2]


def _time(dev, f):
    f()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(N):
        f()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / N * 1e6


@MESH_PARAMS
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_fabric_rs(mesh_device, device_params, cluster_axis):
    from ttnn.operations.examples.fabric_reduce_scatter.program_descriptor_with_inline_kernels import (
        fabric_reduce_scatter,
    )

    rows, cols = tuple(mesh_device.shape)
    G = (rows, cols)[cluster_axis]
    if G < 2:
        pytest.skip("one chip on this axis")
    T = int(os.environ.get("MIMO_RS_ROWS", str(4096 if cluster_axis == 0 else 4096 // rows)))
    H = 4096
    torch.manual_seed(cluster_axis)
    parts = torch.randn(rows, cols, T, H).bfloat16()
    x = ttnn.from_torch(
        parts,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    x = ttnn.reshape(x, (1, 1, T, H))
    Sb = T // G
    # reference: chip (r, c) gets block (its position on the axis) of the sum over the axis
    ref = parts.float().sum(cluster_axis, keepdim=True)
    sp_topo, tp_topo = per_axis_topology()
    topo = (sp_topo, tp_topo)[cluster_axis]

    worst = {}
    for name, f in (
        ("fabric", lambda: fabric_reduce_scatter(x, cluster_axis=cluster_axis, num_links=2)),
        (
            "ttnn",
            lambda: ttnn.reduce_scatter(
                x, dim=2, cluster_axis=cluster_axis, topology=topo, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
        ),
    ):
        out = f()
        err = 0.0
        for d, t in enumerate(ttnn.get_device_tensors(out)):
            r, c = divmod(d, cols)
            pos = (r, c)[cluster_axis]
            want = ref[0 if cluster_axis == 0 else r, 0 if cluster_axis == 1 else c, pos * Sb : (pos + 1) * Sb]
            err = max(err, (ttnn.to_torch(t).float()[0, 0] - want).abs().max().item())
        worst[name] = err
        us = _time(mesh_device, f)
        dev_us = _device_us(mesh_device, f)
        busiest = (G - 1) * Sb * H * 2  # bytes over the busiest link direction (line)
        print(
            f"FABRIC_RS {mesh_id(mesh_device)} axis {cluster_axis} G {G} T {T}: {name:6s} {us:8.1f} us/call, "
            f"busiest link {busiest / us / 1e3:5.1f} GB/s, max abs err {err:.3g}"
            + (f" | device {dev_us:7.1f} us, {busiest / dev_us / 1e3:5.1f} GB/s" if dev_us else "")
        )
    assert worst["fabric"] <= max(0.25, 2 * worst["ttnn"]), worst
