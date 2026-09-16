# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Transport cost of rebuilding the same 56320-row KV slab in 8 hops versus 32.

Both arms all-gather the identical total bytes onto every chip and differ only in how the source is
split: 8 shards of 7040 rows over the SP axis, or 32 shards of 1760 over the whole mesh. The
difference bounds what ring_mla's fused all-gather pays for a 32-deep ring, which is the quantity its
per-step arrival exposure is drawn from.
"""

import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    MESH_CONFIG,
    close_ring_joint_sdpa_runtime,
    fa_rand,
    open_ring_joint_sdpa_runtime,
)

WARMUP = 2
ITERS = 10
D_K = 576
TOTAL_ROWS = 56320


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("hops", [8, 32], ids=["sp_axis_8hops", "full_mesh_32hops"])
def test_ag_transport_cost(hops):
    if not MESH_CONFIG.is_galaxy:
        pytest.skip("needs the 8x4 Galaxy")
    sp, tp = MESH_CONFIG.sp_size, MESH_CONFIG.tp_size
    full_mesh = hops == 32
    runtime = open_ring_joint_sdpa_runtime(MESH_CONFIG, full_mesh=True, sp_outer=True)
    try:
        mesh_device = runtime.mesh_device
        if not ttnn.device.IsProgramRealtimeProfilerActive():
            pytest.fail("realtime profiler is not active")

        torch.manual_seed(2026)
        src = fa_rand(1, 1, TOTAL_ROWS, D_K)
        rows_per_shard = TOTAL_ROWS // hops
        mapper = (
            ttnn.ShardTensorToMesh(mesh_device, dim=2)
            if full_mesh
            else ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[2, None])
        )
        tt_in = ttnn.from_torch(
            src, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper
        )
        assert tt_in.shape[2] == rows_per_shard, f"{tt_in.shape[2]} != {rows_per_shard}"
        if full_mesh:
            dist_shape = ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1])
            coords = [
                ttnn.MeshCoordinate([coord[i] for i in range(coord.dims())])
                for coord in ttnn.MeshCoordinateRange(dist_shape)
            ]
            tt_in.update_tensor_topology(
                ttnn.TensorTopology(dist_shape, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords)
            )

        out_buf = ttnn.from_torch(
            torch.zeros(1, 1, TOTAL_ROWS, D_K),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        def one():
            ttnn.experimental.high_bw_all_gather(
                tt_in,
                dim=2,
                output_tensor=out_buf,
                num_links=runtime.num_links,
                cluster_axis=None if full_mesh else 0,
                gathered_dim_size=TOTAL_ROWS,
            )

        for _ in range(WARMUP):
            one()
        ttnn.synchronize_device(mesh_device)

        samples = []
        for _ in range(ITERS):
            _, records = profile_realtime_program(mesh_device, one, collect_all=True)
            per_program = {}
            for record in records:
                rid = record["runtime_id"]
                if not rid:
                    continue
                per_program[rid] = max(per_program.get(rid, 0.0), float(record["duration_ns"]))
            if not per_program:
                pytest.fail("realtime profiler returned no program records")
            samples.append(sum(per_program.values()) / 1e6)
        samples.sort()
        mb = TOTAL_ROWS * D_K / 1e6
        logger.info(
            f"[ag-transport] RESULT {hops} hops ({rows_per_shard} rows/shard): DEVICE min {samples[0]:.4f} ms "
            f"median {samples[len(samples)//2]:.4f} ms  max {samples[-1]:.4f} ms  "
            f"({mb:.2f} MB gathered, {mb/samples[0]*1e-3:.1f} GB/s, {samples[0]/hops*1000:.1f} us/hop)"
        )
    finally:
        close_ring_joint_sdpa_runtime(runtime)
