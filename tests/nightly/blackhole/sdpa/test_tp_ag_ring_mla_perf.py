# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device cost of the shipped TP-dedup path: a TP all-gather feeding an 8-deep SP ring_mla.

Both arms run the same kimi50k final chunk -- 5120 Q rows against a 56320 prefix, Q sharded over SP
alone -- on an (sp, tp) mesh over FABRIC_2D_TORUS_XY, and differ only in how the KVPE cache is stored
and what it takes to make ring_mla readable:

  sp_only   cache block-cyclic over SP, 7040 rows/device      ring_mla alone
  tp_ag     cache block-cyclic over SP*TP, 1760 rows/device   high_bw_all_gather over TP, then ring_mla

The gather rebuilds one SP rank's slab rank-major and ring_mla's reader decodes it, which is what
kv_block_cyclic_cache_tp_sharded selects. Timing is device program duration from the realtime
profiler: each program contributes its max across chips and the programs are summed, so the tp_ag arm
counts the gather and the attention together.
"""

import pytest
import torch
from loguru import logger

import ttnn
from ttnn import Topology

from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    CHUNKED_PREFILL_CHUNK_SIZE,
    CHUNKED_PREFILL_HEADS_PER_RING,
    CHUNKED_PREFILL_N_CHUNKS,
    MESH_CONFIG,
    close_ring_joint_sdpa_runtime,
    fa_rand,
    open_ring_joint_sdpa_runtime,
)

WARMUP = 2
ITERS = 10
D_Q = D_K = 576
D_V = 512
Q_CHUNK = 32
K_CHUNK = 640


def _block_cyclic(kv_global, b, nhk, n_chunks, ranks, region, d):
    """Lay the sequence out so a dim-2 shard hands rank r its region of every chunk, chunk-major."""
    return (
        kv_global.reshape(b, nhk, n_chunks, ranks, region, d)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(b, nhk, ranks * n_chunks * region, d)
        .contiguous()
    )


def _profile(mesh_device, run_fn):
    """Per-dispatch device time in ms, plus the number of device programs it covered."""
    samples = []
    programs = 0
    for _ in range(ITERS):
        _, records = profile_realtime_program(mesh_device, run_fn, collect_all=True)
        per_program = {}
        for record in records:
            runtime_id = record["runtime_id"]
            if not runtime_id:  # 0 is the profiler's own sentinel
                continue
            per_program[runtime_id] = max(per_program.get(runtime_id, 0.0), float(record["duration_ns"]))
        if not per_program:
            pytest.fail("realtime profiler returned no program records")
        programs = len(per_program)
        samples.append(sum(per_program.values()) / 1e6)
    samples.sort()
    return samples, programs


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("arm", ["sp_only", "tp_ag"])
def test_tp_ag_ring_mla_cost(arm):
    if not MESH_CONFIG.is_galaxy:
        pytest.skip("needs the 8x4 Galaxy")
    sp, tp = MESH_CONFIG.sp_size, MESH_CONFIG.tp_size
    chunk_global = CHUNKED_PREFILL_CHUNK_SIZE  # 5120
    n_chunks = CHUNKED_PREFILL_N_CHUNKS  # 11
    logical_n = chunk_global * n_chunks  # 56320
    nhq, nhk, b = CHUNKED_PREFILL_HEADS_PER_RING, 1, 1
    tp_sharded = arm == "tp_ag"

    runtime = open_ring_joint_sdpa_runtime(MESH_CONFIG, full_mesh=True, sp_outer=True)
    try:
        mesh_device = runtime.mesh_device
        assert tuple(mesh_device.shape) == (sp, tp), tuple(mesh_device.shape)
        if not ttnn.device.IsProgramRealtimeProfilerActive():
            pytest.fail("realtime profiler is not active; device timing would fall back to host wall clock")

        torch.manual_seed(2026)
        q_global = fa_rand(b, nhq, chunk_global, D_Q)
        tt_q = ttnn.from_torch(
            q_global,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[2, None]),
        )

        kv_global = fa_rand(b, nhk, logical_n, D_K)
        sp_slab = logical_n // sp  # 7040, what ring_mla reads either way
        if tp_sharded:
            ranks, region = sp * tp, chunk_global // (sp * tp)  # 32, 160
            kv_host = _block_cyclic(kv_global, b, nhk, n_chunks, ranks, region, D_K)
            kv_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
        else:
            ranks, region = sp, chunk_global // sp  # 8, 640
            kv_host = _block_cyclic(kv_global, b, nhk, n_chunks, ranks, region, D_K)
            kv_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[2, None])

        tt_kv = ttnn.from_torch(
            kv_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=kv_mapper
        )
        assert tt_kv.shape[2] == (logical_n // ranks), f"{tt_kv.shape[2]} != {logical_n // ranks}"
        if tp_sharded:
            # A TP-deduped cache is dim-2 sharded across both mesh axes, and high_bw_all_gather validates
            # cluster_axis against the declared rank, so the 1-D mapper topology has to be restated.
            dist_shape = ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1])
            coords = [
                ttnn.MeshCoordinate([coord[i] for i in range(coord.dims())])
                for coord in ttnn.MeshCoordinateRange(dist_shape)
            ]
            tt_kv.update_tensor_topology(
                ttnn.TensorTopology(dist_shape, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords)
            )

        # Gather destination: one SP rank's slab, replicated so every chip in the row lands the same.
        tp_gather_buf = (
            ttnn.from_torch(
                torch.zeros(b, nhk, sp_slab, D_K),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            if tp_sharded
            else None
        )
        gathered_kv = ttnn.from_torch(
            torch.zeros(b, nhk, sp * sp_slab, D_K),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=runtime.sdpa_compute_grid,
            q_chunk_size=Q_CHUNK,
            k_chunk_size=K_CHUNK,
            exp_approx_mode=False,
        )
        bc_kwargs = {"kv_block_cyclic_cache_tp_sharded": True} if tp_sharded else {}

        def one():
            ring_kv = tt_kv
            if tp_sharded:
                ring_kv = ttnn.experimental.high_bw_all_gather(
                    tt_kv,
                    dim=2,
                    output_tensor=tp_gather_buf,
                    num_links=runtime.num_links,
                    cluster_axis=1,  # sp_outer puts TP on axis 1
                    gathered_dim_size=sp_slab,
                )
            out, _ = ttnn.transformer.ring_mla(
                tt_q,
                ring_kv,
                persistent_output_buffer_kv=gathered_kv,
                head_dim_v=D_V,
                logical_n=logical_n,
                program_config=program_config,
                compute_kernel_config=runtime.compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=runtime.ccl_semaphore_handles,
                num_links=runtime.num_links,
                cluster_axis=0,  # SP axis
                mesh_device=mesh_device,
                topology=Topology.Ring,
                subdevice_id=runtime.worker_sub_device_id,
                ccl_core_grid_offset=(runtime.ccl_column, 0),
                use_column_major_ccl=True,
                **bc_kwargs,
            )
            ttnn.deallocate(out)

        for _ in range(WARMUP):
            one()
        ttnn.synchronize_device(mesh_device)

        logger.info(
            f"[tp-ag] {arm}: mesh={sp}x{tp} kv/device={tt_kv.shape[2]} region={region} "
            f"sp_slab={sp_slab} q_local={tt_q.shape[2]} logical_n={logical_n} k_chunk={K_CHUNK}"
        )
        samples, programs = _profile(mesh_device, one)
        mean = sum(samples) / len(samples)
        logger.info(
            f"[tp-ag] RESULT {arm}: DEVICE min {samples[0]:.4f} ms  median {samples[len(samples)//2]:.4f} ms  "
            f"mean {mean:.4f} ms  max {samples[-1]:.4f} ms  (n={len(samples)}, {programs} program(s))"
        )
    finally:
        close_ring_joint_sdpa_runtime(runtime)
