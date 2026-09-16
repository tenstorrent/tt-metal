# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""What a 32-device ring_mla ring costs against the 8-device SP ring, and where the cost sits.

Every arm runs the same kimi50k final chunk -- 5120 Q rows against a 56320 prefix, Q sharded over
SP alone -- on the same (sp, tp) mesh over FABRIC_2D_TORUS_XY. The 8-ring arms hold ring width and
total K math fixed and vary only the K work-unit count, which separates per-unit overhead from
transport:

  arm                 ring  kv/device  region  k_chunk   units (planned / region-bound)
  sp_ring_8-k640         8       7040     640      640    88 / 88
  sp_ring_8-k320         8       7040     640      320   176 / 176
  sp_ring_8-k160         8       7040     640      160   352 / 352
  mesh_ring_32-k640     32       1760     160      640    96 / 352
  mesh_ring_32-k160     32       1760     160      160   352 / 352

Measured 2026-09-16, 8x4 Blackhole Galaxy, wall clock, 10 iters after 2 warmups (min / median ms):

  sp_ring_8-k640     6.919 /  7.271      mesh_ring_32-k640   8.677 /  9.630
  sp_ring_8-k320     7.817 /  8.286      mesh_ring_32-k160   8.788 /  9.650
  sp_ring_8-k160     8.650 /  9.204

Unit count, not ring width, sets the cost. Going 88 -> 352 units on a fixed 8-ring reproduces the
whole 32-ring gap (1.25x); at a matched 352 units the two ring widths differ by 1.6%. The 32-ring's
1.254x factors as 1.250x unit count x 1.016x ring width.

mesh_ring_32-k640 does not compute the right answer and its timing is not usable. The compute
loop's straddle handler records a single region-boundary jump per K chunk; a 20-tile chunk over a
5-tile region crosses three, so the diagonal stamp is wrong for every column past the first
boundary. It is kept as the standing demonstration that k_chunk > 2x region is unsupported today.
The other four arms are straddle-free and carry the result.
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
    CHUNKED_PREFILL_PER_DEVICE_CHUNK,
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


def _profile_ring_mla(runtime, tt_q, tt_kv, gathered_kv, *, logical_n, cluster_axis, k_chunk):
    """Device program time per dispatch, in ms. Each program contributes its max duration across chips
    and the programs are summed, matching the sparse-MLA / PR #49840 convention."""
    mesh_device = runtime.mesh_device
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("realtime profiler is not active; device timing would fall back to host wall clock")
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=runtime.sdpa_compute_grid,
        q_chunk_size=Q_CHUNK,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )

    def one():
        out, _ = ttnn.transformer.ring_mla(
            tt_q,
            tt_kv,
            persistent_output_buffer_kv=gathered_kv,
            head_dim_v=D_V,
            logical_n=logical_n,
            program_config=program_config,
            compute_kernel_config=runtime.compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=runtime.ccl_semaphore_handles,
            num_links=runtime.num_links,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=Topology.Ring,
            subdevice_id=runtime.worker_sub_device_id,
            ccl_core_grid_offset=(runtime.ccl_column, 0),
            use_column_major_ccl=True,
        )
        ttnn.deallocate(out)

    for _ in range(WARMUP):
        one()
    ttnn.synchronize_device(mesh_device)

    samples = []
    programs = 0
    for _ in range(ITERS):
        _, records = profile_realtime_program(mesh_device, one, collect_all=True)
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


def _report(tag, samples, region_rows, k_chunk, planned, region_bound, pad_pct, programs):
    mean = sum(samples) / len(samples)
    logger.info(
        f"[ring-width] RESULT {tag}: DEVICE min {samples[0]:.4f} ms  median {samples[len(samples)//2]:.4f} ms  "
        f"mean {mean:.4f} ms  max {samples[-1]:.4f} ms  (n={len(samples)}, {programs} program(s), "
        f"units {planned}, pad {pad_pct:.1f}%, region {region_rows} rows, k_chunk {k_chunk})"
    )
    return mean


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "arm,k_chunk,n_chunks",
    [
        # 11 chunks: the shard is 1760 rows, so only 160/352/1760 divide it and the tuned 640 pads 9.1%.
        ("sp_ring_8", 640, 11),
        ("mesh_ring_32", 352, 11),
        ("mesh_ring_32", 640, 11),
        # 12 chunks: the shard is 1920 rows and every width below divides it, so padding is zero
        # throughout and the curve reflects unit count, compute width and exposure alone.
        ("sp_ring_8", 640, 12),
        ("mesh_ring_32", 192, 12),
        ("mesh_ring_32", 320, 12),
        ("mesh_ring_32", 384, 12),
        ("mesh_ring_32", 480, 12),
        ("mesh_ring_32", 640, 12),
    ],
    ids=[
        "sp8-k640-ch11",
        "mesh32-k352-ch11",
        "mesh32-k640-ch11",
        "sp8-k640-ch12",
        "mesh32-k192-ch12",
        "mesh32-k320-ch12",
        "mesh32-k384-ch12",
        "mesh32-k480-ch12",
        "mesh32-k640-ch12",
    ],
)
def test_ring_mla_ring_width_cost(arm, k_chunk, n_chunks):
    if not MESH_CONFIG.is_galaxy:
        pytest.skip("needs the 8x4 Galaxy")
    sp, tp = MESH_CONFIG.sp_size, MESH_CONFIG.tp_size
    chunk_global = CHUNKED_PREFILL_CHUNK_SIZE  # 5120
    logical_n = chunk_global * n_chunks
    nhq, nhk = CHUNKED_PREFILL_HEADS_PER_RING, 1
    b = 1

    full_mesh = arm.startswith("mesh_ring_32")
    runtime = open_ring_joint_sdpa_runtime(MESH_CONFIG, full_mesh=True, sp_outer=True)
    try:
        mesh_device = runtime.mesh_device
        assert tuple(mesh_device.shape) == (sp, tp), tuple(mesh_device.shape)

        torch.manual_seed(2026)
        # Q is the final chunk, sharded over SP alone in BOTH arms.
        q_global = fa_rand(b, nhq, chunk_global, D_Q)
        tt_q = ttnn.from_torch(
            q_global,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[2, None]),
        )
        assert tt_q.shape[2] == chunk_global // sp == CHUNKED_PREFILL_PER_DEVICE_CHUNK

        kv_global = fa_rand(b, nhk, logical_n, D_K)
        if full_mesh:
            ranks, region = sp * tp, chunk_global // (sp * tp)  # 32, 160
            kv_local = logical_n // ranks  # 1760
            # Block-cyclic: position (chunk, dev, off) -> row chunk*region + off on device dev.
            kv_host = (
                kv_global.reshape(b, nhk, n_chunks, ranks, region, D_K)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(b, nhk, ranks * kv_local, D_K)
                .contiguous()
            )
            kv_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
            cluster_axis = None
        else:
            ranks, region = sp, chunk_global // sp  # 8, 640
            kv_local = logical_n // ranks  # 7040
            kv_host = kv_global
            kv_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[2, None])
            cluster_axis = 0  # sp_outer puts SP on axis 0

        tt_kv = ttnn.from_torch(
            kv_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=kv_mapper
        )
        assert tt_kv.shape[2] == kv_local, f"{tt_kv.shape[2]} != {kv_local}"

        gathered_kv = ttnn.from_torch(
            torch.zeros(b, nhk, ranks * kv_local, D_K),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Two counts, because they disagree when k_chunk exceeds the region. `planned` mirrors the host
        # (num_local_k_chunks = div_up(kv_local_padded_N, k_chunk_size)); `region_bound` is what a unit
        # clipped at the 160-row stripe would give. Which one the timing follows is the measurement.
        planned = -(-kv_local // k_chunk) * ranks
        region_bound = -(-kv_local // min(k_chunk, region)) * ranks
        pad_pct = ((-(-kv_local // k_chunk) * k_chunk) / kv_local - 1.0) * 100.0
        logger.info(
            f"[ring-width] {arm}: mesh={sp}x{tp} ring={ranks} kv_local={kv_local} region={region} "
            f"q_local={tt_q.shape[2]} logical_n={logical_n} chunks={n_chunks} k_chunk={k_chunk} "
            f"units planned={planned} region_bound={region_bound} pad={pad_pct:.1f}%"
        )
        samples, programs = _profile_ring_mla(
            runtime, tt_q, tt_kv, gathered_kv, logical_n=logical_n, cluster_axis=cluster_axis, k_chunk=k_chunk
        )
        _report(f"{arm}-k{k_chunk}-ch{n_chunks}", samples, region, k_chunk, planned, region_bound, pad_pct, programs)
    finally:
        close_ring_joint_sdpa_runtime(runtime)
