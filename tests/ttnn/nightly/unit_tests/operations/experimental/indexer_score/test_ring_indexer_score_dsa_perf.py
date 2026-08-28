# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Realtime-profiler coverage for the fused DSA ring indexer.

This is a per-TP-lane proxy for GLM-5.2: the production 8x4 Galaxy mesh has an
8-rank SP ring and TP=4. Its 5120-token chunk makes a 640-row SP slab, which
the model splits over TP before the fused op, yielding 160 query rows and all
32 indexer heads on every physical chip. QuietBox (4x1) and LoudBox (8x1)
cannot represent the TP dimension, so this test retains their complete physical
SP rings and measures one such post-TP lane per chip. The 55K and 512K
key-cache lengths cover the normal production prefix and the long-context
transport/score regime.

Run (requires an IOMMU-enabled Blackhole runner):
    scripts/run_safe_pytest.sh \\
        tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_ring_indexer_score_dsa_perf.py -s
"""

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import get_indexer_key_chunk
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler


# 55 Ki tokens is the GLM chunked-prefill width (50 Ki history + 5 Ki chunk).
# 512 Ki is the long-context target; both are tile-aligned.
GLM52_GLOBAL_CHUNK = 5120
GLM52_SP = 8
GLM52_TP = 4
GLM52_Q_PER_SP_RANK = GLM52_GLOBAL_CHUNK // GLM52_SP
GLM52_Q_PER_CHIP = GLM52_Q_PER_SP_RANK // GLM52_TP
GLM52_KV_55K = 56320
GLM52_KV_512K = 512 * 1024
GLM52_INDEX_HEADS = GLM52Config.INDEX_N_HEADS
GLM52_INDEX_DIM = GLM52Config.INDEX_HEAD_DIM
# The runtime requires a whole 5120-token global chunk. Keep the largest legal
# cache no greater than GLM-5.2's advertised 1 Mi position limit.
GLM52_K_CACHE_CAPACITY = GLM52Config.MAX_POSITION_EMBEDDINGS // GLM52_GLOBAL_CHUNK * GLM52_GLOBAL_CHUNK
GLM52_INDEX_CACHE_SLOTS = sum(indexer_type == "full" for indexer_type in GLM52Config.indexer_types())
GLM52_INDEX_CACHE_SLOT = GLM52_INDEX_CACHE_SLOTS - 1

RING_PERF_KV_LENS = (GLM52_KV_55K, GLM52_KV_512K)
RING_PERF_KV_IDS = ("55k", "512k")
GLM52_K_CHUNK = get_indexer_key_chunk(GLM52_INDEX_HEADS)

# Do not measure a logical subset of another box: bandwidth and torus routing are
# properties of the complete physical box.  The unmatched shape skips, leaving
# 4x1 on QuietBox and 8x1 on LoudBox.
RING_PERF_MESHES = ((4, 1), (8, 1))
RING_PERF_MESH_IDS = ("quietbox_4x1", "loudbox_8x1")

# Match the indexer_score perf gate: expected FPU utilization is compared with
# a symmetric +/-2% relative band.  These are warm trace-replay realtime-profiler
# baselines taken on the complete physical boxes.  A value is deliberately per
# SKU: the proxy retains the box's actual fabric path rather than treating a 4x1
# subset of an 8x1 box as equivalent.
RING_INDEXER_PERF_MARGIN = 0.02
RING_INDEXER_EXPECTED_FPU_UTIL = {
    # (SP ranks, KV prefix): expected fused-program FPU utilization, percent.
    (4, GLM52_KV_55K): 37.16,
    (4, GLM52_KV_512K): 39.31,
    (8, GLM52_KV_55K): 38.48,
    (8, GLM52_KV_512K): 42.72,
}

_FABRIC_2D_TORUS_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "reliability_mode": ttnn.FabricReliabilityMode.STRICT_INIT,
    "fabric_tensix_config": ttnn.FabricTensixConfig.DISABLED,
    "require_exact_physical_num_devices": True,
}

# Keep this in lockstep with IndexerScoreDeviceOperation's Blackhole perf
# model. The profiler duration is measured in ns; its device clock is 1.35
# cycles/ns. LoFi is the deployed bfp8 Q/K path used by GLM-5.2.
_BH_CLOCK_GHZ = 1.35
_LOFI_MUL_ADDS_PER_CYCLE_PER_CORE = 4096


def _make_ccl_context(mesh_device):
    """Create the two ring-direction semaphores on the mesh's worker sub-device."""
    grid = mesh_device.compute_with_storage_grid_size()
    worker_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_subdevice_id = ttnn.SubDeviceId(0)
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([worker_cores])], 0)
    mesh_device.load_sub_device_manager(manager)
    mesh_device.set_sub_device_stall_group([worker_subdevice_id])
    semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(2)]
    return semaphores, worker_subdevice_id


def _clear_ccl_context(mesh_device):
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


def _ring_indexer_duration_ns(programs):
    """Return the critical path of the one fused program, identified by its defining kernels."""
    fused = [
        info["duration_ns"]
        for info in programs.values()
        if any("/experimental/indexer_score/" in source.replace("\\", "/") for source in info["kernel_sources"])
    ]
    assert len(fused) == 1, f"expected one fused ring-indexer program, found {len(fused)} in {list(programs)}"
    return fused[0]


def _largest_divisor_leq(value, cap):
    return max(divisor for divisor in range(1, min(value, cap) + 1) if value % divisor == 0)


def _ring_indexer_ideal_compute_cycles(mesh_device, kv_len, chunk_start):
    """Mirror the fused op's Blackhole ideal-compute-cycle performance model.

    This is intentionally test-local until op-performance-model fields are
    directly exposed to the realtime-profiler API. The calculation is
    fusion-aware: two links reserve four all-gather worker cores, so score
    math is credited only to the remaining compute rectangle.
    """
    q_tiles = GLM52_Q_PER_CHIP // 32
    k_tiles = GLM52_K_CACHE_CAPACITY // 32
    kv_tiles = kv_len // 32
    chunk_start_tiles = chunk_start // 32
    valid_tiles = sum(min(kv_tiles, chunk_start_tiles + row + 1) for row in range(q_tiles))

    # IndexerScoreProgramConfig(q_chunk=32, k_chunk=224) maps q groups by
    # grid rows and K bands by columns. This is the same banded_core_count()
    # arithmetic as the C++ program factory/perf model.
    q_groups = q_tiles
    k_bands = math.ceil(k_tiles / (GLM52_K_CHUNK // 32))
    grid = mesh_device.compute_with_storage_grid_size()
    ag_worker_cores = 2 * 2  # two directions for each of the two links
    compute_grid_x = grid.x - math.ceil(ag_worker_cores / grid.y)
    assert compute_grid_x > 0
    group_rows = _largest_divisor_leq(q_groups, grid.y)
    band_columns = min(k_bands, compute_grid_x)
    row_blocks = max(1, min(grid.y // group_rows, k_bands // band_columns))
    core_count = group_rows * row_blocks * band_columns

    num_mul_adds = 2 * valid_tiles * GLM52_INDEX_HEADS * (32 * 32) * GLM52_INDEX_DIM
    return math.ceil(num_mul_adds / (core_count * _LOFI_MUL_ADDS_PER_CYCLE_PER_CORE))


def _ring_perf_config():
    """The resident-head GLM-5.2 indexer-score program configuration."""
    return ttnn.IndexerScoreProgramConfig(
        # indexer.py chooses 32 when the 160-row TP shard is not divisible by 64.
        q_chunk_size=32,
        k_chunk_size=GLM52_K_CHUNK,
        head_group_size=0,
    )


@run_for_blackhole("ring_indexer_score_dsa perf requires Blackhole fabric")
@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("mesh_device", RING_PERF_MESHES, ids=RING_PERF_MESH_IDS, indirect=True)
@pytest.mark.parametrize("device_params", [_FABRIC_2D_TORUS_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("kv_len", RING_PERF_KV_LENS, ids=RING_PERF_KV_IDS)
@pytest.mark.timeout(0)
def test_ring_indexer_score_dsa_perf(mesh_device, kv_len):
    """Profile one warm fused ring score at a 55K or 512K KV prefix.

    The realtime profiler's fused-program critical path is converted to FPU
    utilization using a test-local mirror of the op's fusion-aware
    ideal-compute-cycle model. Each physical box has its own expected
    utilization and a +/-2% relative CI band, matching the existing
    single-chip indexer perf gate.
    """
    require_realtime_profiler("ring_indexer_score_dsa perf checks")

    sp, tp = mesh_device.shape
    assert (sp, tp) in RING_PERF_MESHES
    assert ttnn.get_num_devices() == sp * tp, "perf proxy must use the complete physical box"
    # The control plane consolidates torus axes no mesh realizes (a wrapped dimension needs more than
    # 2 devices), so the latched config may be a reduced torus flavor of the requested TORUS_XY.
    assert ttnn.get_fabric_config() in (
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        ttnn.FabricConfig.FABRIC_2D_TORUS_X,
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
    )
    assert kv_len % 32 == 0
    assert GLM52_Q_PER_SP_RANK == GLM52_Q_PER_CHIP * GLM52_TP

    # Match the GLM-5.2 fused-op inputs: 128-d BFP8 Q/K, BF16 gates, 32
    # resident index heads, and 160 post-TP query rows per physical chip. A
    # TP-free box runs one TP lane on every SP rank. The physical proxy has no
    # TP axis, so its cache uses the corresponding 160-row lane slab; the
    # production 640-row SP slab needs its real TP=4 mesh axis for the op's
    # exact 2-D block-cyclic geometry.
    q_rows = sp * GLM52_Q_PER_CHIP
    chunk_start = kv_len - q_rows
    q_host = torch.randn((1, GLM52_INDEX_HEADS, q_rows, GLM52_INDEX_DIM), dtype=torch.bfloat16)
    w_host = torch.randn((1, GLM52_INDEX_HEADS, q_rows, 1), dtype=torch.bfloat16)

    sp_shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(2, None))
    q_dev = ttnn.from_torch(
        q_host, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=sp_shard
    )
    w_dev = ttnn.from_torch(
        w_host, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=sp_shard
    )
    # The production GLM-5.2 cache is 1 Mi tokens wide, ND-sharded in DRAM,
    # and has one slot per full indexer layer. The fused program is compiled
    # from this *capacity* (not kv_len): kv_len bounds the populated prefix,
    # but leaves the capacity-sized K-band schedule intact. Selecting slot 20
    # matches the final full GLM-5.2 indexer layer.
    k_local = init_kvpe_cache(
        kvpe_cache_head_dim=GLM52_INDEX_DIM,
        mesh_device=mesh_device,
        seq_len=GLM52_K_CACHE_CAPACITY,
        mesh_shape=(sp, tp),
        sp_axis=0,
        num_kvpe_cache_layers=GLM52_INDEX_CACHE_SLOTS,
        num_users=1,
        dtype=ttnn.bfloat8_b,
    )
    # The all-gather output is persistent and full-width on every rank.  It is
    # zero-seeded because only shape/route/device timing matters here.
    k_gathered = ttnn.from_torch(
        torch.zeros((1, 1, GLM52_K_CACHE_CAPACITY, GLM52_INDEX_DIM), dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    semaphores, subdevice_id = _make_ccl_context(mesh_device)
    trace_id = None
    trace_capture_ended = False
    trace_out = None
    try:

        def run_once():
            return ttnn.experimental.ring_indexer_score_dsa(
                q_dev,
                k_gathered,
                w_dev,
                k_local,
                semaphores,
                cluster_axis=0,
                topology=ttnn.Topology.Ring,
                num_links=2,
                ag_sub_device_id=subdevice_id,
                chunk_start_idx=chunk_start,
                kv_len=kv_len,
                cache_batch_idx=GLM52_INDEX_CACHE_SLOT,
                block_cyclic_sp_axis=0,
                block_cyclic_chunk_local=GLM52_Q_PER_CHIP,
                program_config=_ring_perf_config(),
            )

        # Compile/cache warm-up is deliberately outside the profiler window.
        warmup = run_once()
        ttnn.synchronize_device(mesh_device, sub_device_ids=[subdevice_id])
        ttnn.deallocate(warmup)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_out = run_once()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        trace_capture_ended = True
        ttnn.synchronize_device(mesh_device, sub_device_ids=[subdevice_id])

        # Warm the trace replay outside the profiler window, then measure the
        # identical captured command stream without first-replay jitter.
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device, sub_device_ids=[subdevice_id])

        def replay_once():
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

        _, programs = profile_realtime_program_merged(mesh_device, replay_once, record_timeout_seconds=30.0)
        duration_ns = _ring_indexer_duration_ns(programs)
        ideal_compute_cycles = _ring_indexer_ideal_compute_cycles(mesh_device, kv_len, chunk_start)
        fpu_utilization = ideal_compute_cycles / (duration_ns * _BH_CLOCK_GHZ) * 100
        expected_fpu_utilization = RING_INDEXER_EXPECTED_FPU_UTIL[(sp, kv_len)]
        lower = expected_fpu_utilization * (1 - RING_INDEXER_PERF_MARGIN)
        upper = expected_fpu_utilization * (1 + RING_INDEXER_PERF_MARGIN)
        logger.info(
            "ring_indexer_score_dsa perf: mesh={} fabric={} topology=ring heads={} k_capacity={} kv_len={} "
            "q_per_chip={} duration={:.3f} ms, fpu_util={:.2f}% (expected {:.2f}%, band [{:.2f}, {:.2f}])".format(
                tuple(mesh_device.shape),
                ttnn.get_fabric_config(),
                GLM52_INDEX_HEADS,
                GLM52_K_CACHE_CAPACITY,
                kv_len,
                GLM52_Q_PER_CHIP,
                duration_ns / 1e6,
                fpu_utilization,
                expected_fpu_utilization,
                lower,
                upper,
            )
        )
        assert duration_ns > 0, "fused ring-indexer profiler duration must be positive"
        assert lower <= fpu_utilization <= upper, (
            f"ring_indexer_score_dsa mesh={(sp, tp)} kv_len={kv_len}: FPU utilization "
            f"{fpu_utilization:.2f}% outside band [{lower:.2f}, {upper:.2f}] "
            f"(expected {expected_fpu_utilization:.2f}%, margin +/- {RING_INDEXER_PERF_MARGIN * 100:.1f}%)"
        )
    finally:
        if trace_id is not None:
            try:
                if not trace_capture_ended:
                    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            finally:
                ttnn.release_trace(mesh_device, trace_id)
        if trace_out is not None:
            ttnn.deallocate(trace_out)
        _clear_ccl_context(mesh_device)
