# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device kernel time: today's SP-only ring MLA against the deduped full-mesh split-KV variant.

Both legs run the identical logical problem -- one Q chunk of a chunked prefill attending to a
filled KV prefix -- with the identical per-device Q shape, on the identical SDPA core grid, over
the identical 2D torus fabric. Only the KV distribution and the gather's reach differ:

  SP-only (production today)  KV sharded over SP=8, replicated across TP=4.
                              Gather on the SP axis only, ring_size 8.
                              Every TP lane keeps its own copy of the whole SP-sharded cache.

  Full-mesh split-KV          KV sharded over all 32 devices, no TP replication (the dedup).
                              Gather fused over the full mesh, ring_size 32.
                              Per-device cache is 1/TP the size; per layer, across 61 layers.

Both legs run FABRIC_2D_TORUS_XY so the fabric is not a variable, and both hold their cache in the
same block-cyclic layout -- source s owns global regions s, s+ranks, s+2*ranks... The only
difference is (ranks, region): the SP leg is (8, q_slab), the split leg is (32, q_slab / TP).
Same function builds both, so the layouts cannot drift apart.

Cache capacity is always the full preallocation: production does not right-size the KV cache per
chunk, it allocates for max context once and fills a prefix. kv_actual_isl is what tells the op to
gather and attend only the filled part -- without it compute_gather_valid_Ht returns the whole
input extent and the op would move the entire allocation.

Only ring_mla is dispatched inside the measured region -- no surrounding model.

These are measurement reports, not perf gates: they assert the legs really ran the geometry they
claim and print the numbers. There is no committed baseline for the fused leg yet.
"""

import statistics
from contextlib import contextmanager
from dataclasses import dataclass, replace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_with_llk_assert, skip_with_watcher
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    CHUNKED_PREFILL_N_CHUNKS,
    CHUNKED_PREFILL_PER_DEVICE_CHUNK,
    CHUNKED_PREFILL_SEED,
    MESH_CONFIG,
    RING_MLA_CHUNKED_MODEL_CONFIGS,
    _make_ring_mla_metadata,
    _ring_mla_host_scalar_tensor,
    close_ring_joint_sdpa_runtime,
    open_ring_joint_sdpa_runtime,
)
from tests.ttnn.profiling.realtime_profiler_utils import (
    profile_realtime_program_merged,
    require_realtime_profiler,
)

# Discard the first dispatches of each measurement point (program build / cache warm) and report
# the median of the rest, so one scheduling outlier cannot decide a comparison.
WARMUP_DISPATCHES = 2
MEASURED_DISPATCHES = 7

# One captured ring_mla is small; the split-KV op suite sizes its own replay region at 4 MB. 16 MB is
# ample for the one capture live at a time here, plus its runtime args.
TRACE_REGION_SIZE = 16 * 1024 * 1024

# Cache preallocation, in whole prefill chunks. Both layouts need capacity to be a whole number of
# global chunks -- the SP cache in Q-sized slabs, the split cache in whole block-cyclic regions
# (RingMLAGeometry::valid rejects a partial region) -- so capacity is counted in chunks, not tokens.
# 205 chunks == 1_049_600 tokens: the smallest whole-chunk capacity at or above 2^20, and enough to
# hold the deepest prefix of the sweep below plus its current chunk.
CACHE_CHUNKS = 205

# Prefix sweep, in whole prefill chunks. A prefill advances one chunk at a time, so the chunk is the
# quantum a real prefix moves in; it is also tile-aligned by construction, which kv_actual_isl
# requires. Fine steps of one chunk (~5k tokens) to 12 chunks (~61k), then a coarse ladder to ~1M.
FINE_PREFIX_CHUNKS = tuple(range(0, 13))  # 0 .. 61_440 tokens
COARSE_PREFIX_CHUNKS = (25, 50, 100, 200)  # 128_000, 256_000, 512_000, 1_024_000 tokens
PREFIX_CHUNKS = FINE_PREFIX_CHUNKS + COARSE_PREFIX_CHUNKS


@dataclass(frozen=True)
class SplitKVPerfGeometry:
    """Both legs share every field here; they disagree only on where the KV rows live."""

    sp: int  # Q sequence shards == SP gather depth
    tp: int  # head shards == the extra KV stripe factor the fused leg exploits
    q_slab: int  # per-device Q rows, identical in both layouts
    chunk: int  # global Q rows per prefill chunk
    cache_chunks: int  # prefill chunks the cache is allocated for
    nhq_total: int
    d_k: int
    d_v: int
    kv_dtype: ttnn.DataType

    @property
    def ranks(self) -> int:
        return self.sp * self.tp

    @property
    def cache_capacity(self) -> int:
        return self.chunk * self.cache_chunks

    @property
    def sp_region(self) -> int:
        """SP-only leg: one Q-sized KV slab per source per chunk."""
        return self.q_slab

    @property
    def split_region(self) -> int:
        """Fused leg: the Q slab split across TP lanes."""
        return self.q_slab // self.tp

    @property
    def sp_rows_per_device(self) -> int:
        return self.cache_capacity // self.sp

    @property
    def split_rows_per_device(self) -> int:
        return self.cache_capacity // self.ranks

    def logical_n(self, prefix: int) -> int:
        """The current chunk sits immediately after the filled prefix."""
        return min(prefix + self.chunk, self.cache_capacity)


def build_geometry(model, cache_chunks=CACHE_CHUNKS) -> SplitKVPerfGeometry:
    sp, tp = MESH_CONFIG.sp_size, MESH_CONFIG.tp_size
    return SplitKVPerfGeometry(
        sp=sp,
        tp=tp,
        q_slab=CHUNKED_PREFILL_PER_DEVICE_CHUNK,
        chunk=CHUNKED_PREFILL_PER_DEVICE_CHUNK * sp,
        cache_chunks=cache_chunks,
        # model.nhq is per ring; every TP lane holds its own head shard.
        nhq_total=model.nhq * tp,
        d_k=model.d_k,
        d_v=model.d_v,
        kv_dtype=model.kv_dtype,
    )


def build_host_inputs(geometry: SplitKVPerfGeometry):
    """One Q chunk and a fully populated K/V latent cache, shared verbatim by both legs."""
    torch.manual_seed(CHUNKED_PREFILL_SEED)
    q = torch.randn(1, geometry.nhq_total, geometry.chunk, geometry.d_k, dtype=torch.bfloat16)
    cache = torch.randn(1, 1, geometry.cache_capacity, geometry.d_k, dtype=torch.bfloat16)
    return q, cache


def block_cyclic_cache(src, ranks, region):
    """Source s owns global regions s, s+ranks, s+2*ranks, ... laid out contiguously in its shard.

    This is one function for both legs. At (ranks=SP, region=q_slab) it reproduces the growing
    balanced chunked layout the SP path already uses (to_balanced_growing_cache_layout: device d
    holds global region chunk*sp + d at local slab chunk). At (ranks=SP*TP, region=q_slab/TP) it is
    the deduped split-KV placement, the inverse of RingMLAGeometry::global_k_tile.
    """
    batch, _, rows, width = src.shape
    regions = src.reshape(batch, rows // region, region, width)
    per_source = regions.shape[1] // ranks
    shards = [regions[:, source::ranks].reshape(batch, per_source * region, width) for source in range(ranks)]
    return torch.stack(shards, dim=1).reshape(batch, 1, ranks * per_source * region, width)


@contextmanager
def captured_trace(mesh_device, call):
    """Warm up eagerly, capture ONE trace, yield its replay. Mirrors the model: a chunked prefill
    captures the forward once and replays it per chunk. Capture needs the program already compiled
    and in the cache, which the warm-up provides."""
    for _ in range(WARMUP_DISPATCHES):
        call()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        yield lambda: ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    finally:
        ttnn.release_trace(mesh_device, trace_id)


def measure_replays(mesh_device, replay):
    """Mean total device kernel time per replay, from the real-time profiler.

    Sums across every program a dispatch produced rather than taking the first, so a leg that needed
    a second program could not hide half its cost. Each program's own duration is the max across
    chips -- its critical path through the mesh.
    """
    # Checked here rather than at test entry: the query needs an open device.
    require_realtime_profiler("ring_mla split-KV kernel-time comparison")
    per_dispatch, program_counts, kernel_sources = [], set(), set()
    for _ in range(MEASURED_DISPATCHES):
        _, per_program = profile_realtime_program_merged(mesh_device, replay)
        per_dispatch.append(sum(entry["duration_ns"] for entry in per_program.values()))
        program_counts.add(len(per_program))
        for entry in per_program.values():
            kernel_sources.update(entry["kernel_sources"])
    return {
        "mean_ns": statistics.mean(per_dispatch),
        "median_ns": statistics.median(per_dispatch),
        "min_ns": min(per_dispatch),
        "max_ns": max(per_dispatch),
        "programs": max(program_counts),
        "kernel_sources": kernel_sources,
    }


def measure_dispatch_ns(mesh_device, call):
    """One capture, measured. For a single operating point; a sweep should hold the capture open."""
    with captured_trace(mesh_device, call) as replay:
        return measure_replays(mesh_device, replay)


def assert_finite_output(output, context):
    """Cheap guard that a measured dispatch computed attention rather than NaNs. One shard is enough
    here -- test_ring_mla_fused_gather.py owns the per-rank numerical coverage."""
    shard = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).float()
    assert torch.isfinite(shard).all(), f"non-finite output from the measured dispatch ({context})"


def replicated_gather_buffer(mesh, geometry: SplitKVPerfGeometry):
    """The persistent gathered-KV scratch. Sized to the whole allocation in both legs (the legacy
    layout is validated equal to it); kv_actual_isl is what keeps the gather itself short."""
    return ttnn.from_torch(
        torch.zeros(1, 1, geometry.cache_capacity, geometry.d_k, dtype=torch.bfloat16),
        dtype=geometry.kv_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def make_prefix_control(mesh, geometry, use_metadata):
    """How the per-chunk prefix reaches the op, and how a sweep moves it.

    Scalar: kv_actual_isl is a host arg, so it is BAKED INTO a capture and every prefix needs its
    own. Metadata: the op reads the prefix on-device from a tensor, so ONE capture serves every
    prefix and the sweep refreshes the tensor in place between replays -- which is exactly what the
    model does (ttMLA._chunked_attn's metadata path + copy_host_to_device_tensor per chunk).
    """
    if not use_metadata:
        state = {"prefix": 0}

        def kwargs_for():
            prefix = state["prefix"]
            return {"logical_n": geometry.logical_n(prefix), "kv_actual_isl": prefix, "kv_cache_batch_idx": 0}

        def set_prefix(prefix):
            state["prefix"] = prefix

        return kwargs_for, set_prefix, False

    slot, prefix_tensor = _make_ring_mla_metadata(mesh, 0, 0)

    def kwargs_for():
        # logical_n is a placeholder on this path: every kernel derives it on-device from the
        # prefix, which is what makes one capture valid for all of them. Matches the model, which
        # passes the global cache capacity here.
        return {
            "logical_n": geometry.cache_capacity,
            "slot_id": slot,
            "kv_actual_isl_tensor": prefix_tensor,
            "kv_cache_num_layers": 1,
            "kv_cache_layer_idx": 0,
        }

    def set_prefix(prefix):
        ttnn.copy_host_to_device_tensor(_ring_mla_host_scalar_tensor(mesh, prefix), prefix_tensor)

    return kwargs_for, set_prefix, True


def ring_mla_call(runtime, geometry, tt_q, tt_k, scratch, q_chunk, k_chunk, cluster_axis, kwargs_for):
    """A dispatch whose per-chunk scalars come from `kwargs_for` at call time, so a sweep can move
    the prefix without rebuilding anything."""

    def call():
        output, _ = ttnn.transformer.ring_mla(
            tt_q,
            tt_k,
            persistent_output_buffer_kv=scratch,
            head_dim_v=geometry.d_v,
            is_balanced=False,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=runtime.sdpa_compute_grid,
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            ),
            compute_kernel_config=runtime.compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=runtime.ccl_semaphore_handles,
            num_links=runtime.num_links,
            cluster_axis=cluster_axis,
            mesh_device=runtime.mesh_device,
            topology=runtime.topology,
            subdevice_id=runtime.worker_sub_device_id,
            ccl_core_grid_offset=(runtime.ccl_column, 0),
            use_column_major_ccl=True,
            **kwargs_for(),
        )
        return output

    return call


@contextmanager
def sp_only_leg(
    geometry: SplitKVPerfGeometry, q_full, cache_full, q_chunk, k_chunk, trace_region_size=0, use_metadata=False
):
    """Today's Kimi path: gather on the SP axis, KV replicated across TP."""
    runtime = open_ring_joint_sdpa_runtime(
        MESH_CONFIG, fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_XY, trace_region_size=trace_region_size
    )
    try:
        mesh = runtime.mesh_device
        assert tuple(mesh.shape) == (geometry.tp, geometry.sp)
        mesh.enable_program_cache()

        tt_q = ttnn.from_torch(
            q_full,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh, mesh_shape=(geometry.tp, geometry.sp), dims=[1, 2]  # axis0=TP: heads, axis1=SP: seq
            ),
        )
        tt_k = ttnn.from_torch(
            block_cyclic_cache(cache_full, geometry.sp, geometry.sp_region),
            dtype=geometry.kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh, mesh_shape=(geometry.tp, geometry.sp), dims=[None, 2]  # KV replicated across TP
            ),
        )
        scratch = replicated_gather_buffer(mesh, geometry)
        assert tuple(ttnn.get_device_tensors(tt_k)[0].shape)[2] == geometry.sp_rows_per_device

        kwargs_for, set_prefix, one_capture = make_prefix_control(mesh, geometry, use_metadata)
        yield mesh, ring_mla_call(
            runtime, geometry, tt_q, tt_k, scratch, q_chunk, k_chunk, runtime.sp_axis, kwargs_for
        ), set_prefix, one_capture
    finally:
        close_ring_joint_sdpa_runtime(runtime)


@contextmanager
def split_kv_leg(
    geometry: SplitKVPerfGeometry, q_full, cache_full, q_chunk, k_chunk, trace_region_size=0, use_metadata=False
):
    """Deduped cache, gather fused across the whole mesh."""
    # open_ring_joint_sdpa_runtime builds MeshShape(tp_size, sp_size); the fused layout wants Q
    # sequence on mesh axis 0, so hand it the axes swapped. full_mesh already defaults the fabric to
    # FABRIC_2D_TORUS_XY -- the same fabric the SP leg is pinned to above.
    config = replace(MESH_CONFIG, tp_size=geometry.sp, sp_size=geometry.tp)
    runtime = open_ring_joint_sdpa_runtime(config, full_mesh=True, trace_region_size=trace_region_size)
    try:
        mesh = runtime.mesh_device
        assert tuple(mesh.shape) == (geometry.sp, geometry.tp)
        mesh.enable_program_cache()

        tt_q = ttnn.from_torch(
            q_full,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh, mesh_shape=(geometry.sp, geometry.tp), dims=[2, 1]  # axis0=SP: seq, axis1=TP: heads
            ),
        )
        tt_k = ttnn.from_torch(
            block_cyclic_cache(cache_full, geometry.ranks, geometry.split_region),
            dtype=geometry.kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
        )
        scratch = replicated_gather_buffer(mesh, geometry)
        # The dedup itself: 1/TP the cache rows per device, same global KV.
        assert tuple(ttnn.get_device_tensors(tt_k)[0].shape)[2] == geometry.split_rows_per_device

        kwargs_for, set_prefix, one_capture = make_prefix_control(mesh, geometry, use_metadata)
        yield mesh, ring_mla_call(
            runtime, geometry, tt_q, tt_k, scratch, q_chunk, k_chunk, None, kwargs_for
        ), set_prefix, one_capture
    finally:
        close_ring_joint_sdpa_runtime(runtime)


def sweep_prefixes(mesh, call, set_prefix, one_capture, prefixes, label):
    """Measure one leg across every prefix, reusing a single uploaded tensor set and program.

    one_capture (metadata path) holds ONE captured trace open for the whole sweep and refreshes the
    prefix tensor between replays -- the model's own loop. Otherwise the prefix is baked into the
    capture and each point captures and releases its own.
    """
    results = {}
    entries_after_warm = None

    def point(prefix, measure):
        nonlocal entries_after_warm
        results[prefix] = measure()
        if entries_after_warm is None:
            entries_after_warm = mesh.num_program_cache_entries()
        else:
            # The prefix is a runtime-patched scalar (or read on-device). If a point minted a new
            # program, the sweep would be timing compiles rather than prefix depth.
            assert mesh.num_program_cache_entries() == entries_after_warm, (
                f"{label} prefix={prefix} added a program "
                f"({mesh.num_program_cache_entries()} vs {entries_after_warm}); "
                "the sweep no longer isolates prefix depth"
            )

    if one_capture:
        set_prefix(prefixes[0])
        with captured_trace(mesh, call) as replay:
            for prefix in prefixes:
                set_prefix(prefix)
                point(prefix, lambda: measure_replays(mesh, replay))
    else:
        for prefix in prefixes:
            set_prefix(prefix)
            point(prefix, lambda: measure_dispatch_ns(mesh, call))

    set_prefix(prefixes[-1])
    assert_finite_output(call(), f"{label} prefix={prefixes[-1]}")
    return results


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "model_name, q_chunk, k_chunk",
    # kimi50k carries the Kimi-K2.6/K2.7 attention shape (64 heads = 16/ring x TP=4, latent 576/512;
    # K2.6 and K2.7 are architecturally identical). kimi_k3 is Kimi-K3: same latent geometry at 96
    # heads. Named after the configs, not the model versions, so the ids cannot drift from them.
    [("kimi50k", 32, 640), ("kimi_k3", 32, 640)],
    ids=["kimi50k-q32-k640", "kimi_k3-q32-k640"],
)
# The model drives the prefix through metadata under trace; scalar is the non-traced deployment and
# the control that shows the on-device extent recompute costs nothing.
@pytest.mark.parametrize("use_metadata", [True, False], ids=["metadata", "scalar"])
@skip_with_llk_assert("No need to verify LLK asserts for performance measurements.")
@skip_with_watcher("Watcher perturbs kernel timing; perf measurements are not meaningful with it enabled.")
def test_ring_mla_split_kv_prefix_depth_sweep(model_name, q_chunk, k_chunk, use_metadata):
    """Kernel time of both KV distributions as the filled prefix grows from empty to ~1M tokens.

    The cache allocation is fixed at CACHE_CHUNKS for every point, so what moves is only how much
    of it kv_actual_isl declares filled. That isolates the thing worth knowing: whether the fused
    leg's overhead is a fixed cost or one that scales with the prefix.
    """
    if not MESH_CONFIG.is_galaxy:
        pytest.skip(f"Split-KV comparison needs the 32-device Galaxy (SP=8, TP=4), got {MESH_CONFIG.num_devices}")

    model = RING_MLA_CHUNKED_MODEL_CONFIGS[model_name]
    geometry = build_geometry(model)
    prefixes = [chunks * geometry.chunk for chunks in PREFIX_CHUNKS]
    assert geometry.logical_n(prefixes[-1]) <= geometry.cache_capacity
    q_full, cache_full = build_host_inputs(geometry)

    with sp_only_leg(geometry, q_full, cache_full, q_chunk, k_chunk, TRACE_REGION_SIZE, use_metadata) as (
        mesh,
        call,
        set_prefix,
        one_capture,
    ):
        sp_results = sweep_prefixes(mesh, call, set_prefix, one_capture, prefixes, "SP-only")
    with split_kv_leg(geometry, q_full, cache_full, q_chunk, k_chunk, TRACE_REGION_SIZE, use_metadata) as (
        mesh,
        call,
        set_prefix,
        one_capture,
    ):
        split_results = sweep_prefixes(mesh, call, set_prefix, one_capture, prefixes, "full-mesh")

    rows = [
        "",
        f"ring_mla prefix-depth sweep -- {model_name} (q{q_chunk}/k{k_chunk}), FABRIC_2D_TORUS_XY on both legs, captured trace "
        f"({'one capture, prefix via metadata' if use_metadata else 'one capture per prefix, scalar'}), "
        f"mean of {MEASURED_DISPATCHES} replays",
        f"  cache      {geometry.cache_capacity} tokens allocated on every point "
        f"({geometry.sp_rows_per_device} rows/device SP-only, {geometry.split_rows_per_device} split-KV)",
        f"  per device Q {geometry.q_slab} rows x {geometry.nhq_total // geometry.tp} heads, "
        f"d_k={geometry.d_k}, latent d_v={geometry.d_v}",
        "",
        f"  {'prefix':>9} {'logical_n':>10} | {'SP-only':>9} {'full-mesh':>10} {'ratio':>7} {'delta':>9}",
        f"  {'-'*9} {'-'*10} | {'-'*9} {'-'*10} {'-'*7} {'-'*9}",
    ]
    for prefix in prefixes:
        sp_ms = sp_results[prefix]["mean_ns"] / 1e6
        split_ms = split_results[prefix]["mean_ns"] / 1e6
        rows.append(
            f"  {prefix:>9} {geometry.logical_n(prefix):>10} | "
            f"{sp_ms:>8.3f}m {split_ms:>9.3f}m {split_ms/sp_ms:>7.3f} {split_ms-sp_ms:>+8.3f}m"
        )
    logger.info("\n".join(rows))

    # Invariants only: this reports numbers, it does not gate them.
    assert set(sp_results) == set(split_results) == set(prefixes)
    assert all(result["mean_ns"] > 0 for result in list(sp_results.values()) + list(split_results.values()))
    # Attention against a longer prefix cannot be cheaper; a flat or falling curve would mean the
    # prefix never reached the kernels and the sweep measured nothing.
    for results, label in ((sp_results, "SP-only"), (split_results, "full-mesh")):
        shallow, deep = results[prefixes[0]]["mean_ns"], results[prefixes[-1]]["mean_ns"]
        assert deep > shallow, f"{label} did not grow with prefix depth ({shallow} -> {deep} ns)"


@pytest.mark.timeout(2400)
@pytest.mark.parametrize(
    "model_name, q_chunk, k_chunk",
    [("kimi50k", 32, 640), ("kimi_k3", 32, 640)],
    ids=["kimi50k-q32-k640", "kimi_k3-q32-k640"],
)
@pytest.mark.parametrize("cache_chunks", [CHUNKED_PREFILL_N_CHUNKS, CACHE_CHUNKS], ids=["exact_cache", "prealloc_1m"])
@skip_with_llk_assert("No need to verify LLK asserts for performance measurements.")
@skip_with_watcher("Watcher perturbs kernel timing; perf measurements are not meaningful with it enabled.")
def test_ring_mla_split_kv_vs_sp_only_kernel_time(model_name, q_chunk, k_chunk, cache_chunks):
    """The production point: the final chunk of an 11-chunk prefill, at exact vs preallocated
    capacity. The exact leg is the control that keeps an oversized-allocation effect from being
    mistaken for a KV-distribution effect."""
    if not MESH_CONFIG.is_galaxy:
        pytest.skip(f"Split-KV comparison needs the 32-device Galaxy (SP=8, TP=4), got {MESH_CONFIG.num_devices}")

    model = RING_MLA_CHUNKED_MODEL_CONFIGS[model_name]
    geometry = build_geometry(model, cache_chunks=cache_chunks)
    prefix = (CHUNKED_PREFILL_N_CHUNKS - 1) * geometry.chunk
    q_full, cache_full = build_host_inputs(geometry)

    region = TRACE_REGION_SIZE
    with sp_only_leg(geometry, q_full, cache_full, q_chunk, k_chunk, region) as (mesh, call, set_prefix, _):
        set_prefix(prefix)
        assert_finite_output(call(), "SP-only")
        sp_only = measure_dispatch_ns(mesh, call)
    with split_kv_leg(geometry, q_full, cache_full, q_chunk, k_chunk, region) as (mesh, call, set_prefix, _):
        set_prefix(prefix)
        assert_finite_output(call(), "full-mesh")
        split_kv = measure_dispatch_ns(mesh, call)

    ratio = split_kv["mean_ns"] / sp_only["mean_ns"]
    cache_ratio = geometry.sp_rows_per_device / geometry.split_rows_per_device
    logical_n = geometry.logical_n(prefix)

    logger.info(
        f"\nring_mla KV-distribution comparison -- {model_name} (q{q_chunk}/k{k_chunk})\n"
        f"  fabric        FABRIC_2D_TORUS_XY on both legs\n"
        f"  dispatch      captured trace, mean of {MEASURED_DISPATCHES} replays "
        f"(after {WARMUP_DISPATCHES} eager warm-up)\n"
        f"  workload      {geometry.chunk} Q rows against logical_n={logical_n}, kv_actual_isl={prefix}, "
        f"{geometry.nhq_total} Q heads, d_k={geometry.d_k}, latent d_v={geometry.d_v}\n"
        f"  per device    Q {geometry.q_slab} rows x {geometry.nhq_total // geometry.tp} heads "
        f"(identical in both legs)\n"
        f"  cache         {geometry.cache_capacity} tokens allocated, {logical_n} filled "
        f"({100.0 * logical_n / geometry.cache_capacity:.1f}%)\n"
        f"  SP-only       ring {geometry.sp:2d}, KV {geometry.sp_rows_per_device:6d} rows/device, "
        f"{sp_only['mean_ns']/1e6:8.3f} ms  "
        f"[{sp_only['min_ns']/1e6:.3f}, {sp_only['max_ns']/1e6:.3f}], {sp_only['programs']} program(s)\n"
        f"  full-mesh     ring {geometry.ranks:2d}, KV {geometry.split_rows_per_device:6d} rows/device, "
        f"{split_kv['mean_ns']/1e6:8.3f} ms  "
        f"[{split_kv['min_ns']/1e6:.3f}, {split_kv['max_ns']/1e6:.3f}], {split_kv['programs']} program(s)\n"
        f"  result        kernel time {ratio:.3f}x  |  KV cache per device {cache_ratio:.2f}x smaller"
    )

    assert cache_ratio == geometry.tp
    assert sp_only["mean_ns"] > 0 and split_kv["mean_ns"] > 0
    assert any("ring_joint" in source for source in sp_only["kernel_sources"] | split_kv["kernel_sources"]), (
        f"measured programs are not ring_joint SDPA: "
        f"{sorted(sp_only['kernel_sources'] | split_kv['kernel_sources'])}"
    )
