# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Op-level split-KV MLA: independent TP heads and poisoned fixed-capacity caches."""

import math
import os
from dataclasses import replace
from unittest import mock

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_high_power
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    CHUNKED_PREFILL_CHUNK_ID_ENV,
    CHUNKED_PREFILL_CHUNK_SIZE,
    CHUNKED_PREFILL_TOTAL_SEQ,
    MESH_CONFIG,
    RING_MLA_CHUNKED_MODEL_CONFIGS,
    RING_MLA_CHUNKED_PERF_CHECK_CONFIGS,
    _make_ring_mla_metadata,
    _ring_mla_host_scalar_tensor,
    close_ring_joint_sdpa_runtime,
    compute_chunked_prefill_perf_check_utilization,
    open_ring_joint_sdpa_runtime,
    profile_ring_joint_runtime_duration_ns,
    run_ring_joint_sdpa_chunked,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# Gather scratch rows, deliberately larger than every case's input KV capacity so
# the op cannot rely on scratch and cache capacities matching.
OVERSIZED_SCRATCH_ROWS = 1 << 20


# Each case covers a distinct layout or dispatch boundary. The replay tests
# below cover additional prefix depths without recompiling every combination.
@pytest.mark.parametrize(
    "depth,capacity_depth,prefix_offset,metadata,q_chunk,k_chunk,local_heads",
    [
        pytest.param(
            depth, capacity, offset, metadata, 32, 352, 4, id=f"{mode}-depth{depth}-capacity{capacity}-offset{offset}"
        )
        for metadata, mode in ((False, "scalar"), (True, "metadata"))
        for depth, capacity, offset in ((1, 1, 0), (1, 8, 0), (3, 8, 0), (5, 8, 0), (1, 8, 32), (2, 8, 288), (8, 8, 32))
    ]
    + [
        pytest.param(
            depth, capacity, 0, None, q_chunk, k_chunk, 4, id=f"implicit-q{q_chunk}-depth{depth}-capacity{capacity}"
        )
        for depth, capacity, q_chunk, k_chunk in ((1, 1, 32, 352), (1, 8, 32, 352), (3, 8, 32, 352), (5, 8, 64, 640))
    ]
    + [
        pytest.param(depth, 8, offset, metadata, 64, 640, 4, id=f"materialized-{mode}-depth{depth}-offset{offset}")
        for metadata, mode in ((False, "scalar"), (True, "metadata"))
        for depth, offset in ((1, 0), (2, 288), (8, 32))
    ]
    + [
        pytest.param(1, 8, 0, True, 32, 32, 4, id="single-tile-k-first"),
        pytest.param(3, 8, 32, False, 32, 32, 4, id="single-tile-k-rotated"),
        pytest.param(1, 8, 0, False, 32, 640, 32, id="many-heads-first"),
        pytest.param(3, 8, 288, True, 32, 640, 32, id="many-heads-rotated"),
    ],
)
def test_ring_mla_split_kv_geometry(depth, capacity_depth, prefix_offset, metadata, q_chunk, k_chunk, local_heads):
    run_ring_mla_split_kv_geometry(depth, capacity_depth, prefix_offset, metadata, q_chunk, k_chunk, local_heads)


def run_ring_mla_split_kv_geometry(
    depth,
    capacity_depth,
    prefix_offset,
    metadata,
    q_chunk,
    k_chunk,
    local_heads,
    trace_replay=False,
    effective_n=None,
    indexed=False,
    production=False,
    ordinary_mesh=False,
    scalar_replay=False,
    structural_cache=False,
    segmented=False,
):
    mesh_config = MESH_CONFIG
    if mesh_config.num_devices not in (8, 32):
        pytest.skip("Split-KV coverage requires an eight-device LB or 32-device Galaxy")
    tp = 4
    sp = mesh_config.num_devices // tp
    # Existing runtime helper names axes (TP, SP); here explicitly open (SP, TP).
    config = replace(mesh_config, tp_size=sp, sp_size=tp)
    runtime = open_ring_joint_sdpa_runtime(
        config,
        full_mesh=True,
        fabric_config=ttnn.FabricConfig.FABRIC_2D if ordinary_mesh else None,
        trace_region_size=4 * 1024 * 1024 if trace_replay else 0,
    )
    trace_ids = []
    try:
        mesh = runtime.mesh_device
        assert tuple(mesh.shape) == (sp, tp)
        assert mesh.get_num_devices() == sp * tp
        mesh.enable_program_cache()
        region, d_k, d_v = (160, 576, 512) if production else (64, 64, 32)
        kv_dtype = ttnn.bfloat8_b if production else ttnn.bfloat16
        ranks = sp * tp
        q_slab = tp * region
        chunk = ranks * region
        actual_isl = (depth - 1) * chunk + prefix_offset
        source_capacity = capacity_depth * region
        input_capacity = ranks * source_capacity
        live_end = min(actual_isl + chunk, input_capacity) if effective_n is None else effective_n
        # Equal-shard legacy layout requires exact-capacity scratch. Keep that
        # spec fixed across both layouts in the structural cache-key test.
        scratch_capacity = input_capacity if structural_cache else OVERSIZED_SCRATCH_ROWS
        torch.manual_seed(20260918)
        # Quantize before constructing the independent reference. Every TP lane
        # owns different heads, so an incorrect Q-rank mapping cannot hide behind
        # identical replicated inputs.
        q = torch.randn(1, local_heads * tp, chunk, d_k).bfloat16()
        num_layers = 2 if indexed else 1
        num_users = 3 if indexed else 1
        cache_batch = num_users * num_layers
        k = torch.randn(cache_batch, 1, live_end, d_k).bfloat16()
        cache_sources = torch.full((cache_batch, ranks, source_capacity, d_k), 17.0, dtype=torch.bfloat16)
        for global_region in range(math.ceil(live_end / region)):
            source = global_region % ranks
            local_start = (global_region // ranks) * region
            count = min(region, live_end - global_region * region)
            cache_sources[:, source, local_start : local_start + count] = k[
                :, 0, global_region * region : global_region * region + count
            ]
        tt_q = ttnn.from_torch(
            q,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=[2, 1]),
        )
        tt_k = ttnn.from_torch(
            cache_sources.reshape(cache_batch, 1, input_capacity, d_k),
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
        )
        scratch = ttnn.from_torch(
            torch.full((1, 1, scratch_capacity, d_k), 13.0, dtype=torch.bfloat16),
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        if production:
            # Reconstruct the logical reference from the quantized device input;
            # bf16-only references can hide conversion error in block-float KV.
            quantized_sources = torch.stack(
                [ttnn.to_torch(shard)[:, 0] for shard in ttnn.get_device_tensors(tt_k)], dim=1
            )
            for global_region in range(math.ceil(live_end / region)):
                source = global_region % ranks
                local_start = (global_region // ranks) * region
                count = min(region, live_end - global_region * region)
                k[:, 0, global_region * region : global_region * region + count] = quantized_sources[
                    :, source, local_start : local_start + count
                ]
            cache_sources = quantized_sources
        kwargs = dict(
            persistent_output_buffer_kv=scratch,
            head_dim_v=d_v,
            kv_cache_num_layers=num_layers,
            kv_cache_layer_idx=0,
            logical_n=input_capacity if metadata else live_end,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=runtime.sdpa_compute_grid,
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
                segmented_accumulation=segmented,
            ),
            compute_kernel_config=runtime.compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=runtime.ccl_semaphore_handles,
            num_links=runtime.num_links,
            cluster_axis=None,
            mesh_device=mesh,
            topology=ttnn.Topology.Ring,
            subdevice_id=runtime.worker_sub_device_id,
            ccl_core_grid_offset=(runtime.ccl_column, 0),
            use_column_major_ccl=True,
            is_balanced=False,
        )
        if metadata:
            slot, prefix = _make_ring_mla_metadata(mesh, 0, actual_isl)
            kwargs.update(slot_id=slot, kv_actual_isl_tensor=prefix)
        elif metadata is False:
            kwargs.update(kv_actual_isl=actual_isl, kv_cache_batch_idx=0)

        def check_output(output, check_prefix, check_end, selected_batch=0):
            check_k = k[selected_batch : selected_batch + 1, :, :check_end]
            shards = ttnn.get_device_tensors(output)
            assert len(shards) == ranks
            for rank, shard in enumerate(shards):
                q_rank, lane = divmod(rank, tp)
                local_q = q[
                    :, lane * local_heads : (lane + 1) * local_heads, q_rank * q_slab : (q_rank + 1) * q_slab
                ].float()
                # Enumerate the absolute rows owned by this SP rank, independent
                # of the kernel's pre-wrap/post-wrap interval formulas.
                q_positions = torch.tensor(
                    [pos for pos in range(check_prefix, check_end) if (pos // q_slab) % sp == q_rank]
                )
                valid_rows = len(q_positions)
                if not valid_rows:
                    continue  # The API does not promise outputs for padded Q rows.
                local_q = local_q[:, :, :valid_rows]
                scores = local_q @ check_k.float().transpose(-1, -2) / math.sqrt(d_k)
                mask = torch.arange(check_end)[None, :] > q_positions[:, None]
                expected = scores.masked_fill(mask, -torch.inf).softmax(-1) @ check_k[..., :d_v].float()
                actual = ttnn.to_torch(shard).float()[:, :, :valid_rows]
                assert torch.isfinite(actual).all(), f"non-finite output on tensor rank {rank}"
                passed, message = comp_pcc(expected, actual, 0.999)
                assert passed, f"tensor rank {rank}: {message}"

        def check_gather_tail(check_end):
            # Each physical source retains its allocated stride even when only
            # a prefix is transferred. The first tile past its live slabs must
            # remain untouched on every destination.
            live_slabs = math.ceil(check_end / chunk)
            if live_slabs < capacity_depth:
                for source in range(ranks):
                    row = source * source_capacity + live_slabs * region
                    tail = ttnn.slice(scratch, (0, 0, row, 0), (1, 1, row + 32, d_k))
                    for shard in ttnn.get_device_tensors(tail):
                        assert torch.all(ttnn.to_torch(shard) == 13.0)

        program_count = None
        for _ in range(2):
            output, _ = ttnn.transformer.ring_mla(tt_q, tt_k, **kwargs)
            ttnn.synchronize_device(mesh)
            if program_count is None:
                program_count = mesh.num_program_cache_entries()
                assert program_count > 0
            else:
                assert mesh.num_program_cache_entries() == program_count
            check_output(output, actual_isl, live_end)
        if scalar_replay or structural_cache or trace_replay:
            # Host copy restored into scratch before each replay, so earlier
            # dispatches cannot satisfy a broken gather through stale rows.
            scratch_poison = ttnn.from_torch(
                torch.full((1, 1, scratch_capacity, d_k), 13.0, dtype=torch.bfloat16),
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        if scalar_replay:
            assert metadata is False and live_end == input_capacity and not indexed
            prefixes = [(d - 1) * chunk for d in (1, 5, 2, 8, 1, 3)] + [chunk + 32, actual_isl]
            for replay_prefix in prefixes:
                replay_end = min(replay_prefix + chunk, input_capacity)
                kwargs.update(kv_actual_isl=replay_prefix, logical_n=replay_end)
                ttnn.copy_host_to_device_tensor(scratch_poison, scratch)
                # Oracle slices may add their own cached programs. Measure only
                # the ring dispatch's cache delta, after warming it above.
                before = mesh.num_program_cache_entries()
                output, _ = ttnn.transformer.ring_mla(tt_q, tt_k, **kwargs)
                ttnn.synchronize_device(mesh)
                assert mesh.num_program_cache_entries() == before
                check_output(output, replay_prefix, replay_end)
                check_gather_tail(replay_end)
        if structural_cache:
            assert metadata is True and not indexed and not trace_replay
            original_topology = tt_q.tensor_topology()
            original_shape = tuple(tt_q.shape)
            before = mesh.num_program_cache_entries()
            try:
                # Same physical Q/K shards and local specs, but Q now has the
                # full KV sequence distribution: stripe ratio changes T -> 1.
                tt_q.update_tensor_topology(tt_k.tensor_topology())
                assert tuple(tt_q.shape) == original_shape
                ttnn.copy_host_to_device_tensor(scratch_poison, scratch)
                ttnn.transformer.ring_mla(tt_q, tt_k, **kwargs)
                ttnn.synchronize_device(mesh)
                assert mesh.num_program_cache_entries() == before + 1
                ttnn.transformer.ring_mla(tt_q, tt_k, **kwargs)
                ttnn.synchronize_device(mesh)
                assert mesh.num_program_cache_entries() == before + 1
            finally:
                tt_q.update_tensor_topology(original_topology)
            ttnn.copy_host_to_device_tensor(scratch_poison, scratch)
            output, _ = ttnn.transformer.ring_mla(tt_q, tt_k, **kwargs)
            ttnn.synchronize_device(mesh)
            assert mesh.num_program_cache_entries() == before + 1
            check_output(output, actual_isl, live_end)
        if indexed:
            # Scalar layer and user choices are runtime offsets, so every call
            # must reuse the existing program. Metadata layers are structural;
            # warm each layer once, then demand reuse when switching back.
            for iteration, (user, layer) in enumerate(((2, 1), (1, 0), (0, 1), (2, 0))):
                kwargs["kv_cache_layer_idx"] = layer
                if metadata:
                    ttnn.copy_host_to_device_tensor(_ring_mla_host_scalar_tensor(mesh, user), slot)
                else:
                    kwargs["kv_cache_batch_idx"] = user
                output, _ = ttnn.transformer.ring_mla(tt_q, tt_k, **kwargs)
                ttnn.synchronize_device(mesh)
                check_output(output, actual_isl, live_end, user * num_layers + layer)
                if metadata and iteration == 0:
                    program_count = mesh.num_program_cache_entries()
                else:
                    assert mesh.num_program_cache_entries() == program_count

        if trace_replay:
            assert metadata and live_end == input_capacity
            traced_outputs = []
            for layer in range(num_layers):
                kwargs["kv_cache_layer_idx"] = layer
                ttnn.copy_host_to_device_tensor(_ring_mla_host_scalar_tensor(mesh, 0), prefix)
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                traced_output, _ = ttnn.transformer.ring_mla(tt_q, tt_k, **kwargs)
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                trace_ids.append(trace_id)
                traced_outputs.append(traced_output)
            # Host logical_n remains capacity for every captured program. Both
            # layer instances consume the same user metadata but distinct data.
            prefixes = [(d - 1) * chunk for d in (1, 5, 2, 8, 1, 3)] + [chunk + 32, actual_isl]
            for replay_index, replay_prefix in enumerate(prefixes):
                user = (2, 0, 1, 2, 1, 0, 2, 1)[replay_index] if indexed else 0
                ttnn.copy_host_to_device_tensor(_ring_mla_host_scalar_tensor(mesh, user), slot)
                ttnn.copy_host_to_device_tensor(_ring_mla_host_scalar_tensor(mesh, replay_prefix), prefix)
                for layer, (trace_id, traced_output) in enumerate(zip(trace_ids, traced_outputs, strict=True)):
                    # Prevent earlier invocations from satisfying a broken gather
                    # through stale scratch; keep the captured address fixed.
                    ttnn.copy_host_to_device_tensor(scratch_poison, scratch)
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                    replay_end = min(replay_prefix + chunk, input_capacity)
                    check_output(traced_output, replay_prefix, replay_end, user * num_layers + layer)
                    check_gather_tail(replay_end)
        if indexed:
            for rank, shard in enumerate(ttnn.get_device_tensors(tt_k)):
                assert torch.equal(ttnn.to_torch(shard)[:, 0], cache_sources[:, rank])
        if metadata is not None and not trace_replay:
            check_gather_tail(live_end)
    finally:
        for trace_id in trace_ids:
            ttnn.release_trace(runtime.mesh_device, trace_id)
        close_ring_joint_sdpa_runtime(runtime)


def test_ring_mla_split_kv_inactive_source_fallback():
    # Only source zero contributes. Missing source bits must disable grouping,
    # and the second tile of its allocated region remains finite poison.
    run_ring_mla_split_kv_geometry(1, 8, 0, False, 32, 352, 4, effective_n=32)


@pytest.mark.timeout(600)
def test_ring_mla_split_kv_metadata_trace_replay():
    run_ring_mla_split_kv_geometry(8, 8, 32, True, 32, 352, 4, trace_replay=True)


def test_ring_mla_split_kv_scalar_prefix_cache_hits():
    run_ring_mla_split_kv_geometry(8, 8, 32, False, 32, 352, 4, scalar_replay=True)


def test_ring_mla_split_kv_structural_layout_cache_key():
    run_ring_mla_split_kv_geometry(1, 8, 0, True, 32, 352, 4, structural_cache=True)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("metadata", [False, True], ids=["scalar_cache_hits", "metadata_traces"])
def test_ring_mla_split_kv_slots_and_layers(metadata):
    run_ring_mla_split_kv_geometry(8, 8, 32, metadata, 32, 352, 4, trace_replay=metadata, indexed=True)


@pytest.mark.parametrize(
    "depth,prefix_offset,metadata",
    [(1, 0, False), (5, 0, True), (1, 32, True), (3, 672, False)],
    ids=["first-scalar", "longer-metadata", "region-rotation-metadata", "q-slab-rotation-scalar"],
)
def test_ring_mla_split_kv_production_dimensions(depth, prefix_offset, metadata):
    run_ring_mla_split_kv_geometry(depth, 8, prefix_offset, metadata, 32, 640, 16, production=True)


@pytest.mark.parametrize("k_chunk", [32, 128, 256, 352])
def test_ring_mla_split_kv_production_k_widths(k_chunk):
    # DK576/BF8's three K buffers alone need 2,350,080 bytes at K1280,
    # beyond Blackhole worker L1. Larger K widths retain tiny-DK coverage.
    run_ring_mla_split_kv_geometry(2, 8, 32, True, 32, k_chunk, 16, production=True)


@pytest.mark.parametrize("depth,k_chunk", [(2, 128), (4, 256), (2, 1280), (4, 2048)])
def test_ring_mla_split_kv_packed_widths(depth, k_chunk):
    run_ring_mla_split_kv_geometry(depth, 8, 0, True, 32, k_chunk, 4)


@pytest.mark.parametrize(
    "depth,prefix_offset,metadata,q_chunk,k_chunk,effective_n,trace_replay",
    [
        pytest.param(5, 0, False, 32, 352, None, False, id="grouped-inplace-scalar"),
        pytest.param(3, 288, True, 64, 640, None, False, id="grouped-materialized-metadata"),
        pytest.param(8, 32, True, 32, 352, None, True, id="grouped-inplace-trace"),
        pytest.param(1, 0, False, 32, 352, 160, False, id="per-source-fallback"),
    ],
)
def test_ring_mla_split_kv_rotated_q_split(depth, prefix_offset, metadata, q_chunk, k_chunk, effective_n, trace_replay):
    # 32 local heads leave float Q chunks on the Blackhole SDPA grid (256 chunks at Q32 and
    # 128 at Q64 over 110 cores), so the rotated Q split migrates them between grid rows.
    # Grouped traversal executes only ring_size / TP of the scheduled ordinals. The fallback case
    # leaves sources 0-2 active (160 rows over 64-row regions): per-source traversal with
    # three active ordinals, so floats still migrate between them.
    run_ring_mla_split_kv_geometry(
        depth,
        8,
        prefix_offset,
        metadata,
        q_chunk,
        k_chunk,
        32,
        effective_n=effective_n,
        trace_replay=trace_replay,
    )


@pytest.mark.parametrize(
    "depth,prefix_offset,metadata,effective_n",
    [(5, 0, False, None), (3, 32, True, None), (1, 0, False, 160)],
    ids=["grouped-scalar", "grouped-rotated-metadata", "per-source-fallback"],
)
def test_ring_mla_split_kv_segmented_accumulation(depth, prefix_offset, metadata, effective_n):
    # Segmented accumulation folds each SDPA iteration into the restore CBs. Grouped traversal
    # makes an iteration span several sources, and the fallback keeps one source per iteration.
    run_ring_mla_split_kv_geometry(
        depth, 8, prefix_offset, metadata, 32, 352, 4, effective_n=effective_n, segmented=True
    )


def test_ring_mla_split_kv_ordinary_mesh():
    run_ring_mla_split_kv_geometry(5, 8, 32, True, 32, 352, 4, ordinary_mesh=True)


@pytest.mark.parametrize(
    "invalid_case,error",
    [
        pytest.param(case, error, id=case)
        for case, error in (
            ("capacity", "must not exceed input KV capacity"),
            ("region_hole", "whole KV regions"),
            ("balanced", "Split KV supports only"),
            ("transposed", "Q sequence sharded only on mesh axis 0"),
            ("fp32", "Split KV requires streaming"),
        )
    ],
)
def test_ring_mla_split_kv_rejects_unsupported_geometry(invalid_case, error, expect_error):
    if MESH_CONFIG.num_devices not in (8, 32):
        pytest.skip("Split-KV coverage requires an eight-device LB or 32-device Galaxy")
    tp, sp = 4, MESH_CONFIG.num_devices // 4
    runtime = open_ring_joint_sdpa_runtime(
        replace(MESH_CONFIG, tp_size=sp, sp_size=tp), full_mesh=True, fp32_dest_acc_en=invalid_case == "fp32"
    )
    try:
        mesh = runtime.mesh_device
        source_rows = 96 if invalid_case == "region_hole" else 64
        capacity = source_rows * sp * tp
        q = ttnn.from_torch(
            torch.zeros(1, 16, sp * 256, 64),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh, mesh_shape=(sp, tp), dims=[1, 2] if invalid_case == "transposed" else [2, 1]
            ),
        )
        k = ttnn.from_torch(
            torch.zeros(1, 1, capacity, 64),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
        )
        scratch = ttnn.from_torch(
            torch.zeros(1, 1, capacity + 64, 64),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        with expect_error(RuntimeError, error):
            ttnn.transformer.ring_mla(
                q,
                k,
                persistent_output_buffer_kv=scratch,
                head_dim_v=32,
                logical_n=capacity + 32 if invalid_case == "capacity" else sp * 256,
                kv_actual_isl=0,
                kv_cache_batch_idx=0,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=runtime.sdpa_compute_grid,
                    q_chunk_size=32,
                    k_chunk_size=32,
                    exp_approx_mode=False,
                ),
                compute_kernel_config=runtime.compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=runtime.ccl_semaphore_handles,
                num_links=runtime.num_links,
                cluster_axis=None,
                mesh_device=mesh,
                topology=ttnn.Topology.Ring,
                subdevice_id=runtime.worker_sub_device_id,
                ccl_core_grid_offset=(runtime.ccl_column, 0),
                use_column_major_ccl=True,
                is_balanced=invalid_case == "balanced",
            )
    finally:
        close_ring_joint_sdpa_runtime(runtime)


def skip_unless_split_kv_perf_mesh():
    if MESH_CONFIG.num_devices != 32:
        pytest.skip("Split-KV production perf is defined for the 32-device Galaxy")


def run_ring_mla_split_kv_perf(model_name, q_chunk_size, k_chunk_size, repeats=1, traced=False):
    """Profile the final chunk of a production chunked prefill through split-KV ring MLA.

    Same work as test_ring_mla_chunked_perf_check (per-device Q slab, heads, latent
    dimensions, dtypes, prefix and chunk sizes), but Q is sharded over SP only and
    KV is block-cyclic over the complete SP x TP mesh. Returns per-repeat utilizations.
    """
    skip_unless_split_kv_perf_mesh()
    model = RING_MLA_CHUNKED_MODEL_CONFIGS[model_name]
    tp = MESH_CONFIG.tp_size
    sp = MESH_CONFIG.sp_size
    ranks = sp * tp
    chunk = CHUNKED_PREFILL_CHUNK_SIZE
    depth = CHUNKED_PREFILL_TOTAL_SEQ // chunk
    perf_chunk = depth - 1
    region = chunk // ranks
    q_slab = chunk // sp
    source_capacity = depth * region

    config = replace(MESH_CONFIG, tp_size=sp, sp_size=tp)
    runtime = open_ring_joint_sdpa_runtime(config, full_mesh=True, trace_region_size=4 * 1024 * 1024 if traced else 0)
    trace_id = None
    try:
        mesh = runtime.mesh_device
        assert tuple(mesh.shape) == (sp, tp)
        torch.manual_seed(20260928)
        tt_q = ttnn.from_torch(
            torch.randn(1, model.nhq * tp, sp * q_slab, model.d_q).bfloat16(),
            dtype=model.q_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=[2, 1]),
        )
        tt_k = ttnn.from_torch(
            torch.randn(1, 1, ranks * source_capacity, model.d_k).bfloat16(),
            dtype=model.kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
        )
        scratch = ttnn.from_torch(
            torch.zeros(1, 1, ranks * source_capacity, model.d_k).bfloat16(),
            dtype=model.kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        kwargs = dict(
            persistent_output_buffer_kv=scratch,
            head_dim_v=model.d_v,
            logical_n=depth * chunk,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=runtime.sdpa_compute_grid,
                q_chunk_size=q_chunk_size,
                k_chunk_size=k_chunk_size,
                exp_approx_mode=False,
            ),
            compute_kernel_config=runtime.compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=runtime.ccl_semaphore_handles,
            num_links=runtime.num_links,
            cluster_axis=None,
            mesh_device=mesh,
            topology=ttnn.Topology.Ring,
            subdevice_id=runtime.worker_sub_device_id,
            ccl_core_grid_offset=(runtime.ccl_column, 0),
            use_column_major_ccl=True,
            is_balanced=False,
        )

        def run():
            if trace_id is None:
                ttnn.transformer.ring_mla(tt_q, tt_k, **kwargs)
            else:
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)

        run()  # compile outside the profiled window
        if traced:
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            ttnn.transformer.ring_mla(tt_q, tt_k, **kwargs)
            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            run()
        utilizations = []
        for _ in range(repeats):
            duration_ns, _ = profile_ring_joint_runtime_duration_ns(mesh, run)
            # Q rows per device and heads per device match the classic SP ring, so the
            # classic utilization model applies unchanged.
            utilization, _ = compute_chunked_prefill_perf_check_utilization(
                MESH_CONFIG, model, chunk, perf_chunk, duration_ns, MESH_CONFIG.sdpa_cores
            )
            logger.info(
                f"split-KV ring_mla {model_name}-q{q_chunk_size}-k{k_chunk_size} 50k+5k: "
                f"duration={duration_ns / 1e6:.3f} ms, math_util={utilization:.2f}%"
            )
            utilizations.append(utilization)
        return utilizations
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        close_ring_joint_sdpa_runtime(runtime)


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize("model_name", ["kimi50k", "kimi_k3"])
def test_ring_mla_split_kv_perf_impl(model_name):
    """Repeated split-KV profile for local A/B comparison against the classic SP ring."""
    run_ring_mla_split_kv_perf(model_name, 32, 640, repeats=int(os.environ.get("SPLIT_KV_PERF_REPEATS", "5")))


# Split KV and the classic SP ring do identical math per device; allow this much relative
# utilization loss for the extra fabric hops and packed-source bookkeeping.
SPLIT_KV_RELATIVE_PERF_MARGIN = 0.025


def classic_ring_mla_chunked_utilization(model_name, q_chunk_size, k_chunk_size):
    """Utilization of the classic SP-ring ring_mla on the final 50k+5k chunk (test_ring_mla_chunked_perf_check)."""
    model = RING_MLA_CHUNKED_MODEL_CONFIGS[model_name]
    chunk = CHUNKED_PREFILL_CHUNK_SIZE
    perf_chunk = CHUNKED_PREFILL_TOTAL_SEQ // chunk - 1
    runtime = open_ring_joint_sdpa_runtime(MESH_CONFIG)
    try:
        with mock.patch.dict(os.environ, {CHUNKED_PREFILL_CHUNK_ID_ENV: str(perf_chunk)}):
            duration_ns, _ = profile_ring_joint_runtime_duration_ns(
                runtime.mesh_device,
                lambda: run_ring_joint_sdpa_chunked(
                    MESH_CONFIG,
                    model,
                    chunk_size=chunk,
                    qk_configs=[(q_chunk_size, k_chunk_size)],
                    persistent_buffer_mode="reuse_max",
                    use_ring_mla=True,
                    do_check=False,
                    reuse_kv_buffer=False,
                    runtime=runtime,
                ),
            )
    finally:
        close_ring_joint_sdpa_runtime(runtime)
    utilization, _ = compute_chunked_prefill_perf_check_utilization(
        MESH_CONFIG, model, chunk, perf_chunk, duration_ns, MESH_CONFIG.sdpa_cores
    )
    logger.info(f"classic ring_mla {model_name}-q{q_chunk_size}-k{k_chunk_size} 50k+5k: math_util={utilization:.2f}%")
    return utilization


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "model_name, q_chunk_size, k_chunk_size, ring_size_expected, expected_util, classic_margin",
    RING_MLA_CHUNKED_PERF_CHECK_CONFIGS,
    ids=[f"{cfg[0]}-q{cfg[1]}-k{cfg[2]}-ring{cfg[3]}" for cfg in RING_MLA_CHUNKED_PERF_CHECK_CONFIGS],
)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.skipif(
    MESH_CONFIG.is_galaxy and not is_high_power(),
    reason="galaxy perf job requires a high-power (>=130W TDP) host",
)
def test_ring_mla_split_kv_perf_check(
    model_name, q_chunk_size, k_chunk_size, ring_size_expected, expected_util, classic_margin
):
    """Split KV must stay within SPLIT_KV_RELATIVE_PERF_MARGIN of the classic SP ring, measured back to
    back on the same host so power limits and matmul throttling affect both alike.

    Split KV is profiled under trace replay, as deployed: eager dispatch skews program start across the
    32 chips, and a split-KV device holds only 1/TP of the KV locally to hide that skew behind.
    """
    # The gate is relative to the classic ring measured back to back, so expected_util is only
    # logged and classic_margin is unused; test_ring_mla_chunked_perf_check enforces that band.
    skip_unless_split_kv_perf_mesh()  # before the classic baseline spends its run
    if MESH_CONFIG.sp_size != ring_size_expected:
        pytest.skip(f"Expected SP size {ring_size_expected}, current topology has {MESH_CONFIG.sp_size}")
    classic = classic_ring_mla_chunked_utilization(model_name, q_chunk_size, k_chunk_size)
    split = sorted(run_ring_mla_split_kv_perf(model_name, q_chunk_size, k_chunk_size, repeats=3, traced=True))[1]
    logger.info(
        f"split-KV vs classic {model_name}: {split:.2f}% vs {classic:.2f}% "
        f"({(split / classic - 1) * 100:+.2f}%, classic expected {expected_util:.2f}%)"
    )
    assert split >= classic * (1 - SPLIT_KV_RELATIVE_PERF_MARGIN), (
        f"Split-KV math utilization {split:.2f}% is more than {SPLIT_KV_RELATIVE_PERF_MARGIN * 100:.1f}% "
        f"below the classic SP ring's {classic:.2f}%"
    )
