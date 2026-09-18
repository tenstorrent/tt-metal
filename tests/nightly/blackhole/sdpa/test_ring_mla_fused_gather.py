# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Op-level split-KV MLA: independent TP heads and poisoned fixed-capacity caches."""

import math
from dataclasses import replace

import pytest
import torch

import ttnn
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    MESH_CONFIG,
    _make_ring_mla_metadata,
    _ring_mla_host_scalar_tensor,
    close_ring_joint_sdpa_runtime,
    open_ring_joint_sdpa_runtime,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


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
def test_ring_mla_split_kv_geometry(
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
):
    if metadata is None and prefix_offset:
        pytest.skip("Rotated prefixes require actual ISL")
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
        full_mesh_fabric=ttnn.FabricConfig.FABRIC_2D if ordinary_mesh else None,
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
        scratch_capacity = input_capacity if structural_cache else 1048576
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
        if scalar_replay or structural_cache:
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
            scratch_poison = ttnn.from_torch(
                torch.full((1, 1, scratch_capacity, d_k), 13.0, dtype=torch.bfloat16),
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
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
    test_ring_mla_split_kv_geometry(1, 8, 0, False, 32, 352, 4, effective_n=32)


def test_ring_mla_split_kv_metadata_trace_replay():
    test_ring_mla_split_kv_geometry(8, 8, 32, True, 32, 352, 4, trace_replay=True)


def test_ring_mla_split_kv_scalar_prefix_cache_hits():
    test_ring_mla_split_kv_geometry(8, 8, 32, False, 32, 352, 4, scalar_replay=True)


def test_ring_mla_split_kv_structural_layout_cache_key():
    test_ring_mla_split_kv_geometry(1, 8, 0, True, 32, 352, 4, structural_cache=True)


@pytest.mark.parametrize("metadata", [False, True], ids=["scalar_cache_hits", "metadata_traces"])
def test_ring_mla_split_kv_slots_and_layers(metadata):
    test_ring_mla_split_kv_geometry(8, 8, 32, metadata, 32, 352, 4, trace_replay=metadata, indexed=True)


@pytest.mark.parametrize(
    "depth,prefix_offset,metadata",
    [(1, 0, False), (5, 0, True), (1, 32, True), (3, 672, False)],
    ids=["first-scalar", "longer-metadata", "region-rotation-metadata", "q-slab-rotation-scalar"],
)
def test_ring_mla_split_kv_production_dimensions(depth, prefix_offset, metadata):
    test_ring_mla_split_kv_geometry(depth, 8, prefix_offset, metadata, 32, 640, 16, production=True)


@pytest.mark.parametrize("k_chunk", [32, 128, 256, 352])
def test_ring_mla_split_kv_production_k_widths(k_chunk):
    # DK576/BF8's three K buffers alone need 2,350,080 bytes at K1280,
    # beyond Blackhole worker L1. Larger K widths retain tiny-DK coverage.
    test_ring_mla_split_kv_geometry(2, 8, 32, True, 32, k_chunk, 16, production=True)


@pytest.mark.parametrize("depth,k_chunk", [(2, 128), (4, 256), (2, 1280), (4, 2048)])
def test_ring_mla_split_kv_packed_widths(depth, k_chunk):
    test_ring_mla_split_kv_geometry(depth, 8, 0, True, 32, k_chunk, 4)


def test_ring_mla_split_kv_ordinary_mesh():
    test_ring_mla_split_kv_geometry(5, 8, 32, True, 32, 352, 4, ordinary_mesh=True)


@pytest.mark.parametrize(
    "invalid_case,error",
    [
        ("capacity", "must not exceed input KV capacity"),
        ("region_hole", "whole KV regions"),
        ("balanced", "Split KV supports only"),
        ("transposed", "Q sequence on mesh axis 0"),
        ("fp32", "Split KV requires streaming"),
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
