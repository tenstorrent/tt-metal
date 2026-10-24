# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LOCAL ONLY (prototype, not for commit): sliding halo exchanged up front, then CCL-free attention per request.

halo_mode on ring_joint_scaled_dot_product_attention: 1 runs only the neighbor-halo exchange (for B lanes, into the
B-batch persistent buffers, no attention); 2 + b runs one request's attention with no CCL, reading the halo already
in batch b. One exchange-only call for B requests plus B pregathered calls must match, bit for bit, one fused
single-request call per request (K split 1 and 3), eagerly and under trace replay with the lane order changed.

Gemma4's sliding shape on the Galaxy: TP4 x CP8 Linear, 8Q:4KV heads per device, D256, W1024, 256-row slabs.
"""

import math

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.operations.ccl import Topology
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    CHUNKED_PREFILL_PCC_THRESHOLD,
    CHUNKED_PREFILL_SEED,
    DEFAULT_RMSE_THRESHOLD,
    MESH_CONFIG,
    RING_JOINT_TRACE_REGION_SIZE,
    _SLIDING_LANES_SCENARIOS,
    _ring_joint_lane_metadata_host_tensor,
    _ring_joint_lane_metadata_tensor,
    assert_pcc_rmse,
    close_ring_joint_sdpa_runtime,
    fa_rand,
    open_ring_joint_sdpa_runtime,
    to_balanced_growing_cache_layout,
    torch_chunked_causal_sdpa_reference,
)

LOCAL, WINDOW, HQ, HKV, D, QC, KC = 256, 1024, 8, 4, 256, 128, 128


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("scenario", ["lane0_deepest", "same_one_request", "two_lanes", "eight_lanes"])
def test_halo_preexchange_matches_fused(scenario):
    if not MESH_CONFIG.is_galaxy or MESH_CONFIG.sp_size != 8:
        pytest.skip("needs the TP4 x SP8 Galaxy")
    prefix_chunks, step_chunks = _SLIDING_LANES_SCENARIOS[scenario]
    mesh_config = MESH_CONFIG
    sp, tp = mesh_config.sp_size, mesh_config.tp_size
    nhq, nhk = HQ * tp, HKV * tp
    gqa = HQ // HKV
    num_lanes, num_slots = len(step_chunks), len(prefix_chunks)
    chunk_global = LOCAL * sp
    filled = [max([n] + [idx + 1 for s, idx in step_chunks if s == slot]) for slot, n in enumerate(prefix_chunks)]
    cache_groups = max(filled) + 1
    cache_seq_per_dev = cache_groups * LOCAL
    capacity = sp * cache_seq_per_dev
    halo_tokens = math.ceil((WINDOW - 1) / KC) * KC

    torch.manual_seed(CHUNKED_PREFILL_SEED + 902)
    k_tokens = [fa_rand(1, nhk, n * chunk_global, D) for n in filled]
    v_tokens = [fa_rand(1, nhk, n * chunk_global, D) for n in filled]
    lane_q = [fa_rand(1, nhq, chunk_global, D) for _ in step_chunks]
    lane_slots = [slot for slot, _ in step_chunks]
    lane_prefixes = [idx * chunk_global for _, idx in step_chunks]

    def cache_host(tokens):
        per_dev = torch.randn(num_slots, nhk, sp, cache_seq_per_dev, D) * 100
        for slot, n in enumerate(filled):
            packed = to_balanced_growing_cache_layout(tokens[slot], sp, chunk_global, n - 1)
            per_dev[slot, :, :, : n * LOCAL] = packed.reshape(nhk, sp, n * LOCAL, D)
        return per_dev.reshape(num_slots, nhk, capacity, D)

    def stacked_q(order):
        return (
            torch.stack([lane_q[lane] for lane in order], dim=2)
            .reshape(1, nhq, num_lanes, sp, LOCAL, D)
            .transpose(2, 3)
            .reshape(1, nhq, sp * num_lanes * LOCAL, D)
        )

    runtime = open_ring_joint_sdpa_runtime(
        mesh_config, topology=Topology.Linear, trace_region_size=RING_JOINT_TRACE_REGION_SIZE
    )
    mesh_device = runtime.mesh_device
    trace_id = None
    try:
        mesh_device.enable_program_cache()
        input_dims = [None, None]
        input_dims[runtime.sp_axis] = 2
        input_dims[runtime.tp_axis] = 1
        halo_dims = [None, None]
        halo_dims[runtime.tp_axis] = 1
        mapper_of = lambda dims: ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)

        def upload(host, dtype, dims):
            return ttnn.from_torch(
                host,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper_of(dims),
            )

        tt_k = upload(cache_host(k_tokens), ttnn.bfloat8_b, input_dims)
        tt_v = upload(cache_host(v_tokens), ttnn.bfloat8_b, input_dims)
        composer = ttnn.create_mesh_composer(
            mesh_device, ttnn.MeshComposerConfig(input_dims[0] if input_dims[0] is not None else -1, input_dims[1])
        )

        def config(ks):
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=runtime.sdpa_compute_grid,
                q_chunk_size=QC,
                k_chunk_size=KC,
                exp_approx_mode=False,
                k_splits=ks,
            )

        def halo_pair(batch):
            # Garbage, so a halo that was not exchanged shows up as a mismatch.
            return [upload(torch.randn(batch, nhk, halo_tokens, D) * 100, ttnn.bfloat8_b, halo_dims) for _ in range(2)]

        def call(q, metadata, halos, ks, halo_mode):
            out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q,
                tt_k,
                tt_v,
                None,
                None,
                None,
                persistent_output_buffer_k=halos[0],
                persistent_output_buffer_v=halos[1],
                joint_strategy="rear",
                logical_n=capacity,
                is_causal=True,
                is_balanced=False,
                program_config=config(ks),
                compute_kernel_config=runtime.compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=runtime.ccl_semaphore_handles,
                num_links=runtime.num_links,
                cluster_axis=runtime.sp_axis,
                mesh_device=mesh_device,
                topology=runtime.topology,
                subdevice_id=runtime.worker_sub_device_id,
                ccl_core_grid_offset=(runtime.ccl_column, 0),
                use_column_major_ccl=True,
                sliding_window_size=WINDOW,
                slot_id=metadata[0],
                kv_actual_isl_tensor=metadata[1],
                halo_mode=halo_mode,
            )
            return out

        def read(out):
            return ttnn.to_torch(out, mesh_composer=composer)

        tt_lane_q = [upload(q, ttnn.bfloat16, input_dims) for q in lane_q]
        lane_meta = [
            [_ring_joint_lane_metadata_tensor(mesh_device, [v]) for v in (s, p)]
            for s, p in zip(lane_slots, lane_prefixes)
        ]

        # Reference: one fused call per request.
        fused = {}
        for ks in (1, 3):
            for lane in range(num_lanes):
                out = call(tt_lane_q[lane], lane_meta[lane], halo_pair(1), ks, 0)
                fused[ks, lane] = read(out)
                ttnn.deallocate(out)
        for lane in range(num_lanes):
            slot, prefix = lane_slots[lane], lane_prefixes[lane]
            ref = torch_chunked_causal_sdpa_reference(
                lane_q[lane],
                k_tokens[slot][:, :, : prefix + chunk_global].repeat_interleave(gqa, dim=1),
                v_tokens[slot][:, :, : prefix + chunk_global].repeat_interleave(gqa, dim=1),
                prefix,
                sliding_window_size=WINDOW,
            )
            assert_pcc_rmse(ref, fused[1, lane], CHUNKED_PREFILL_PCC_THRESHOLD, DEFAULT_RMSE_THRESHOLD, f"lane {lane}")

        def check(outs, order, ks, label):
            # outs[i] serves the step's lane order[i].
            for position, lane in enumerate(order):
                got, expected = outs[position], fused[ks, lane]
                mismatch = got != expected
                assert not mismatch.any(), (
                    f"{label} ks{ks}: position {position} (slot {lane_slots[lane]}, prefix {lane_prefixes[lane]}) "
                    f"differs from its fused call: mismatches={mismatch.sum().item()}, "
                    f"max abs diff={(got - expected).abs().max().item()}"
                )

        vec_meta = [_ring_joint_lane_metadata_tensor(mesh_device, v) for v in (lane_slots, lane_prefixes)]
        tt_stacked = upload(stacked_q(range(num_lanes)), ttnn.bfloat16, input_dims)
        halos = halo_pair(num_lanes)

        def split_step(ks, keep=True):
            exchanged = call(tt_stacked, vec_meta, halos, 1, 1)
            outs = [call(tt_lane_q[b], lane_meta[b], halos, ks, 2 + b) for b in range(num_lanes)]
            return exchanged, outs

        identity = list(range(num_lanes))
        for ks in (1, 3):
            exchanged, outs = split_step(ks)
            ttnn.synchronize_device(mesh_device)
            check([read(o) for o in outs], identity, ks, "eager")
            logger.info(f"[halo_pre] {scenario} eager ks{ks}: bit-identical to the fused calls")
            for o in outs:
                ttnn.deallocate(o)

        # Trace: capture in one lane order, replay in others with Q, metadata and halo buffers refreshed in place.
        zero_halo = ttnn.from_torch(
            torch.randn(num_lanes, nhk, halo_tokens, D) * 100,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper_of(halo_dims),
        )

        def stage(order):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    stacked_q(order), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper_of(input_dims)
                ),
                tt_stacked,
            )
            for position, lane in enumerate(order):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(
                        lane_q[lane], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper_of(input_dims)
                    ),
                    tt_lane_q[position],
                )
                for tensor, value in zip(lane_meta[position], (lane_slots[lane], lane_prefixes[lane])):
                    ttnn.copy_host_to_device_tensor(_ring_joint_lane_metadata_host_tensor(mesh_device, [value]), tensor)
            for tensor, values in zip(
                vec_meta, ([lane_slots[lane] for lane in order], [lane_prefixes[lane] for lane in order])
            ):
                ttnn.copy_host_to_device_tensor(_ring_joint_lane_metadata_host_tensor(mesh_device, values), tensor)
            for halo in halos:
                ttnn.copy_host_to_device_tensor(zero_halo, halo)

        ks = 3
        stage(identity)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        _, traced_outs = split_step(ks)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        for order in (identity[1:] + identity[:1], identity[::-1], identity):
            stage(order)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            check([read(o) for o in traced_outs], order, ks, f"trace replay {order}")
        logger.info(f"[halo_pre] {scenario} trace replay ks{ks}: bit-identical in 3 lane orders")
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh_device, trace_id)
        close_ring_joint_sdpa_runtime(runtime, clear_program_cache=True)
