# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LOCAL ONLY (PR 2 review, not for commit): one B-lane sliding ring SDPA call vs B per-request calls.

Gemma4's sliding shape on the Galaxy as the model runs it: TP4 x CP8, Linear, 8Q:4KV heads per device, D256,
W1024, 256-row slabs per rank (chunk 2048), q128 / k128, every lane at a nonzero prefix (steady state).

Variants, per step of B requests (one "layer"):
  lanes        one call, B-element metadata, one B-batch halo, K split 1 (PR 2)
  loop_ks3     B calls on per-request Q tensors, K split 3 (what ring_sdpa_chunk_sizes picks per request)
  loop_ks1     the same with K split 1 (same SDPA config as lanes)
  glue_ks3     the model's loop today: slice each request's Q rows out of the stacked Q, B calls, concat
Two numbers each:
  device       sum of per-program device time (real-time profiler, max over chips), per step, eager
  traced       wall time per step of a trace holding R steps back to back (R=50: Gemma4's sliding layers)
LANES_AB_B=2,4,8  LANES_AB_R=50  LANES_AB_PREFIX_CHUNKS=2
"""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.operations.ccl import Topology
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    MESH_CONFIG,
    RING_JOINT_TRACE_REGION_SIZE,
    _ring_joint_lane_metadata_tensor,
    call_sdpa,
    close_ring_joint_sdpa_runtime,
    fa_rand,
    open_ring_joint_sdpa_runtime,
)
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged

LOCAL, WINDOW, HQ, HKV, D = int(os.environ.get("LANES_AB_LOCAL", "256")), 1024, 8, 4, 256
QC = int(os.environ.get("LANES_AB_QC", "128"))  # loop calls' Q chunk; lanes keep 128 unless LANES_AB_LANES_QC
KC = int(os.environ.get("LANES_AB_KC", "128"))
LANES_QC = int(os.environ.get("LANES_AB_LANES_QC", "128"))
LANE_COUNTS = [int(x) for x in os.environ.get("LANES_AB_B", "2,4,8").split(",")]
REPS = int(os.environ.get("LANES_AB_R", "50"))
PREFIX_CHUNKS = int(os.environ.get("LANES_AB_PREFIX_CHUNKS", "2"))
NUM_LINKS = int(os.environ.get("LANES_AB_LINKS", "2"))
TOPOLOGY = Topology.Ring if os.environ.get("LANES_AB_TOPO", "linear") == "ring" else Topology.Linear
TRACE_REPLAYS = 10
# Few steps per profiler window: ~17 programs x 32 chips per step, and a dropped record fails the run.
PROFILE_STEPS = 3


def _category(sources):
    s = " ".join(sources)
    if "ring_joint" in s or "sdpa" in s:
        return "sdpa"
    if "slice" in s:
        return "slice"
    if "concat" in s:
        return "concat"
    return "other:" + (sources[0].rsplit("/", 1)[-1] if sources else "?")


@pytest.mark.timeout(3600)
def test_lanes_vs_loop_ab():
    if not MESH_CONFIG.is_galaxy or MESH_CONFIG.sp_size != 8:
        pytest.skip("needs the TP4 x SP8 Galaxy")
    mesh_config = MESH_CONFIG
    sp, tp = mesh_config.sp_size, mesh_config.tp_size
    nhq, nhk = HQ * tp, HKV * tp
    chunk_global = LOCAL * sp
    max_b = max(LANE_COUNTS)
    groups = PREFIX_CHUNKS + 2  # filled through this step's chunk, plus a spare group
    capacity = sp * groups * LOCAL
    halo_tokens = -(-(WINDOW - 1) // KC) * KC
    prefix = PREFIX_CHUNKS * chunk_global

    runtime = open_ring_joint_sdpa_runtime(
        mesh_config, topology=TOPOLOGY, trace_region_size=RING_JOINT_TRACE_REGION_SIZE * 4
    )
    mesh_device = runtime.mesh_device
    runtime.num_links = NUM_LINKS
    logger.info(f"[lanes_ab] num_links={NUM_LINKS} topology={TOPOLOGY}")
    results = []
    try:
        mesh_device.enable_program_cache()
        input_dims = [None, None]
        input_dims[runtime.sp_axis] = 2
        input_dims[runtime.tp_axis] = 1
        halo_dims = [None, None]
        halo_dims[runtime.tp_axis] = 1

        def upload(host, dtype, dims, memory_config=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(
                host,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=memory_config,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
            )

        torch.manual_seed(1234)
        # K/V content does not matter for timing; random cache rows for max_b slots.
        kv_mc = ttnn.DRAM_MEMORY_CONFIG
        if os.environ.get("LANES_AB_ND_CACHE") == "1":
            # The Gemma4 model's cache layout: one 32-token row per round-robin DRAM shard.
            from models.demos.gemma4_d_p.tt.attention.ring_prefill import migration_ring_memory_config

            kv_mc = migration_ring_memory_config(mesh_device, D)
        tt_k = upload(torch.randn(max_b, nhk, capacity, D), ttnn.bfloat8_b, input_dims, kv_mc)
        tt_v = upload(torch.randn(max_b, nhk, capacity, D), ttnn.bfloat8_b, input_dims, kv_mc)

        def config(ks, qc=QC):
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=runtime.sdpa_compute_grid,
                q_chunk_size=qc,
                k_chunk_size=KC,
                exp_approx_mode=False,
                k_splits=ks,
            )

        def halo_pair(batch):
            mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("LANES_AB_HALO_L1") == "1" else ttnn.DRAM_MEMORY_CONFIG
            return [upload(torch.zeros(batch, nhk, halo_tokens, D), ttnn.bfloat8_b, halo_dims, mc) for _ in range(2)]

        def sdpa(q, metadata, halos, ks, halo_mode=0, qc=QC):
            if not halo_mode:
                return call_sdpa(
                    q,
                    tt_k,
                    tt_v,
                    capacity,
                    is_causal=True,
                    is_balanced=False,
                    p_buf_k=halos[0],
                    p_buf_v=halos[1],
                    program_config=config(ks, qc),
                    compute_kernel_config=runtime.compute_kernel_config,
                    ccl_semaphore_handles=runtime.ccl_semaphore_handles,
                    num_links=runtime.num_links,
                    sp_axis=runtime.sp_axis,
                    mesh_device=mesh_device,
                    topology=runtime.topology,
                    worker_sub_device_id=runtime.worker_sub_device_id,
                    ccl_column=runtime.ccl_column,
                    sliding_window_size=WINDOW,
                    slot_id=metadata[0],
                    kv_actual_isl_tensor=metadata[1],
                )
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
                program_config=config(ks, qc),
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

        for b in LANE_COUNTS:
            slots, prefixes = list(range(b)), [prefix] * b
            lane_q = [fa_rand(1, nhq, chunk_global, D) for _ in range(b)]
            stacked_host = (
                torch.stack(lane_q, dim=2)
                .reshape(1, nhq, b, sp, LOCAL, D)
                .transpose(2, 3)
                .reshape(1, nhq, sp * b * LOCAL, D)
            )
            tt_stacked = upload(stacked_host, ttnn.bfloat16, input_dims)
            tt_lane_q = [upload(q, ttnn.bfloat16, input_dims) for q in lane_q]
            vec_meta = [_ring_joint_lane_metadata_tensor(mesh_device, v) for v in (slots, prefixes)]
            lane_meta = [
                [_ring_joint_lane_metadata_tensor(mesh_device, [v]) for v in (s, p)] for s, p in zip(slots, prefixes)
            ]
            lanes_halo = halo_pair(b)
            lane_halos = [halo_pair(1) for _ in range(b)]
            rows = tt_stacked.shape[-2] // b  # per-device rows per lane

            def lanes():
                ttnn.deallocate(sdpa(tt_stacked, vec_meta, lanes_halo, 1, qc=LANES_QC))

            def loop(ks):
                def run():
                    for i in range(b):
                        ttnn.deallocate(sdpa(tt_lane_q[i], lane_meta[i], lane_halos[i], ks))

                return run

            def glue(ks):
                def run():
                    outs = []
                    for i in range(b):
                        shape = tuple(tt_stacked.shape)
                        q = ttnn.slice(
                            tt_stacked,
                            (0, 0, i * rows, 0),
                            (shape[0], shape[1], (i + 1) * rows, shape[3]),
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        outs.append(sdpa(q, lane_meta[i], lane_halos[i], ks))
                        ttnn.deallocate(q)
                    out = ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    for o in outs:
                        ttnn.deallocate(o)
                    ttnn.deallocate(out)

                return run

            def split(*parts):
                def run():
                    for part in parts:
                        part()

                return run

            # split_*: the halo-up-front design. Only meaningful with the split probe kernels, where a lanes call
            # runs just the B-request halo exchange and a single-request call runs SDPA with no CCL.
            # halo_*: the real halo_mode op modes (needs the halo_mode build). One exchange-only call for the B
            # requests into the B-batch halo buffer, then one pregathered call per request reading its batch.
            def exchange():
                ttnn.deallocate(sdpa(tt_stacked, vec_meta, lanes_halo, 1, halo_mode=1, qc=LANES_QC))

            def pregathered(ks):
                def run():
                    for i in range(b):
                        ttnn.deallocate(sdpa(tt_lane_q[i], lane_meta[i], lanes_halo, ks, halo_mode=2 + i))

                return run

            def pregathered_glue(ks):
                def run():
                    outs = []
                    for i in range(b):
                        shape = tuple(tt_stacked.shape)
                        q = ttnn.slice(
                            tt_stacked,
                            (0, 0, i * rows, 0),
                            (shape[0], shape[1], (i + 1) * rows, shape[3]),
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        outs.append(sdpa(q, lane_meta[i], lanes_halo, ks, halo_mode=2 + i))
                        ttnn.deallocate(q)
                    out = ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    for o in outs:
                        ttnn.deallocate(o)
                    ttnn.deallocate(out)

                return run

            variants = {
                "lanes": lanes,
                "halo_exchange": exchange,
                "halo_pregathered_ks3": pregathered(3),
                "halo_split_ks3": split(exchange, pregathered(3)),
                "halo_split_ks1": split(exchange, pregathered(1)),
                "halo_split_glue_ks3": split(exchange, pregathered_glue(3)),
                "loop_ks3": loop(3),
                "loop_ks1": loop(1),
                "loop_ks2": loop(2),
                "glue_ks3": glue(3),
                "split_ks3": split(lanes, loop(3)),
                "split_ks1": split(lanes, loop(1)),
                "split_glue_ks3": split(lanes, glue(3)),
            }
            # halo_* variants need the halo_mode prototype build (kmabee/halo-preexchange-proto-1023).
            wanted = os.environ.get("LANES_AB_VARIANTS", "lanes,loop_ks3,loop_ks1,loop_ks2,glue_ks3")
            if wanted:
                variants = {k: v for k, v in variants.items() if k in wanted.split(",") or k == "lanes"}
            if b == 1:
                # A one-request slice covers all of Q and aliases it, so the glue's deallocate would free Q.
                variants = {k: v for k, v in variants.items() if "glue" not in k}
            for name, fn in variants.items():
                fn()  # compile
                ttnn.synchronize_device(mesh_device)

                def steps(fn=fn, n=REPS):
                    for _ in range(n):
                        fn()

                _, programs = profile_realtime_program_merged(mesh_device, lambda: steps(n=PROFILE_STEPS))
                by_cat, sdpa_each = {}, []
                for entry in programs.values():
                    cat = _category(entry["kernel_sources"])
                    by_cat[cat] = by_cat.get(cat, 0.0) + entry["duration_ns"] / PROFILE_STEPS / 1e3
                    if cat == "sdpa":
                        sdpa_each.append(entry["duration_ns"] / 1e3)
                device_us = sum(by_cat.values())

                trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                steps()
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                ttnn.synchronize_device(mesh_device)
                times = []
                for _ in range(TRACE_REPLAYS):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    times.append(time.perf_counter() - t0)
                ttnn.release_trace(mesh_device, trace_id)
                traced_us = statistics.median(times) / REPS * 1e6

                row = dict(
                    B=b,
                    variant=name,
                    traced_us=round(traced_us, 1),
                    device_us=round(device_us, 1),
                    sdpa_call_median_us=round(statistics.median(sdpa_each), 1) if sdpa_each else None,
                    by_cat={k: round(v, 1) for k, v in sorted(by_cat.items())},
                )
                results.append(row)
                logger.info(f"[lanes_ab] {row}")
            for t in [tt_stacked, *tt_lane_q, *vec_meta, *lanes_halo] + [x for m in lane_meta for x in m]:
                ttnn.deallocate(t)
            for pair in lane_halos:
                for t in pair:
                    ttnn.deallocate(t)
    finally:
        close_ring_joint_sdpa_runtime(runtime, clear_program_cache=True)

    logger.info("[lanes_ab] SUMMARY (us per step of B requests; traced = R back-to-back steps in one trace)")
    for r in results:
        base = next(x for x in results if x["B"] == r["B"] and x["variant"] == "lanes")
        logger.info(
            f"[lanes_ab] B={r['B']} {r['variant']:9s} traced {r['traced_us']:8.1f} "
            f"(+{r['traced_us'] - base['traced_us']:7.1f} vs lanes)  device {r['device_us']:8.1f}  "
            f"sdpa/call {r['sdpa_call_median_us']}  {r['by_cat']}"
        )
