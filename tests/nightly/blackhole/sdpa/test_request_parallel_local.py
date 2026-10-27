# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LOCAL ONLY (lanes A/B, not for commit): Gemma4 sliding attention for B requests without lanes.

Request-parallel layout: the CP8 axis is split into B groups of 8 / B ranks, and each group runs one ordinary
single-request ring SDPA call over its own 2k chunk (2048 / (8 / B) rows per rank). The B calls run on disjoint
submeshes at the same time. Same work as one B-lane call on CP8: B requests x 2048 tokens, W1024, prefix
RP_PREFIX_CHUNKS chunks.

RP_B=4  RP_R=50  RP_PREFIX_CHUNKS=2  RP_KS=1  RP_QC=128
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
    create_global_semaphores,
    fa_rand,
)

CHUNK, WINDOW, HQ, HKV, D, KC = 2048, 1024, 8, 4, 256, 128
GROUPS = int(os.environ.get("RP_GROUPS", "2"))  # CP groups running at once
PER_GROUP = int(os.environ.get("RP_PER_GROUP", "2"))  # requests each group runs one after another
B = GROUPS * PER_GROUP
REPS = int(os.environ.get("RP_R", "50"))
PREFIX_CHUNKS = int(os.environ.get("RP_PREFIX_CHUNKS", "2"))
KS = int(os.environ.get("RP_KS", "1"))
QC = int(os.environ.get("RP_QC", "128"))
TRACE_REPLAYS = 10


@pytest.mark.timeout(1800)
def test_request_parallel_sliding():
    if not MESH_CONFIG.is_galaxy or MESH_CONFIG.sp_size != 8:
        pytest.skip("needs the TP4 x SP8 Galaxy")
    tp, sp = MESH_CONFIG.tp_size, MESH_CONFIG.sp_size
    group = sp // GROUPS
    assert group * GROUPS == sp, f"{GROUPS} groups must divide CP{sp}"
    local = CHUNK // group
    nhq, nhk = HQ * tp, HKV * tp
    capacity = CHUNK * (PREFIX_CHUNKS + 2)
    halo_tokens = -(-(WINDOW - 1) // KC) * KC
    prefix = PREFIX_CHUNKS * CHUNK

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(tp, sp), trace_region_size=RING_JOINT_TRACE_REGION_SIZE * 4)
    try:
        subs = mesh.create_submeshes(ttnn.MeshShape(tp, group))
        assert len(subs) == GROUPS
        grid = mesh.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(MESH_CONFIG.sdpa_cols, MESH_CONFIG.grid_rows),
            q_chunk_size=QC,
            k_chunk_size=KC,
            exp_approx_mode=False,
            k_splits=KS,
        )
        torch.manual_seed(1234)
        lanes = []
        for sub in subs:
            sub.enable_program_cache()
            sdm = sub.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
            sub.load_sub_device_manager(sdm)
            sub.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
            sems = create_global_semaphores(sub, crs, 0, count=3)
            ckc = ttnn.init_device_compute_kernel_config(
                sub.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

            def upload(host, dtype, dims, sub=sub):
                return ttnn.from_torch(
                    host,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=sub,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(sub, mesh_shape=tuple(sub.shape), dims=dims),
                )

            k = upload(torch.randn(PER_GROUP, nhk, capacity, D), ttnn.bfloat8_b, [1, 2])
            v = upload(torch.randn(PER_GROUP, nhk, capacity, D), ttnn.bfloat8_b, [1, 2])
            reqs = []
            for r in range(PER_GROUP):
                reqs.append(
                    dict(
                        q=upload(fa_rand(1, nhq, CHUNK, D), ttnn.bfloat16, [1, 2]),
                        halos=[
                            upload(torch.zeros(1, nhk, halo_tokens, D), ttnn.bfloat8_b, [1, None]) for _ in range(2)
                        ],
                        meta=[_ring_joint_lane_metadata_tensor(sub, [x]) for x in (r, prefix)],
                    )
                )
            lanes.append(dict(sub=sub, k=k, v=v, reqs=reqs, sems=sems, ckc=ckc))

        def run(l):
            for r in l["reqs"]:
                out = call_sdpa(
                    r["q"],
                    l["k"],
                    l["v"],
                    capacity,
                    is_causal=True,
                    is_balanced=False,
                    p_buf_k=r["halos"][0],
                    p_buf_v=r["halos"][1],
                    program_config=program_config,
                    compute_kernel_config=l["ckc"],
                    ccl_semaphore_handles=l["sems"],
                    num_links=2,
                    sp_axis=1,
                    mesh_device=l["sub"],
                    topology=Topology.Linear,
                    worker_sub_device_id=ttnn.SubDeviceId(0),
                    ccl_column=MESH_CONFIG.ccl_column,
                    sliding_window_size=WINDOW,
                    slot_id=r["meta"][0],
                    kv_actual_isl_tensor=r["meta"][1],
                )
                ttnn.deallocate(out)

        for l in lanes:
            run(l)
        for l in lanes:
            ttnn.synchronize_device(l["sub"])
        for l in lanes:
            tid = ttnn.begin_trace_capture(l["sub"], cq_id=0)
            for _ in range(REPS):
                run(l)
            ttnn.end_trace_capture(l["sub"], tid, cq_id=0)
            ttnn.synchronize_device(l["sub"])
            l["tid"] = tid

        def replay(selected):
            times = []
            for _ in range(TRACE_REPLAYS):
                t0 = time.perf_counter()
                for l in selected:
                    ttnn.execute_trace(l["sub"], l["tid"], cq_id=0, blocking=False)
                for l in selected:
                    ttnn.synchronize_device(l["sub"])
                times.append(time.perf_counter() - t0)
            return statistics.median(times) / REPS * 1e6

        one = replay(lanes[:1])
        together = replay(lanes)
        for l in lanes:
            ttnn.release_trace(l["sub"], l["tid"])
        logger.info(
            f"[req_parallel] B={B} groups={GROUPS}x{PER_GROUP} group=CP{group} rows/rank={local} ks={KS} qc={QC}: "
            f"one group alone {one:.1f} us/layer, all {B} groups at once {together:.1f} us/layer"
        )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
