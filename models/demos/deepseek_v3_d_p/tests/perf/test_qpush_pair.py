# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Two-op repro for the Q-stem || sparse KV push overlap (GLM_QPUSH_OVERLAP, tt/mla/mla.py).

Manager = the model's sparse-MLA overlap manager (TT_CCL "auto" profile: SD0 = top-k rows 2-9, SD1 = gather rows 0-1).
  qb_full  the seqq q_b matmul as the model runs it today: [1,1,160,2048] x [2048,16384] bfp8 DRAM width-sharded,
           1D mcast_in0 on 11x10 (per_core_N 5, in0_block_w 2), L1 output, default manager
  qb_off   the same matmul on SD0: 11x8 anchored at (0,2) through allowed_worker_cores, per_core_N 6, into a
           preallocated L1 output (what _qpush_region issues)
  kvg      stand-in for the push (same fabric pattern: full-mesh high_bw_all_gather, cluster_axis=None, 2 links) on SD1
           with sub_core_grids = SD1: [1,1,160,384] bf16 ROW_MAJOR per chip -> [1,1,5120,384] (3.9 MB in per chip)
Checks: qb_off output bit-identical to qb_full (torch.equal over all chips); traced loops of QP_N iterations, QP_R
replays each, a watchdog (QP_WATCHDOG_S) exits 124 when one replay hangs. Reports per-iteration medians of qb_full,
qb_off, kvg alone, and kvg || qb_off issued back to back (no ordering between them).
"""

import os
import threading
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.tt_ccl import clear_tt_ccl_cache, get_tt_ccl

NUM_LINKS = 2


def _watchdog(deadline_box, label_box):
    while True:
        time.sleep(0.25)
        d = deadline_box[0]
        if d is not None and time.monotonic() > d:
            logger.error(f"[qpush pair] WATCHDOG: replay of {label_box[0]} exceeded its deadline -> hang; exiting 124")
            os._exit(124)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=12352, l1_small_size=1536, trace_region_size=64 * 1024 * 1024),
            id="torus-xy-8x4",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(900)
def test_qpush_pair(mesh_device, device_params):
    n, r = int(os.environ.get("QP_N", "16")), int(os.environ.get("QP_R", "20"))
    wd = float(os.environ.get("QP_WATCHDOG_S", "15"))
    rows, k, nn, kv_w = 160, 2048, 16384, int(os.environ.get("QP_KVG_W", "384"))
    tt_ccl = get_tt_ccl(mesh_device)
    res = tt_ccl.get_sparse_mla_overlap_resources("auto")
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    torch.manual_seed(0)

    qr = ttnn.from_torch(
        torch.randn(1, 1, rows, k),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, k, nn) * 0.02,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    dg = mesh_device.dram_grid_size()
    banks = dg.x * dg.y
    ws = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))}),
            [k, nn // banks],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    w = ttnn.to_memory_config(w, ws)
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    pc_full = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=5,
        per_core_M=5,
        per_core_N=5,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    box = res.topk_core_grid.bounding_box()
    gy = box.end.y - box.start.y + 1
    pc_off = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(11, gy),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=6,
        per_core_M=5,
        per_core_N=6,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
        allowed_worker_cores=ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(box.start.x, box.start.y), ttnn.CoreCoord(box.start.x + 10, box.end.y))]
        ),
    )

    def qb(pc, out=None):
        return ttnn.linear(
            qr,
            w,
            program_config=pc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ckc,
            optional_output_tensor=out,
        )

    kv_in = ttnn.from_torch(
        torch.randn(1, 1, rows, kv_w),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    dist = ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1])
    coords = [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(dist)]
    kv_in.update_tensor_topology(ttnn.TensorTopology(dist, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords))
    kv_out = ttnn.from_torch(
        torch.zeros(1, 1, rows * mesh_device.get_num_devices(), kv_w),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )

    def kvg():
        ttnn.experimental.high_bw_all_gather(
            kv_in,
            dim=2,
            output_tensor=kv_out,
            cluster_axis=None,
            num_links=NUM_LINKS,
            subdevice_id=res.gather_subdevice_id,
            sub_core_grids=res.gather_core_grid,
        )

    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    deadline, label = [None], [""]
    threading.Thread(target=_watchdog, args=(deadline, label), daemon=True).start()

    def timed(name, body):
        """Trace `body` QP_N times, replay QP_R times; median us per iteration."""
        outs = []
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(n):
            o = body()
            if o is not None:
                outs.append(o)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        label[0] = name
        times = []
        for _ in range(r):
            deadline[0] = time.monotonic() + wd
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            times.append((time.perf_counter() - t0) * 1e6 / n)
            deadline[0] = None
        ttnn.release_trace(mesh_device, tid)
        for o in outs:
            ttnn.deallocate(o)
        times.sort()
        med = times[len(times) // 2]
        logger.info(f"[qpush pair] RESULT {name}: OK, per iteration median {med:.1f} us (n={n}, r={r})")
        return med

    # Default manager: today's q_b (reference output + timing).
    ref = qb(pc_full)
    ttnn.synchronize_device(mesh_device)
    ref_host = ttnn.to_torch(ref, mesh_composer=comp)
    ttnn.deallocate(ref)
    t_full = timed("qb_full (11x10, default manager)", lambda: qb(pc_full))

    mesh_device.load_sub_device_manager(res.manager_id)
    try:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows, nn]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.L1_MEMORY_CONFIG
        )
        kvg()
        qb(pc_off, out)
        ttnn.synchronize_device(mesh_device)
        off_host = ttnn.to_torch(out, mesh_composer=comp)
        same = torch.equal(ref_host, off_host)
        logger.info(
            f"[qpush pair] RESULT q_b on SD0 (11x{gy} at ({box.start.x},{box.start.y}), per_core_N 6) bit-identical to "
            f"11x10: {same} (max abs diff {(ref_host.float() - off_host.float()).abs().max().item():.3e})"
        )
        t_off = timed("qb_off (SD0)", lambda: qb(pc_off, out))
        t_kvg = timed("kvg (SD1, full-mesh gather)", lambda: kvg())

        def pair():
            kvg()
            qb(pc_off, out)

        t_pair = timed("kvg || qb_off", pair)
        logger.info(
            f"[qpush pair] SUMMARY qb_full {t_full:.1f} | qb_off {t_off:.1f} | kvg {t_kvg:.1f} | pair {t_pair:.1f} us "
            f"(serial would be {t_full + t_kvg:.1f}; ideal overlap max = {max(t_off, t_kvg):.1f})"
        )
        ttnn.deallocate(out)
        assert same, "q_b on the top-k sub-device is not bit-identical to the full-grid q_b"
    finally:
        mesh_device.clear_loaded_sub_device_manager()
        clear_tt_ccl_cache()  # drains, resets and removes the overlap manager before the mesh closes
