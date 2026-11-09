# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Non-blocking prefill, P0 go/stop microbenchmark (one Blackhole chip, no fabric).

Question: if a rank's Tensix grid is split into a DECODE sub-device (rows [0, R), ~44-48 cores, command queue 0) and a
PREFILL sub-device (rows [R, grid_y), command queue 1), how much does each side slow down when both run at once, and what
does loading / clearing the split cost? (scratchpad non_blocking_prefill/DESIGN_NOTES.md, option b+d.)

Stand-ins (Qwen3.8-27B TP8 per-chip shapes, DRAM-interleaved weights, traced):
  decode : DEC_OPS x matmul [32, 5120] x [5120, 2176] bf8 weights (the gate/up shape; DRAM-bound like decode)
  prefill: PF_OPS  x matmul [2048, 5120] x [5120, 2176] bf8 weights (a 2K-chunk projection; compute-bound)

Phases (each reports device wall ms from host timing around event syncs, median of REPS):
  A  full grid, no sub-device manager: decode trace alone, prefill trace alone          (today's reference)
  B  split loaded: decode trace alone on SD0/CQ0, prefill trace alone on SD1/CQ1
  C  split loaded, concurrent: prefill trace on CQ1 (non-blocking) + decode trace on CQ0 back to back for the same span;
     decode per-iteration time while prefill runs, prefill time while decode runs
  D  load + clear of the split manager with both traces resident (host wall, synchronized)

GO / STOP (the design notes' sim thresholds; adaptive split per prefill):
  decode_slowdown = C.decode_iter / A.decode_iter, prefill_slowdown = C.prefill / A.prefill, switch = D.load+clear
  GO   : decode_slowdown <= 1.20 and prefill_slowdown <= 1.50 and switch <= 2 ms
  STOP : decode_slowdown >= 1.30 or prefill_slowdown >= 1.80 or switch >= 10 ms
  else : MAYBE (re-run with R in 3..5 / real decode configs before committing to the 40-60 engineer-day build)

Run in the dev pod with the inference server stopped (sleep mode), one chip, no fabric:
  TT_VISIBLE_DEVICES=0 python models/demos/blackhole/qwen36/tests/bench_nbp_p0.py --rows 4 --reps 20 --out /tmp/nbp_p0.json
"""
import argparse
import json
import statistics
import time

import torch
from loguru import logger

import ttnn

K, N = 5120, 2176


def _mk(device, shape, dtype, scale=1.0):
    return ttnn.from_torch(
        torch.randn(*shape, dtype=torch.bfloat16) * scale,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _ckc(device, fidelity):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )


def _chain(x, w, n, grid, sd, ckc):
    """n matmuls x @ w; the output of one is NOT fed to the next (shapes differ), so each re-reads x (like independent
    projections). Outputs are deallocated inside the trace body so the trace's footprint stays one output."""
    kw = dict(memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckc)
    if grid is not None:
        kw["core_grid"] = grid
    if sd is not None:
        kw["sub_device_id"] = sd
    out = None
    for _ in range(n):
        if out is not None:
            ttnn.deallocate(out)
        out = ttnn.matmul(x, w, **kw)
    return out


def _capture(device, cq, fn):
    """Compile then capture ``fn`` on command queue ``cq`` (ops dispatch to the thread's current queue)."""
    with ttnn.command_queue(cq):
        ttnn.deallocate(fn())  # compile
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=cq)
        out = fn()
        ttnn.end_trace_capture(device, tid, cq_id=cq)
    ttnn.synchronize_device(device)
    return tid, out


def _sync_cq(device, cq):
    ttnn.event_synchronize(ttnn.record_event(device, cq))


def _time_trace(device, tid, cq, reps, iters=1):
    ts = []
    for _ in range(reps):
        _sync_cq(device, cq)
        t0 = time.perf_counter()
        for _ in range(iters):
            ttnn.execute_trace(device, tid, cq_id=cq, blocking=False)
        _sync_cq(device, cq)
        ts.append((time.perf_counter() - t0) * 1e3 / iters)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4, help="decode sub-device = worker rows [0, rows)")
    ap.add_argument("--dec-cols", type=int, default=11, help="decode matmul grid width (x); 11 x 4 = 44 cores")
    ap.add_argument("--dec-ops", type=int, default=64)
    ap.add_argument("--pf-ops", type=int, default=16)
    ap.add_argument("--pf-rows", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="/tmp/nbp_p0.json")
    a = ap.parse_args()

    device = ttnn.open_device(device_id=0, num_command_queues=2, trace_region_size=64 << 20)
    res = dict(args=vars(a))
    try:
        g = device.compute_with_storage_grid_size()
        gx, gy = g.x, g.y
        assert 0 < a.rows < gy and a.dec_cols <= gx
        res["grid"] = [gx, gy]
        hifi2, lofi = _ckc(device, ttnn.MathFidelity.HiFi2), _ckc(device, ttnn.MathFidelity.LoFi)
        x_dec = _mk(device, (1, 1, 32, K), ttnn.bfloat16)
        x_pf = _mk(device, (1, 1, a.pf_rows, K), ttnn.bfloat16)
        w_dec = _mk(device, (1, 1, K, N), ttnn.bfloat8_b, 0.02)
        w_pf = _mk(device, (1, 1, K, N), ttnn.bfloat8_b, 0.02)

        # ---- A: full grid, no manager (decode on its usual 44 cores; prefill on the full grid) ----
        dgrid = ttnn.CoreGrid(x=a.dec_cols, y=a.rows)
        fgrid = ttnn.CoreGrid(x=gx, y=gy)
        tA_d, oA_d = _capture(device, 0, lambda: _chain(x_dec, w_dec, a.dec_ops, dgrid, None, lofi))
        tA_p, oA_p = _capture(device, 0, lambda: _chain(x_pf, w_pf, a.pf_ops, fgrid, None, hifi2))
        res["A_decode_ms"] = _time_trace(device, tA_d, 0, a.reps)
        res["A_prefill_ms"] = _time_trace(device, tA_p, 0, a.reps)
        ref_d = ttnn.to_torch(oA_d)
        ttnn.release_trace(device, tA_d)
        ttnn.release_trace(device, tA_p)
        logger.info(f"A full grid: decode {res['A_decode_ms']:.3f} ms, prefill {res['A_prefill_ms']:.3f} ms")

        # ---- B: split loaded, each side alone ----
        sd0 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, a.rows - 1))})
        sd1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, a.rows), ttnn.CoreCoord(gx - 1, gy - 1))})
        mgr = device.create_sub_device_manager([ttnn.SubDevice([sd0]), ttnn.SubDevice([sd1])], 0)
        t0 = time.perf_counter()
        device.load_sub_device_manager(mgr)
        ttnn.synchronize_device(device)
        res["load_first_ms"] = (time.perf_counter() - t0) * 1e3
        SD0, SD1 = ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)
        pgrid = ttnn.CoreGrid(x=gx, y=gy - a.rows)
        device.set_sub_device_stall_group([SD0])
        tB_d, oB_d = _capture(device, 0, lambda: _chain(x_dec, w_dec, a.dec_ops, dgrid, SD0, lofi))
        device.set_sub_device_stall_group([SD1])
        tB_p, oB_p = _capture(device, 1, lambda: _chain(x_pf, w_pf, a.pf_ops, pgrid, SD1, hifi2))
        device.reset_sub_device_stall_group()
        res["B_decode_ms"] = _time_trace(device, tB_d, 0, a.reps)
        res["B_prefill_ms"] = _time_trace(device, tB_p, 1, a.reps)
        res["decode_bit_identical_split_vs_full"] = bool(torch.equal(ttnn.to_torch(oB_d), ref_d))
        logger.info(f"B split alone: decode {res['B_decode_ms']:.3f} ms, prefill {res['B_prefill_ms']:.3f} ms")

        # ---- C: concurrent ----
        conc_d, conc_p = [], []
        for _ in range(a.reps):
            _sync_cq(device, 0)
            _sync_cq(device, 1)
            tp0 = time.perf_counter()
            ttnn.execute_trace(device, tB_p, cq_id=1, blocking=False)
            ev_p = ttnn.record_event(device, 1)
            n_dec = max(1, int(res["B_prefill_ms"] / max(res["B_decode_ms"], 1e-3)))
            td0 = time.perf_counter()
            for _ in range(n_dec):
                ttnn.execute_trace(device, tB_d, cq_id=0, blocking=False)
            _sync_cq(device, 0)
            conc_d.append((time.perf_counter() - td0) * 1e3 / n_dec)
            ttnn.event_synchronize(ev_p)
            conc_p.append((time.perf_counter() - tp0) * 1e3)
        res["C_decode_iter_ms"] = statistics.median(conc_d)
        res["C_prefill_ms"] = statistics.median(conc_p)
        res["decode_bit_identical_concurrent"] = bool(torch.equal(ttnn.to_torch(oB_d), ref_d))
        logger.info(f"C concurrent: decode/iter {res['C_decode_iter_ms']:.3f} ms, prefill {res['C_prefill_ms']:.3f} ms")

        # ---- D: load / clear cost with traces resident ----
        lc = []
        for _ in range(a.reps):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            device.clear_loaded_sub_device_manager()
            device.load_sub_device_manager(mgr)
            ttnn.synchronize_device(device)
            lc.append((time.perf_counter() - t0) * 1e3)
        res["D_clear_plus_load_ms"] = statistics.median(lc)
        ttnn.release_trace(device, tB_d)
        ttnn.release_trace(device, tB_p)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(mgr)

        ds = res["C_decode_iter_ms"] / res["A_decode_ms"]
        ps = res["C_prefill_ms"] / res["A_prefill_ms"]
        sw = res["D_clear_plus_load_ms"]
        res.update(decode_slowdown=ds, prefill_slowdown=ps, switch_ms=sw)
        if ds <= 1.20 and ps <= 1.50 and sw <= 2.0:
            verdict = "GO"
        elif ds >= 1.30 or ps >= 1.80 or sw >= 10.0:
            verdict = "STOP"
        else:
            verdict = "MAYBE"
        res["verdict"] = verdict
        logger.info(f"decode x{ds:.3f}  prefill x{ps:.3f}  switch {sw:.2f} ms  -> {verdict}")
    finally:
        json.dump(res, open(a.out, "w"), indent=2)
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
