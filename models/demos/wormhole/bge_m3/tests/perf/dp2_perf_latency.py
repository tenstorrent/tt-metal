# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 DP2 LATENCY-ATTRIBUTION harness (the ~140ms wall-vs-device gap hunt).

Context
-------
Device-kernel sum on each ASIC is ~876-954ms (already ~sub-1s) but the traced
DP2 replay wall is ~1020ms. Host trace-issue is ~0.01ms (measured), so the ~140ms
gap is ON-DEVICE, inside the traced replay. It is EITHER:
  (A) per-program firmware/dispatch gaps across the ~171 programs/forward, or
  (B) cross-device START SKEW (FDMeshCommandQueue::enqueue_trace issues trace
      commands serially over the 2 mesh chips, fd_mesh_command_queue.cpp).

This harness attributes the gap WITHOUT create_submeshes (which crashes ETH on
this rig). It combines:

  Part 1 - HOST-TIMING DECOMPOSITION (always safe, no profiler):
    * full blocking wall (the deliverable metric) with percentiles
    * enqueue-return (host issue cost)   -> confirms host is idle-waiting
    * completion-only (sync after issue)
    * pipelined burst (K non-blocking + 1 sync) -> steady-state per-replay floor
    Verdict: host-bound vs device-bound, and whether pipelining hides the gap.

  Part 2 - PER-DEVICE SPAN ATTRIBUTION (from an ops CSV, safe path):
    Parses a device-profiler ops CSV (one SIGNPOSTED UNTRACED forward, generated
    per .auto workflow step 1) and reports, per DEVICE ID:
      * program count
      * kernel-duration sum
      * first-kernel-start -> last-kernel-end span
      * intra-forward gap = span - sum  (the per-program firmware gap budget)
      * cross-device start/end skew
    Then a CROSSING analysis: at the measured wall-per-program cost, how many
    programs must be fused to cross 1000ms.

Run
---
  # Part 1 (host timing, no profiler):
  source /localdev/gtobar/bge_optimization/local_env.sh
  TT_VISIBLE_DEVICES=0 pytest \
    models/demos/wormhole/bge_m3/tests/perf/dp2_perf_latency.py::test_traced_latency_decomposition -s -q

  # Part 2 (analyze an existing ops CSV; generate one first via the .auto tracy step):
  BGE_OPS_CSV=generated/profiler/reports/<TS>/ops_perf_results_<TS>.csv \
    python3 models/demos/wormhole/bge_m3/tests/perf/dp2_perf_latency.py

No external pip packages required (stdlib only).
"""

import csv
import glob
import os
import statistics
import sys
import time

BATCH = 12
SEQ_LEN = 8192


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: per-device span attribution from a profiler ops CSV
# ─────────────────────────────────────────────────────────────────────────────
def _find_latest_ops_csv():
    env = os.environ.get("BGE_OPS_CSV")
    if env:
        return env
    hits = sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"))
    return hits[-1] if hits else None


def _num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def analyze_ops_csv(path, target_ms=1000.0):
    """Attribute the wall gap: per-device kernel sum, FW span, and inter-op gap.

    The profiler CSV typically has 2 forwards on device 0 (compile + signposted);
    we take the LAST `programs_per_fwd` rows per device as one clean forward.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[part2] empty CSV: {path}")
        return

    cols = {c.lower().strip(): c for c in rows[0].keys()}

    def col(*cands):
        for cand in cands:
            for lc, orig in cols.items():
                if cand in lc:
                    return orig
        return None

    c_dev = col("device id")
    c_dur = col("device kernel duration")
    c_start = col("device fw start", "fw start cycle", "device fw start cycle")
    c_end = col("device fw end", "fw end cycle", "device fw end cycle")
    c_op = col("op code", "op type")
    c_hit = col("program cache hit")

    print(f"\n[part2] ops CSV: {path}")
    print(f"[part2] columns: dev={c_dev!r} dur={c_dur!r} start={c_start!r} end={c_end!r}")
    if not (c_dev and c_dur):
        print("[part2] required columns missing; dumping header for inspection:")
        print("   " + " | ".join(rows[0].keys()))
        return

    by_dev = {}
    for r in rows:
        d = str(r.get(c_dev) or "").strip()
        if not d.isdigit():  # skip blank / summary rows that carry no device id
            continue
        by_dev.setdefault(d, []).append(r)

    print(f"[part2] rows per device: " + ", ".join(f"dev{d}={len(v)}" for d, v in sorted(by_dev.items())))

    # infer programs/forward from device 0 (CSV captures compile + signposted forward)
    dev_ids = sorted(by_dev.keys(), key=int)
    n0 = len(by_dev[dev_ids[0]])
    ppf = n0 // 2 if n0 % 2 == 0 else n0
    print(f"[part2] assuming {ppf} programs/forward (last forward = last {ppf} rows/device)\n")

    spans = {}
    for d in dev_ids:
        drows = by_dev[d][-ppf:]
        durs = [ns for ns in (_num(r.get(c_dur)) for r in drows) if ns is not None]
        ksum = sum(durs) / 1e6  # ns -> ms
        starts = [s for s in (_num(r.get(c_start)) for r in drows) if s is not None] if c_start else []
        ends = [e for e in (_num(r.get(c_end)) for r in drows) if e is not None] if c_end else []
        span = (max(ends) - min(starts)) / 1e6 if starts and ends else None  # ns -> ms
        spans[d] = dict(n=len(drows), ksum=ksum, span=span,
                        first=min(starts) if starts else None,
                        last=max(ends) if ends else None)
        gap = (span - ksum) if span is not None else float("nan")
        print(f"  device {d}: programs={len(drows):3d}  kernel_sum={ksum:8.2f}ms  "
              f"fw_span={span if span is None else round(span,2)}ms  intra_gap={gap:7.2f}ms  "
              f"(~{gap/max(len(drows),1):.3f}ms/program)")

    # cross-device skew (needs correlated clocks; only meaningful if same clock domain)
    if all(spans[d]["first"] is not None for d in dev_ids) and len(dev_ids) > 1:
        firsts = {d: spans[d]["first"] for d in dev_ids}
        lasts = {d: spans[d]["last"] for d in dev_ids}
        start_skew = (max(firsts.values()) - min(firsts.values())) / 1e6
        end_skew = (max(lasts.values()) - min(lasts.values())) / 1e6
        union = (max(lasts.values()) - min(firsts.values())) / 1e6
        print(f"\n  cross-device start_skew={start_skew:.2f}ms  end_skew={end_skew:.2f}ms  "
              f"union_span={union:.2f}ms")
        print("  NOTE: skew is only valid if both ASIC profiler clocks are correlated;")
        print("        if uncorrelated, treat start/end skew as UNKNOWN (per guidance Finding 1).")

    # crossing analysis
    slow = max(spans.values(), key=lambda s: (s["span"] or s["ksum"]))
    print("\n[part2] CROSSING ANALYSIS")
    print(f"  slowest-device kernel_sum : {slow['ksum']:.1f}ms")
    if slow["span"]:
        print(f"  slowest-device fw_span    : {slow['span']:.1f}ms  (this is the on-device floor)")
        over = slow["span"] - target_ms
        per_prog_gap = (slow["span"] - slow["ksum"]) / max(slow["n"], 1)
        if over > 0 and per_prog_gap > 0:
            need = over / per_prog_gap
            print(f"  to reach {target_ms:.0f}ms by fusing programs: cut ~{need:.1f} programs "
                  f"(~{need/24:.2f}/layer) at {per_prog_gap:.2f}ms gap/program")
    print("  (compare fw_span to the traced wall from Part 1: wall - fw_span = mesh/completion overhead)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: traced-replay host-timing decomposition (pytest, uses device fixture)
# ─────────────────────────────────────────────────────────────────────────────
def _pctile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, int(q * (len(xs) - 1)))
    return xs[i]


def _run_part1(mesh_device, iters=30, burst=16):
    import ttnn
    from loguru import logger
    from models.demos.wormhole.bge_m3.tests.perf.dp2_perf import _to_batchsharded_tensors, prepare_inputs
    from models.demos.wormhole.bge_m3.tt.common import create_tt_model

    args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=BATCH,
        max_seq_len=SEQ_LEN,
        dtype=ttnn.bfloat8_b,
        data_parallel=True,
        use_experimental_encoder_sdpa=True,
        encoder_sdpa_q256_vbf4=True,
        use_qkv_scatter_matmul=True,
    )
    assert model._data_parallel, "DP mode not active"

    inp = prepare_inputs(args.tokenizer, BATCH, SEQ_LEN, args.pad_token_id)
    dev = _to_batchsharded_tensors(inp, mesh_device, device=True)
    out = model.forward(**dev)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)

    model.capture_trace(**dev, mesh_device=mesh_device, cq_id=0)
    tid, tdev = model._trace_id, model._trace_device
    for _ in range(5):
        ttnn.execute_trace(tdev, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(tdev)

    # (1) full blocking wall — the deliverable metric
    full = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ttnn.execute_trace(tdev, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(tdev)
        full.append((time.perf_counter() - t0) * 1e3)

    # (2) enqueue-return (host issue cost) vs (3) completion-only, same call split
    enq, comp = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        ttnn.execute_trace(tdev, tid, cq_id=0, blocking=False)
        t1 = time.perf_counter()
        ttnn.synchronize_device(tdev)
        t2 = time.perf_counter()
        enq.append((t1 - t0) * 1e3)
        comp.append((t2 - t1) * 1e3)

    # (4) pipelined burst — steady-state per-replay floor (hides per-call latency)
    ttnn.synchronize_device(tdev)
    t0 = time.perf_counter()
    for _ in range(burst):
        ttnn.execute_trace(tdev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(tdev)
    burst_ms = (time.perf_counter() - t0) * 1e3
    per_replay = burst_ms / burst

    model.release_trace()

    def line(name, xs):
        return (f"  {name:<26} min={min(xs):8.2f}  p50={_pctile(xs,0.5):8.2f}  "
                f"p90={_pctile(xs,0.9):8.2f}  max={max(xs):8.2f}  "
                f"std={statistics.pstdev(xs):5.2f}")

    logger.info("")
    logger.info("=" * 78)
    logger.info("  DP2 TRACED-REPLAY LATENCY DECOMPOSITION  (ms)")
    logger.info("=" * 78)
    logger.info(line("full blocking wall", full))
    logger.info(line("enqueue-return (issue)", enq))
    logger.info(line("completion-only (sync)", comp))
    logger.info(f"  pipelined {burst}-burst        per_replay={per_replay:8.2f}  "
                f"({1000.0/per_replay:.3f} req/s)")
    logger.info("-" * 78)
    wall = min(full)
    logger.info(f"  host issue is {min(enq):.2f}ms => host is {'IDLE-WAITING (device-bound)' if min(enq) < 5 else 'BUSY (host-bound)'}")
    logger.info(f"  pipelining {'DOES NOT hide' if per_replay > wall*0.97 else 'HIDES'} the gap "
                f"(burst per-replay {per_replay:.1f} vs blocking {wall:.1f})")
    logger.info(f"  => gap is {'genuine on-device replay time (fuse programs / cut skew)' if per_replay > wall*0.97 else 'per-call host/completion latency (pipeline it)'}")
    logger.info("=" * 78)

    # Part 2 inline if a CSV is available
    csv_path = _find_latest_ops_csv()
    if csv_path:
        analyze_ops_csv(csv_path)
    else:
        logger.info("  [part2] no ops CSV found; generate one (see .auto workflow step 1) "
                    "then set BGE_OPS_CSV=... to attribute per-device span.")
    return wall


try:
    import pytest
    import ttnn

    @pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["dp2_n300"])
    @pytest.mark.parametrize(
        "device_params",
        [{
            "trace_region_size": 50_000_000,
            "num_command_queues": 1,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }],
        indirect=True,
    )
    def test_traced_latency_decomposition(mesh_device):
        _run_part1(mesh_device)

except ImportError:
    # Part 2 (__main__ CSV analysis) can run without ttnn/pytest available.
    pass


if __name__ == "__main__":
    # Part 2 only: analyze an ops CSV without opening a device.
    path = _find_latest_ops_csv()
    if not path:
        print("No ops CSV found. Set BGE_OPS_CSV=<path> or generate one via the .auto tracy step.")
        sys.exit(1)
    analyze_ops_csv(path)
