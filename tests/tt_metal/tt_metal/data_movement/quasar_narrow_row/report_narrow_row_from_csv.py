# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Report the Quasar narrow-row pack-untilize results (test id 918) from the profiler CSV.

Why this exists rather than the shared dm harness: on ``emu-quasar-1x3`` there are no
fast-dispatch cores, so the test runs slow-dispatch, and under slow dispatch the profiler
never increments ``run host ID`` -- every point is stamped ``run_host_id = 0`` and the shared
harness collapses them into one. The data is still there in CSV order, so this walks the file
instead. Same stopgap as ``quasar_cache_perf/plot_cache_write_from_csv.py``.

Two zones per program run, and they are deliberately named apart so the two stages can be
told from each other in one log:

  PACK_UNTILIZE   stage 1, on the TRISC pack thread -- the stock whole-tile HW untilize
  COMPACT         stage 2, on a DM core            -- the iDMA/NOC row compaction

Each ZONE_START/ZONE_END pair is followed by that run's stamps, from the same RISC, in a
fixed order ending with the flush stamp (``Tile columns`` for stage 1, ``Number of channels``
for stage 2). Pending state is tracked PER RISC, because the two stages interleave in the
file.

Usage
-----
    python tests/tt_metal/tt_metal/data_movement/quasar_narrow_row/report_narrow_row_from_csv.py \
        [--csv generated/profiler/.logs/profile_log_device.csv]
"""

import argparse
import sys

TEST_ID = 918

# CSV column indices (0-based), per the tracy device-log header:
# PCIe, core_x, core_y, RISC type, timer_id, time, data, run host ID, trace id,
# trace id counter, zone name, type, source line, source file, meta
COL_RISC = 3
COL_TIME = 5
COL_DATA = 6
COL_ZONE = 10
COL_TYPE = 11

ENGINE_IDMA_SCATTER = 0
ENGINE_IDMA_PER_ROW = 1
ENGINE_NOC_PER_ROW = 2
ENGINE_NAMES = {
    ENGINE_IDMA_SCATTER: "iDMA scatter-list",
    ENGINE_IDMA_PER_ROW: "iDMA per-row",
    ENGINE_NOC_PER_ROW: "NOC per-row",
}

# tt-llk reference points, single 32x32 Float16 tile, loop_factor=32, DestSync.Half.
# Normal whole-tile HW pack-untilize, and RV_PACR narrow-row by kept width.
TT_LLK_NORMAL_CYC_PER_TILE = 77.4
RV_PACR_CYC_PER_TILE = {8: 1439.5, 16: 1470.4, 24: 2418.0, 32: 2029.6}


def reconstruct(csv_path):
    """Return (stage1_runs, stage2_runs) in execution order, recovered from CSV order."""
    with open(csv_path) as f:
        rows = [line.rstrip("\n").split(",") for line in f]

    stage1, stage2 = [], []
    # Per-RISC pending state: the two stages run on different processors and their records
    # interleave in the file, so a single global "current run" would mix their stamps.
    start = {}
    cur = {}
    kind = {}

    for r in rows:
        if len(r) <= COL_TYPE:
            continue
        risc, zone, typ = r[COL_RISC], r[COL_ZONE], r[COL_TYPE]
        if typ == "ZONE_START" and zone in ("PACK_UNTILIZE", "COMPACT"):
            start[risc] = int(r[COL_TIME])
            kind[risc] = zone
        elif typ == "ZONE_END" and zone in ("PACK_UNTILIZE", "COMPACT") and start.get(risc) is not None:
            cur[risc] = {"dur": int(r[COL_TIME]) - start[risc]}
            start[risc] = None
        elif typ == "TS_DATA" and cur.get(risc):
            val = int(r[COL_DATA])
            run = cur[risc]
            if zone == "Test id":
                run["test_id"] = val
            elif zone == "Number of transactions":
                run["iters"] = val
            elif zone == "Transaction size in bytes":
                run["row_bytes"] = val
            elif zone == "Number of rows":
                run["rows"] = val
            elif zone == "Packet size in bytes":
                run["packet"] = val
            elif zone == "Engine mode":
                run["engine"] = val
            elif zone == "Tile columns":
                # flush stamp for stage 1
                run["ct_dim"] = val
                if run.get("test_id") == TEST_ID:
                    stage1.append(run)
                cur[risc] = None
            elif zone == "Number of channels":
                # flush stamp for stage 2
                run["channels"] = val
                if run.get("test_id") == TEST_ID:
                    stage2.append(run)
                cur[risc] = None
    return stage1, stage2


RESULTS_CSV = "generated/profiler/.logs/narrow_row_results.csv"


def load_verdicts(path):
    """Per-run pass/fail, written by the host in execution order (see record_result).

    The profiler log knows timing and nothing else, so without this a wrong computation's
    cycle count reads exactly like a result. Returns a list of bools in run order, or None if
    the file is absent -- in which case the report says so rather than implying everything
    passed.
    """
    try:
        with open(path) as f:
            rows = [line.rstrip("\n").split(",") for line in f]
    except FileNotFoundError:
        return None
    if not rows or rows[0][0] != "ct_dim":
        return None
    out = []
    for r in rows[1:]:
        if len(r) >= 9:
            out.append(r[-1] == "1")
    return out


def report_vs_workaround(stage2, verdicts=None):
    """Group the compaction runs by shape and put every engine next to the NOC baseline.

    This is the question the test exists to answer: the NOC per-row read IS the current
    workaround, so a shape is only worth adopting iDMA for if some iDMA row beats engine 2's
    row for that same shape. Grouping by (rows, B/row) is what makes them comparable -- the
    cost is strongly size-dependent (descriptor-bound and flat below ~80 B/row, then roughly
    one cycle per 16-20 B), so comparing engines across different shapes means nothing.
    """
    groups = {}
    for i, run in enumerate(stage2):
        key = (run.get("rows"), run.get("row_bytes"))
        # A repeat of the same configuration is a run-to-run stability check; keep both and
        # average, rather than letting the later one silently win.
        cfg = (run.get("engine"), run.get("channels"), run.get("packet"))
        per_pass = run["dur"] / run["iters"] if run.get("iters") else float("nan")
        # verdicts is in the same execution order as the zones, so index i lines up.
        ok = verdicts[i] if verdicts is not None and i < len(verdicts) else None
        groups.setdefault(key, {}).setdefault(cfg, []).append((per_pass, ok))

    if verdicts is None:
        print("NOTE: no verdict file -- correctness unknown, every number below could be a wrong answer.")
    elif len(verdicts) != len(stage2):
        print(f"NOTE: {len(verdicts)} verdicts vs {len(stage2)} timed runs -- not joining; correctness unknown.")
        verdicts = None

    print("=== vs the current workaround (NOC per-row), grouped by shape ===")
    verdict_rows = []
    for (rows, row_bytes), cfgs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        # Only a CORRECT baseline is a baseline. A wrong one would flatter everything.
        noc_vals = [
            v for (e, _c, _p), vals in cfgs.items() if e == ENGINE_NOC_PER_ROW for v, ok in vals if ok is not False
        ]
        baseline = sum(noc_vals) / len(noc_vals) if noc_vals else None
        # matrix_w -> the tile count that covers it -> how much of the padded row was junk.
        matrix_w = (row_bytes or 0) // 2
        ct = max(1, (matrix_w + 31) // 32)
        pad_bytes = ct * 32 * 2
        waste = 100.0 * (pad_bytes - row_bytes) / pad_bytes if pad_bytes else 0.0
        print(
            f"  {rows} rows x {row_bytes} B  (matrix_w {matrix_w}, ct_dim {ct}, "
            f"padded {pad_bytes} B/row, {waste:.1f}% of the padded row is junk)"
        )
        if baseline is None:
            print("    no correct NOC baseline for this shape -- nothing to compare against")
        best = None
        for (engine, channels, packet), vals in sorted(cfgs.items()):
            avg = sum(v for v, _ in vals) / len(vals)
            bad = sum(1 for _, ok in vals if ok is False)
            unknown = any(ok is None for _, ok in vals)
            split = "" if packet in (None, row_bytes) else f", split {packet} B"
            tag = f"{ENGINE_NAMES.get(engine, '?')} ch={channels}{split}"
            if bad:
                status = f"  !! WRONG OUTPUT ({bad}/{len(vals)}) -- timing meaningless"
            elif unknown:
                status = "  (correctness unknown)"
            elif baseline is None or avg <= 0:
                status = ""
            elif engine == ENGINE_NOC_PER_ROW:
                status = "  <- baseline"
            else:
                status = f"  {baseline / avg:.2f}x vs NOC" + ("" if baseline > avg else "  (SLOWER)")
            n = f" (n={len(vals)})" if len(vals) > 1 else ""
            print(f"    {tag:<44} {avg:>8.1f} cyc/pass {avg / rows:>7.2f} cyc/row{n}{status}")
            # Headline candidate: correct, iDMA, and beats the baseline.
            if not bad and not unknown and engine != ENGINE_NOC_PER_ROW and baseline is not None:
                if best is None or avg < best[1]:
                    best = (tag, avg)
        if best is not None and baseline is not None:
            verdict_rows.append((row_bytes, matrix_w, waste, baseline, best[0], best[1], baseline / best[1]))

    if verdict_rows:
        print()
        print("=== verdict: best VERIFIED iDMA configuration vs the workaround, by row size ===")
        print(f"{'B/row':>7} {'matrix_w':>9} {'junk%':>6} {'NOC cyc':>9} {'best iDMA':>10}  {'speedup':>8}  config")
        for row_bytes, matrix_w, waste, base, tag, avg, ratio in verdict_rows:
            flag = "" if ratio > 1.0 else "   <- workaround wins"
            print(f"{row_bytes:>7} {matrix_w:>9} {waste:>5.1f}% {base:>9.1f} {avg:>10.1f}  {ratio:>7.2f}x  {tag}{flag}")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="generated/profiler/.logs/profile_log_device.csv")
    ap.add_argument("--results", default=RESULTS_CSV, help="per-run pass/fail written by the test")
    args = ap.parse_args()

    try:
        stage1, stage2 = reconstruct(args.csv)
    except FileNotFoundError:
        print(f"no CSV at {args.csv} -- run with TT_METAL_DEVICE_PROFILER=1 or the zones are no-ops")
        return 1

    if not stage1 and not stage2:
        print(f"no test-{TEST_ID} runs found in {args.csv}")
        return 1

    print(f"=== stage 1: HW pack_untilize (PACK_UNTILIZE zone), {len(stage1)} runs ===")
    print(f"{'ct_dim':>7} {'passes':>7} {'cycles':>9} {'cyc/tile':>9}   vs tt-llk {TT_LLK_NORMAL_CYC_PER_TILE}")
    for run in stage1:
        tiles = run["iters"] * run["ct_dim"]
        per_tile = run["dur"] / tiles if tiles else float("nan")
        print(
            f"{run['ct_dim']:>7} {run['iters']:>7} {run['dur']:>9} {per_tile:>9.1f}"
            f"   {per_tile / TT_LLK_NORMAL_CYC_PER_TILE:>6.2f}x"
        )

    print()
    print(f"=== stage 2: row compaction (COMPACT zone), {len(stage2)} runs ===")
    print(
        f"{'engine':>18} {'rows':>5} {'B/row':>7} {'packet':>7} {'ch':>3} "
        f"{'iters':>6} {'cyc/pass':>9} {'cyc/row':>8} {'B/cyc':>7}"
    )
    for run in stage2:
        per_pass = run["dur"] / run["iters"] if run.get("iters") else float("nan")
        rows = run.get("rows", 0)
        per_row = per_pass / rows if rows else float("nan")
        bpc = (rows * run.get("row_bytes", 0)) / per_pass if per_pass else float("nan")
        print(
            f"{ENGINE_NAMES.get(run.get('engine'), '?'):>18} {rows:>5} {run.get('row_bytes', 0):>7} "
            f"{run.get('packet', 0):>7} {run.get('channels', 0):>3} {run['iters']:>6} "
            f"{per_pass:>9.1f} {per_row:>8.2f} {bpc:>7.2f}"
        )

    print()
    report_vs_workaround(stage2, load_verdicts(args.results))

    # Pair by index: each program run emits exactly one zone of each kind, and programs run
    # sequentially, so the Nth of each belongs to the same run. If the counts differ, the
    # assumption broke and the totals would be nonsense, so say so instead of printing them.
    print()
    if len(stage1) != len(stage2):
        print(
            f"NOT pairing stages: {len(stage1)} stage-1 runs vs {len(stage2)} stage-2 runs. "
            "Every run should emit one of each -- a mismatch means the CSV order assumption broke."
        )
        return 0

    print("=== end to end: stage 1 + stage 2, per narrow-row untilize of one tile-row ===")
    print(
        f"{'ct_dim':>7} {'matrix_w':>9} {'engine':>18} {'ch':>3} "
        f"{'pack cyc':>9} {'compact':>9} {'total':>9} {'cyc/tile':>9}   vs RV_PACR"
    )
    for s1, s2 in zip(stage1, stage2):
        ct = s1["ct_dim"]
        pack = s1["dur"] / s1["iters"] if s1.get("iters") else float("nan")
        compact = s2["dur"] / s2["iters"] if s2.get("iters") else float("nan")
        total = pack + compact
        per_tile = total / ct
        # matrix_w in datums, from the compaction's per-row byte count (Float16_b).
        matrix_w = s2.get("row_bytes", 0) // 2
        last_w = matrix_w - (ct - 1) * 32
        ref = RV_PACR_CYC_PER_TILE.get(last_w)
        cmp_str = f"{ref / per_tile:>6.1f}x faster" if ref else "   (no RV_PACR point)"
        print(
            f"{ct:>7} {matrix_w:>9} {ENGINE_NAMES.get(s2.get('engine'), '?'):>18} "
            f"{s2.get('channels', 0):>3} {pack:>9.1f} {compact:>9.1f} {total:>9.1f} {per_tile:>9.1f}   {cmp_str}"
        )

    print()
    print(
        "RV_PACR reference (tt-llk, one 32x32 Float16 tile, loop_factor=32): "
        + ", ".join(f"w{w}={c}" for w, c in sorted(RV_PACR_CYC_PER_TILE.items()))
        + f" cyc/tile; normal HW untilize {TT_LLK_NORMAL_CYC_PER_TILE} cyc/tile."
    )
    print(
        "Stage 1 here is measured single-shot (loop_factor 1), so it carries pipeline fill and "
        "reads high against that 77.4; the compaction is the part this test is about."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
