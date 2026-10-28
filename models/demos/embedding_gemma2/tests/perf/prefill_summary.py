# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build doc/prefill/<stage>/perf_summary.json from prefill.py records and the per-op tables.

    python -m models.demos.embedding_gemma2.tests.perf.prefill_summary \
        --timing run/prefill.jsonl --gate run_gate/prefill.jsonl --tables profiles/ \
        --stage 01-baseline --out doc/prefill/01-baseline/perf_summary.json

--timing: the records of the timed sweep (burst, host, sustained, and its own gate rows); --gate:
optional extra gate records (e.g. the run with TT_METAL_TRACE_ALLOC_TRACKING=1). The last record per
(shape, phase) wins within a file. --tables: a directory with b1_s<bucket>.json from prefill_op_table.py.
--superseded KEY=PATH embeds an earlier summary unchanged under KEY (E9.1-r2b: the 13 x 10 stage-1
summary under `superseded_13x10_p3_qkv_gu`, the 12 x 10 re-baseline at the top level).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _last(path: Path | None, run_id: str | None = None) -> dict:
    out = {}
    if not path:
        return out
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if run_id and r.get("run_id") != run_id:
            continue
        if "shape" in r:
            out[(r["shape"], r["phase"])] = r
    return out


def summarize(timing, gate, tables_dir, stage, commit, extra) -> dict:
    T = _last(timing)
    G = _last(gate)
    shapes = sorted({s for s, _ in T}, key=lambda s: int(s.split("_s")[1]))
    res, host = {}, {"round_trip_ms": {}, "h2d_ms": {}, "d2h_ms": {}, "d2h_last_hidden_ms": {}, "host_prep_ms": {}}
    clocks = []
    for s in shapes:
        b = T.get((s, "burst"))
        su = T.get((s, "sustained"))
        h = T.get((s, "host"))
        g = T.get((s, "gate")) or G.get((s, "gate"))
        g_alloc = G.get((s, "gate"))
        bucket = int(s.split("_s")[1])
        tbl = None
        if tables_dir and (Path(tables_dir) / f"{s}.json").exists():
            tbl = json.loads((Path(tables_dir) / f"{s}.json").read_text())
        e = {}
        if b:
            e["burst_ms"] = {
                "avg": b["avg_ms"],
                "best": b["best_ms"],
                "median": b["median_ms"],
                "n": b["n_per_window"] * b["windows"],
                "windows": b["windows"],
            }
        if su:
            tel = su.get("telemetry", {}).get("last_third", {})
            e["sustained_ms"] = {
                "median": su["median_last_third_ms"],
                "n": su["replays"],
                "windows": su["windows_last_third"],
                "window_replays": su["window_replays"],
                "duration_s": su["duration_s"],
                "first_window": su["first_window_ms"],
                "mean_all": su["mean_all_ms"],
            }
            if tel:
                e["sustained_telemetry"] = {k: tel[k]["median"] for k in tel}
                e["sustained_telemetry"]["aiclk_min_mhz"] = tel["aiclk_mhz"]["min"]
                e["sustained_telemetry"]["power_max_w"] = tel["power_w"]["max"]
                clocks.append(tel["aiclk_mhz"]["median"])
            if b:
                e["sustained_over_burst"] = su["median_last_third_ms"] / b["median_ms"]
        headline = (su or {}).get("median_last_third_ms") or (b or {}).get("median_ms")
        if headline:
            e["latency_ms_headline"] = headline
            e["tokens_per_s_per_chip"] = bucket / (headline / 1e3)
        if tbl:
            e["device_ms"] = tbl["device_kernel_us"] / 1e3
            e["floor_ms"] = tbl["floor_us"] / 1e3
            e["ops"] = tbl["ops"]
            e["eager_op_to_op_gap_ms"] = tbl["op_to_op_gap_us_eager"] / 1e3
        if g:
            e["pcc"] = g["last_hidden_pcc"]
            e["embedding_cosine"] = g["embedding_cosine"]
            e["replay_vs_eager_bitwise"] = g["replay_equals_eager_bitwise"]
            e["gate"] = g["status"]
            e["logical_len"] = g["logical_len"]
        if g_alloc:
            e["gate_with_trace_alloc_tracking"] = g_alloc["status"]
        if h:
            host["round_trip_ms"][s] = h["round_trip"]["mean_ms"]
            host["h2d_ms"][s] = h["h2d"]["mean_ms"]
            host["d2h_ms"][s] = h["d2h_embedding"]["mean_ms"]
            host["d2h_last_hidden_ms"][s] = h["d2h_last_hidden"]["mean_ms"]
            host["host_prep_ms"][s] = h["host_prep"]["mean_ms"]
            e["host_ms"] = {
                "host_prep": h["host_prep"]["mean_ms"],
                "h2d": h["h2d"]["mean_ms"],
                "d2h_embedding": h["d2h_embedding"]["mean_ms"],
                "d2h_last_hidden": h["d2h_last_hidden"]["mean_ms"],
                "round_trip": h["round_trip"]["mean_ms"],
                "replay_sync": h["replay_sync"]["mean_ms"],
                "h2d_bytes": h["h2d_bytes"],
                "d2h_last_hidden_bytes": h["d2h_last_hidden_bytes"],
            }
        if b and tbl:
            wall = b["median_ms"]
            dev = tbl["device_kernel_us"] / 1e3
            e["reconciliation_burst"] = {
                "wall_ms": wall,
                "device_kernel_ms": dev,
                "residual_ms": wall - dev,
                "residual_share": (wall - dev) / wall,
            }
        res[f"{s}_masked"] = e
    clocks.sort()
    return {
        "stage": stage,
        "commit": commit,
        "hardware": {**extra.get("hardware", {}), "clock_mhz_sustained": clocks[len(clocks) // 2] if clocks else None},
        "metric": "latency_ms per traced warmed forward (headline: sustained median)",
        "shapes": res,
        "host": host,
        "named_limitations": extra.get("named_limitations", []),
        "sources": {"timing": str(timing), "gate": str(gate) if gate else None, "tables": str(tables_dir)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", type=Path, required=True)
    ap.add_argument("--gate", type=Path)
    ap.add_argument("--tables", type=Path)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--commit", default="")
    ap.add_argument("--extra", type=Path, help="json with hardware / named_limitations (and any other top-level keys)")
    ap.add_argument(
        "--superseded",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="embed an earlier perf_summary.json unchanged under KEY (e.g. a superseded geometry); repeatable",
    )
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    extra = json.loads(a.extra.read_text()) if a.extra else {}
    s = summarize(a.timing, a.gate, a.tables, a.stage, a.commit, extra)
    for k, v in extra.items():  # e.g. compute_grid, precision_config, controls, roofline (added as-is)
        if k not in ("hardware", "named_limitations") and k not in s:
            s[k] = v
    for item in a.superseded:
        key, _, path = item.partition("=")
        if not key or not path:
            ap.error(f"--superseded wants KEY=PATH, got {item!r}")
        s[key] = json.loads(Path(path).read_text())
    a.out.write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps({k: v.get("latency_ms_headline") for k, v in s["shapes"].items()}, indent=1))


if __name__ == "__main__":
    main()
