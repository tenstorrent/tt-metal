# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the DATA array in agmm/agmm_db.html from the latest sweep results.

Pure post-processing: reads the shape spec (sweep_shapes.json), the measured
best-per-shape (sweep_latest.csv), non-OK statuses (sweep_history.csv), and the
skip-list (skip_shapes.txt), computes roofline bounds + utilization via
roofline_lib (100%-of-peak ceilings — one source of truth with run_sweeps), and
splices a fresh `var DATA=[...]` block into the dashboard HTML in place. Every
other shape row is marked measured (`m`) or projected (`fail` reason).

No heavy deps (no ttnn) — runs under a plain interpreter. run_sweeps calls this
automatically after each sweep; you can also run it by hand:

    python agmm/build_dashboard.py
"""

import csv
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from roofline_lib import compute_roofline  # noqa: E402

DEFAULT_SPEC = os.path.join(_THIS_DIR, "sweep_shapes.json")
DEFAULT_LATEST = os.path.join(_THIS_DIR, "sweep_latest.csv")
DEFAULT_HISTORY = os.path.join(_THIS_DIR, "sweep_history.csv")
DEFAULT_SKIP = os.path.join(_THIS_DIR, "skip_shapes.txt")
DEFAULT_HTML = os.path.join(_THIS_DIR, "agmm_db.html")
DEFAULT_ISO_AG = os.path.join(_THIS_DIR, "isolated_ag_latest.csv")
DEFAULT_ISO_MM = os.path.join(_THIS_DIR, "isolated_mm_latest.csv")

# Per device_config: ring size (cluster-axis length) and fabric links. Keep in
# sync with DEVICE_CONFIGS in models/tt_dit/utils/sweep_mm_block_sizes.py.
DEVICE_META = {
    "bh_4x8": {"ring": 4, "links": 2},
    "bh_4x8_ring8": {"ring": 8, "links": 2},
}

DATA_START = "  var DATA=["
DATA_END = "  ];"


def _fusion_label(fusion):
    fusion = fusion or {}
    if fusion.get("chunks", 1) > 1:
        return f"chunks{fusion['chunks']}"
    if fusion.get("use_addcmul"):
        return "addcmul"
    return "—"  # em dash


def _fail_reason(shape_id, latest_row, last_status, skip):
    """Human reason a shape has no measured best (shown as a projected row)."""
    if shape_id in skip:
        return "profiler-hang"
    if last_status == "NO_COMBOS":
        return "no-valid-blocking"
    if last_status and last_status != "OK":
        return last_status  # SWEEP_FAILED / NO_TIMINGS / ...
    return "not measured"


def _load_iso_us(path):
    """shape_id -> best_duration_us for OK rows of an isolated (AG or MM) sweep CSV."""
    out = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r.get("status") == "OK" and r.get("best_duration_us"):
                out[r["shape_id"]] = float(r["best_duration_us"])
    return out


def build_rows(
    spec_path=DEFAULT_SPEC,
    latest_path=DEFAULT_LATEST,
    history_path=DEFAULT_HISTORY,
    skip_path=DEFAULT_SKIP,
    iso_ag_path=DEFAULT_ISO_AG,
    iso_mm_path=DEFAULT_ISO_MM,
):
    """Return the list of dashboard row dicts (one per spec shape)."""
    spec = json.load(open(spec_path))
    latest = {}
    if os.path.exists(latest_path):
        latest = {r["shape_id"]: r for r in csv.DictReader(open(latest_path))}
    last_status = {}
    if os.path.exists(history_path):
        for r in csv.DictReader(open(history_path)):
            last_status[r["shape_id"]] = r["status"]  # later rows win -> most recent
    skip = set(open(skip_path).read().split()) if os.path.exists(skip_path) else set()
    # Isolated all-gather is deduped by (device_config, M, K); isolated matmul is
    # keyed by the AGMM shape id + "_mm". Both are measured on the healthy 2-link
    # galaxy; see build_comparison.py for the fused-vs-serial join.
    iso_ag = _load_iso_us(iso_ag_path)
    iso_mm = _load_iso_us(iso_mm_path)

    rows = []
    for s in spec:
        cfg = s["device_config"]
        if cfg not in DEVICE_META:
            raise ValueError(f"Unknown device_config '{cfg}' for shape {s['id']}; add it to DEVICE_META.")
        ring = DEVICE_META[cfg]["ring"]
        links = DEVICE_META[cfg]["links"]
        grid = tuple(s["grid"])
        fid = s.get("math_fidelity", "HiFi2")

        base = compute_roofline(s["M"], s["K"], s["N"], ring_size=ring, num_links=links, grid=grid, math_fidelity=fid)
        row = {
            "id": s["id"],
            "M": s["M"],
            "K": s["K"],
            "N": s["N"],
            "fusion": _fusion_label(s.get("fusion")),
            "tag": (s.get("tags") or [""])[0],
            "ring": ring,
            "t_compute": round(base["t_compute_us"], 1),
            "t_dram": round(base["t_dram_us"], 1),
            "t_fabric": round(base["t_fabric_us"], 1),
            "ideal": round(base["ideal_us"], 1),
            "limiter": base["limiter"],
            "ag_us": None,
            "mm_us": None,
            "m": None,
            "fail": None,
        }

        # Isolated timings: AG deduped by (device_config, M, K), MM by id + "_mm".
        ag = iso_ag.get(f"ag_{cfg}_m{s['M']}_k{s['K']}")
        mm = iso_mm.get(s["id"] + "_mm")
        if ag is not None:
            row["ag_us"] = round(ag, 1)
        if mm is not None:
            row["mm_us"] = round(mm, 1)

        lr = latest.get(s["id"])
        if lr and lr.get("status") == "OK" and lr.get("best_duration_us"):
            t = float(lr["best_duration_us"])
            rl = compute_roofline(
                s["M"], s["K"], s["N"], ring_size=ring, num_links=links, grid=grid, math_fidelity=fid, time_us=t
            )
            row["m"] = {
                "measured": round(t, 1),
                "speedup": round(rl["speedup"], 2),
                "blocking": f"M{lr['best_M_block']} K{lr['best_K_block']} N{lr['best_N_block']} sb({lr['best_sb_h']},{lr['best_sb_w']})",
                "flop": round(rl["flop_util"] * 100, 1),
                "dram": round(rl["dram_util"] * 100, 1),
                "fabric": round(rl["fabric_util"] * 100, 1),
                "tflops": round(rl["tflops_ach"], 1),
                "dram_gbps": round(rl["dram_gbps"], 1),
                "fabric_gbps": round(rl["fabric_gbps_per_link"], 1),
            }
        else:
            row["fail"] = _fail_reason(s["id"], lr, last_status.get(s["id"]), skip)
        rows.append(row)
    return rows


def _rows_to_js(rows):
    """Serialize rows to the `var DATA=[...]` JS block the dashboard expects."""
    out = [DATA_START]
    for d in rows:
        m = d["m"]
        if m is None:
            mjs = "null"
        else:
            mjs = (
                "{measured:%s,speedup:%s,blocking:%s,flop:%s,dram:%s,fabric:%s,"
                "tflops:%s,dram_gbps:%s,fabric_gbps:%s}"
                % (
                    m["measured"],
                    m["speedup"],
                    json.dumps(m["blocking"]),
                    m["flop"],
                    m["dram"],
                    m["fabric"],
                    m["tflops"],
                    m["dram_gbps"],
                    m["fabric_gbps"],
                )
            )
        fail = "null" if d["fail"] is None else json.dumps(d["fail"])
        ag = "null" if d["ag_us"] is None else d["ag_us"]
        mm = "null" if d["mm_us"] is None else d["mm_us"]
        out.append(
            "   {id:%s,M:%d,K:%d,N:%d,fusion:%s,tag:%s,ring:%d,t_compute:%s,t_dram:%s,"
            "t_fabric:%s,ideal:%s,limiter:%s,ag:%s,mm:%s,m:%s,fail:%s},"
            % (
                json.dumps(d["id"]),
                d["M"],
                d["K"],
                d["N"],
                json.dumps(d["fusion"]),
                json.dumps(d["tag"]),
                d["ring"],
                d["t_compute"],
                d["t_dram"],
                d["t_fabric"],
                d["ideal"],
                json.dumps(d["limiter"]),
                ag,
                mm,
                mjs,
                fail,
            )
        )
    out.append(DATA_END)
    return "\n".join(out)


def build(html_path=DEFAULT_HTML, **kwargs):
    """Splice a fresh DATA array (from the latest results) into the dashboard HTML."""
    rows = build_rows(**kwargs)
    html = open(html_path).read()
    start = html.find(DATA_START)
    if start == -1:
        raise ValueError(f"{html_path}: could not find '{DATA_START}' marker")
    end = html.find(DATA_END, start)
    if end == -1:
        raise ValueError(f"{html_path}: could not find closing '{DATA_END}' after DATA array")
    end += len(DATA_END)
    html = html[:start] + _rows_to_js(rows) + html[end:]
    with open(html_path, "w") as f:
        f.write(html)

    measured = sum(1 for r in rows if r["m"])
    from collections import Counter

    lim = Counter(r["limiter"] for r in rows)
    return {"total": len(rows), "measured": measured, "limiter": dict(lim), "html": html_path}


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--html", default=DEFAULT_HTML, help=f"Dashboard HTML to update in place (default: {DEFAULT_HTML})")
    p.add_argument("--spec", default=DEFAULT_SPEC)
    p.add_argument("--latest", default=DEFAULT_LATEST)
    p.add_argument("--history", default=DEFAULT_HISTORY)
    p.add_argument("--skip", default=DEFAULT_SKIP)
    args = p.parse_args()
    res = build(
        html_path=args.html,
        spec_path=args.spec,
        latest_path=args.latest,
        history_path=args.history,
        skip_path=args.skip,
    )
    print(f"Dashboard updated: {res['html']}  ({res['measured']}/{res['total']} measured; limiter {res['limiter']})")


if __name__ == "__main__":
    main()
