# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Markdown comparison table from one or more result JSONL files (schema in README.md).

  python merge_results.py results_tt.jsonl results_gpu.jsonl [--latest]

Rows are grouped by input (or sequence length); speedup is against the `fa` row of the same platform, device, input
and rows per device (`extra.rows_per_device`, e.g. fa run at the VSA rung), else the `fa` row at the default length.
--latest keeps only the newest record per (platform, device, backend, config, dtype, input).
"""

import argparse
import json
import sys

REQUIRED = (
    "platform",
    "device",
    "backend",
    "config",
    "seconds",
    "tokens",
    "heads",
    "head_dim",
    "dtype",
    "ms_median",
    "ms_p10",
    "ms_p90",
    "kept_fraction",
    "rel_l2_vs_dense",
    "pcc_vs_dense",
    "scale_vs_dense",
    "commit",
    "timestamp",
)


def load(paths):
    recs = []
    for path in paths:
        with open(path) as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                missing = [k for k in REQUIRED if k not in r]
                if missing:
                    print(f"{path}:{n}: missing {missing}, skipped", file=sys.stderr)
                    continue
                recs.append(r)
    return recs


def group_key(r):
    return r.get("input") or f"{r['tokens']} tokens"


def _cfg(c):
    return ", ".join(f"{k}={v}" for k, v in sorted((c or {}).items())) or "-"


def _rows(r):
    return (r.get("extra") or {}).get("rows_per_device")


def _length(r):
    padded = (r.get("extra") or {}).get("padded_tokens") or r["tokens"]
    rows = _rows(r)
    return f"{padded:,}" + (f" ({rows:,})" if rows else "")


def _f(x, fmt):
    return "-" if x is None else format(x, fmt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--latest", action="store_true")
    a = p.parse_args()
    recs = load(a.files)
    if a.latest:
        newest = {}
        for r in recs:
            k = (r["platform"], r["device"], r["backend"], json.dumps(r["config"], sort_keys=True), r["dtype"])
            k += (group_key(r),)
            if k not in newest or r["timestamp"] > newest[k]["timestamp"]:
                newest[k] = r
        recs = list(newest.values())
    fa = [r for r in recs if r["backend"] == "fa"]
    dense = {(r["platform"], r["device"], group_key(r), _rows(r)): r["ms_median"] for r in fa}
    plain = [r for r in fa if "rung" not in (r["config"] or {})]
    default = {(r["platform"], r["device"], group_key(r)): r["ms_median"] for r in plain}
    groups = {}
    for r in recs:
        groups.setdefault(group_key(r), []).append(r)
    for g, rows in sorted(groups.items(), key=lambda kv: (kv[1][0]["tokens"], kv[0])):
        r0 = rows[0]
        secs = f", {r0['seconds']} s" if r0["seconds"] is not None else ""
        print(f"### {g} ({r0['tokens']} tokens, {r0['heads']} heads x {r0['head_dim']}{secs})\n")
        print(
            "| platform | device | backend | config | dtype | tokens (rows/device) | ms median (p10-p90) | vs fa "
            "| kept | rel L2 | PCC | scale |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: (r["platform"], r["device"], r["backend"] != "fa", r["ms_median"])):
            base = dense.get((r["platform"], r["device"], g, _rows(r))) if _rows(r) else None
            base = base or default.get((r["platform"], r["device"], g))
            speed = f"{base / r['ms_median']:.2f}x" if base else "-"
            print(
                f"| {r['platform']} | {r['device']} | {r['backend']} | {_cfg(r['config'])} | {r['dtype']} | "
                f"{_length(r)} | "
                f"{r['ms_median']:.3f} ({r['ms_p10']:.3f}-{r['ms_p90']:.3f}) | {speed} | "
                f"{_f(r['kept_fraction'], '.3f')} | {_f(r['rel_l2_vs_dense'], '.4f')} | "
                f"{_f(r['pcc_vs_dense'], '.5f')} | {_f(r['scale_vs_dense'], '.4f')} |"
            )
        print()


if __name__ == "__main__":
    main()
