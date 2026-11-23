# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Stage times and per-layer costs from run_stage_probe.sh runs.

    python stage_probe_summary.py --out <json> --split <split_name> <stage>=<run_dir>[@<from_layer>] [...] [--split ...]

@<from_layer> measures a stage that starts at a later layer than its run (a run that starts one full-indexer layer
early, so the stage's leading reuse layers get real top-k): ms spans from that layer's ack, and period_ms
subtracts the earlier layers' ack span.

Per probe chunk (the CI's probe chunks: 0, 9, 19, 50, 101, 203 = 5k, 50k, 100k, 255k, 510k, 1020k tokens
prefilled through that chunk), a stage reports:
  ms         the chunk's first layer ack -> last layer ack on the rank (CI's chunk_time definition)
  period_ms  median retire-to-retire interval over the chunks [d-4, d+4]: the stage's retire period with the
             next chunk already queued, which also covers the parts before the first ack and after the last
  layer_ms   per layer, ack(i+1) - ack(i), attributed to layer i: layer i's work after its KV write plus
             layer i+1's work before its own (norm and projections); a stage's last layer has none.
All times are the producer's clock (time.perf_counter in one process), from the CHUNK_RETIRE lines.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

CHUNK = 5120
PROBES = {"5k": 0, "50k": 9, "100k": 19, "255k": 50, "510k": 101, "1020k": 203}
_RETIRE = re.compile(
    r"CHUNK_RETIRE slot=(\d+) start=(\d+) end=(\d+) first=([\d.]+) retire=([\d.]+) warmup=([01]) acks=(\d+) layers=(\S*)"
)
_LAYER = re.compile(r"(\d+)-(\d+)@([\d.]+)")
# GLM-5.3 (config.json): layers 0-2 dense, the rest MoE; indexer_types "full" layers run the indexer, the others
# reuse the last full layer's top-k.
FULL = {0, 1, 2} | set(range(6, 78, 4))


def layer_type(layer: int) -> str:
    ffn = "dense" if layer < 3 else "moe"
    attn = "indexer" if layer in FULL else "reuse"
    return f"{ffn}+{attn}"


def read_run(run_dir: Path) -> dict:
    cmd = (run_dir / "command.txt").read_text().split()
    first_layer = int(cmd[2])
    chunks = []
    for line in (run_dir / "producer.log").read_text(errors="replace").splitlines():
        m = _RETIRE.search(line)
        if not m or m.group(6) != "0":
            continue
        acks = [(int(a), float(t)) for a, _b, t in _LAYER.findall(m.group(8))]
        chunks.append(
            {
                "start": int(m.group(2)),
                "first": float(m.group(4)),
                "retire": float(m.group(5)),
                "acks": [t for _, t in sorted(acks)],
            }
        )
    chunks.sort(key=lambda c: c["start"])
    assert chunks and all(c["start"] == i * CHUNK for i, c in enumerate(chunks)), f"{run_dir}: chunk gaps"
    sha = re.search(r"sha=(\w+)", (run_dir / "command.txt").read_text()).group(1)
    return {"dir": run_dir, "first_layer": first_layer, "chunks": chunks, "sha": sha}


def stage_probes(run: dict, from_layer: int) -> dict:
    ch = run["chunks"]
    skip = from_layer - run["first_layer"]
    assert skip >= 0, f"from layer {from_layer} precedes the run's first layer {run['first_layer']}"
    # ms from the chunk's first ack to the stage's first ack (0 without a skip).
    lead = [c["acks"][skip] for c in ch]
    retire = [c["retire"] for c in ch]
    out = {}
    for label, d in PROBES.items():
        if d >= len(ch):
            continue
        lo, hi = max(1, d - 4), min(len(ch) - 1, d + 4)
        periods = [(retire[i] - retire[i - 1]) * 1000.0 - lead[i] for i in range(lo, hi + 1)]
        out[label] = {
            "ms": round((ch[d]["retire"] - ch[d]["first"]) * 1000.0 - lead[d], 2),
            "period_ms": round(statistics.median(periods), 2),
            "chunk": d,
            "source": f"run_stage_probe.sh {run['dir'].name} (tt-metal {run['sha'][:11]}), bh-glx-120-b06u02, "
            f"chunk {d}" + (f", from layer {from_layer}'s ack" if skip else ""),
        }
    return out


def layer_costs(run: dict, chunk_ids) -> dict:
    """Mean per-layer ack delta (ms) over chunk_ids, keyed by global layer index."""
    out = {}
    for c in (run["chunks"][i] for i in chunk_ids if i < len(run["chunks"])):
        acks = c["acks"]
        for i in range(len(acks) - 1):
            out.setdefault(run["first_layer"] + i, []).append(acks[i + 1] - acks[i])
    return {layer: statistics.mean(v) for layer, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", action="append", nargs="+", metavar="NAME_OR_STAGE", required=True)
    args = ap.parse_args()

    result, per_layer_runs = {}, []
    for spec in args.split:
        name, stages = spec[0], spec[1:]
        result[name] = {}
        for item in stages:
            stage, run_spec = item.split("=", 1)
            run_dir, _, from_layer = run_spec.partition("@")
            run = read_run(Path(run_dir))
            result[name][stage] = stage_probes(run, int(from_layer) if from_layer else run["first_layer"])
            per_layer_runs.append((name, stage, run))
        # Composed pipeline rate: one chunk per slowest-stage period.
        comp = {}
        for label in PROBES:
            vals = {s: v[label] for s, v in result[name].items() if label in v}
            if len(vals) != len(stages):
                continue
            slow_ms = max(vals, key=lambda s: vals[s]["ms"])
            slow_p = max(vals, key=lambda s: vals[s]["period_ms"])
            comp[label] = {
                "slowest_stage_by_ms": slow_ms,
                "tok_s_from_ms": round(CHUNK / (vals[slow_ms]["ms"] / 1000.0)),
                "slowest_stage_by_period": slow_p,
                "tok_s_from_period": round(CHUNK / (vals[slow_p]["period_ms"] / 1000.0)),
            }
        result[name]["composed"] = comp

    windows = {"short (chunks 0-2, <=15k)": range(0, 3), "~1M (chunks 195-203)": range(195, 204)}
    per_type = {}
    for wname, ids in windows.items():
        by_type = {}
        for name, stage, run in per_layer_runs:
            for layer, ms in layer_costs(run, ids).items():
                by_type.setdefault(layer_type(layer), {}).setdefault(f"{name}/{stage}", []).append(ms)
        per_type[wname] = {
            t: {
                k: {
                    "mean_ms": round(statistics.mean(v), 3),
                    "min_ms": round(min(v), 3),
                    "max_ms": round(max(v), 3),
                    "n": len(v),
                }
                for k, v in runs.items()
            }
            for t, runs in sorted(by_type.items())
        }
    result["per_layer_type"] = per_type
    result["per_layer_note"] = (
        "ack(i+1)-ack(i) attributed to layer i (its work after its KV write plus the next layer's work before "
        "its KV write); a stage's last layer is excluded; producer clock, traced, single user."
    )
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_layer_type"}, indent=1))
    print(json.dumps(per_type, indent=1))


if __name__ == "__main__":
    main()
