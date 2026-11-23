# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Stage times and per-layer costs from run_stage_probe.sh runs.

    python stage_probe_summary.py --out <json> --split <split_name> <stage>=<run_dir>[@<from>[-<to>]] [...] [--split ...]

@<from> measures a stage that starts at a later layer than its run (a run that starts one full-indexer layer
early, so the stage's leading reuse layers get real top-k): ms spans from that layer's ack, and period_ms
subtracts the earlier layers' ack span. @<from>-<to> slices layers [from, to] out of a longer run (e.g. all 78
layers on one rank): ms = ack(to) - ack(from); period_ms is the slice span (previous layer ack, or previous chunk
retire, -> ack(to)), median over chunks d-4..d+4.

Per probe chunk (the CI's probe chunks: 0, 9, 19, 50, 101, 203 = 5k, 50k, 100k, 255k, 510k, 1020k tokens
prefilled through that chunk), a stage reports:
  ms         the chunk's first layer ack -> last layer ack on the rank (CI's chunk_time definition)
  period_ms  median retire-to-retire interval over the chunks [d-4, d+4]: the stage's retire period with the
             next chunk already queued, which also covers the parts before the first ack and after the last

Per layer type: the layer-completion ack fires after a layer's KV write, inside its attention half, and a full
layer's indexer runs before that write. So ack(i+1) - ack(i) = post(i) + pre(i+1): layer i's work after its KV
write (rest of attention, FFN) plus layer i+1's work up to its own KV write (norm, projections, the indexer on a
full layer). The fit delta(i) = P[ffn(i)] + Q[attn(i+1)] (least squares over a stage's layers and a chunk window)
gives each layer type's cost P[ffn] + Q[attn], which does not depend on how a layer splits around its ack.
All times are the producer's clock (time.perf_counter in one process), from the CHUNK_RETIRE lines.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np

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


def stage_probes(run: dict, from_layer: int, to_layer: int, sliced: bool) -> dict:
    ch = run["chunks"]
    skip = from_layer - run["first_layer"]
    last = to_layer - run["first_layer"]
    n_acks = len(ch[0]["acks"])
    assert 0 <= skip <= last < n_acks, f"layers [{from_layer}, {to_layer}] outside the run's {n_acks} acks"
    out = {}
    for label, d in PROBES.items():
        if d >= len(ch):
            continue
        acks = ch[d]["acks"]
        rec = {"ms": round(acks[last] - acks[skip], 2) if sliced else None}
        if not sliced:
            # lead: ms from the chunk's first ack to the stage's first ack (0 without a skip).
            rec["ms"] = round((ch[d]["retire"] - ch[d]["first"]) * 1000.0 - acks[skip], 2)
            lo, hi = max(1, d - 4), min(len(ch) - 1, d + 4)
            periods = [
                (ch[i]["retire"] - ch[i - 1]["retire"]) * 1000.0 - ch[i]["acks"][skip] for i in range(lo, hi + 1)
            ]
            rec["period_ms"] = round(statistics.median(periods), 2)
        else:
            # A slice's pace: from the event before the stage's first ack (the previous layer's ack, or the
            # previous chunk's retire when the slice starts at the run's first layer; chunks are queued back to
            # back) to the stage's last ack, i.e. exactly num_layers layer spans.
            def span(i):
                a = ch[i]["acks"]
                prev = a[skip - 1] if skip else (ch[i - 1]["retire"] - ch[i]["first"]) * 1000.0
                return a[last] - prev

            lo, hi = max(1, d - 4), min(len(ch) - 1, d + 4)
            rec["period_ms"] = round(statistics.median(span(i) for i in range(lo, hi + 1)), 2)
            rec[
                "period_kind"
            ] = "slice span: previous layer ack (or previous chunk retire) -> last ack, median d-4..d+4"
        rec["chunk"] = d
        what = (
            f", ack({to_layer}) - ack({from_layer}) of a {n_acks}-layer run"
            if sliced
            else (f", from layer {from_layer}'s ack" if skip else "")
        )
        rec[
            "source"
        ] = f"run_stage_probe.sh {run['dir'].name} (tt-metal {run['sha'][:11]}), bh-glx-120-b06u02, chunk {d}{what}"
        out[label] = rec
    return out


def fit_layer_types(run: dict, from_layer: int, to_layer: int, chunk_ids) -> dict:
    """Per layer type cost P[ffn] + Q[attn] (ms) from the ack deltas of layers [from_layer, to_layer]."""
    cols = ["P_dense", "P_moe", "Q_indexer", "Q_reuse"]
    rows, ys = [], []
    for c in (run["chunks"][i] for i in chunk_ids if i < len(run["chunks"])):
        acks = c["acks"]
        for layer in range(from_layer, to_layer):
            i = layer - run["first_layer"]
            row = [0.0] * 4
            row[0 if layer < 3 else 1] = 1.0
            row[2 if layer + 1 in FULL else 3] = 1.0
            rows.append(row)
            ys.append(acks[i + 1] - acks[i])
    a, y = np.array(rows), np.array(ys)
    x, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = float(np.sqrt(np.mean((a @ x - y) ** 2)))
    present = {
        "dense+indexer": any(l < 3 for l in range(from_layer, to_layer + 1)),
        "moe+indexer": any(l >= 3 and l in FULL for l in range(from_layer, to_layer)),
        "moe+reuse": any(l >= 3 and l not in FULL for l in range(from_layer, to_layer)),
    }
    cost = {
        "dense+indexer": x[0] + x[2],
        "moe+indexer": x[1] + x[2],
        "moe+reuse": x[1] + x[3],
    }
    out = {t: round(float(v), 3) for t, v in cost.items() if present[t]}
    out["fit_rms_ms"] = round(resid, 3)
    out["n_deltas"] = len(ys)
    return out


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
            run_dir, _, layer_spec = run_spec.partition("@")
            run = read_run(Path(run_dir))
            run_last = run["first_layer"] + len(run["chunks"][0]["acks"]) - 1
            lo, _, hi = layer_spec.partition("-")
            from_layer = int(lo) if lo else run["first_layer"]
            to_layer = int(hi) if hi else run_last
            result[name][stage] = stage_probes(run, from_layer, to_layer, sliced=bool(hi))
            per_layer_runs.append((name, stage, run, from_layer, to_layer))
        # Composed pipeline rate: one chunk per slowest-stage period (and per slowest first->last ack span).
        comp = {}
        for label in PROBES:
            vals = {s: v[label] for s, v in result[name].items() if label in v}
            if len(vals) != len(stages):
                continue
            slow_ms = max(vals, key=lambda s: vals[s]["ms"])
            comp[label] = {
                "slowest_stage_by_ms": slow_ms,
                "tok_s_from_ms": round(CHUNK / (vals[slow_ms]["ms"] / 1000.0)),
            }
            if all(v["period_ms"] is not None for v in vals.values()):
                slow_p = max(vals, key=lambda s: vals[s]["period_ms"])
                comp[label]["slowest_stage_by_period"] = slow_p
                comp[label]["tok_s_from_period"] = round(CHUNK / (vals[slow_p]["period_ms"] / 1000.0))
        result[name]["composed"] = comp

    windows = {"short (chunks 0-2, <=15k)": range(0, 3), "~1M (chunks 195-203, ~1.0M)": range(195, 204)}
    result["per_layer_type_ms"] = {
        wname: {f"{name}/{stage}": fit_layer_types(run, lo, hi, ids) for name, stage, run, lo, hi in per_layer_runs}
        for wname, ids in windows.items()
    }
    result["per_layer_note"] = (
        "Per layer type cost = P[ffn] + Q[attn] from a least-squares fit of ack(i+1)-ack(i) = P[ffn(i)] + "
        "Q[attn(i+1)] over each stage's layers (the ack fires after a layer's KV write; a full layer's indexer "
        "runs before it). Producer clock, traced, one user; acks are quantized at ~1.07 ms, so means over many "
        "deltas."
    )
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
