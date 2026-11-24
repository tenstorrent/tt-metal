# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Stage times and per-layer costs from run_stage_probe.sh runs, written as one JSON artifact.

    python stage_probe_summary.py --out <json>
        --split <name> <stage>=<run_dir>[@<first>[-<last>]] [...]    (repeatable; a "/" in <name> nests it)
        [--reference <split>] [--provisional <text>]
        [--ci <split> <probe>=<tok_s> [...]] [--entry-overhead <split>/<stage> <split>/<stage>]
        [--note <key>=<text>] [...]

A stage is layers [first, last] of a run (default: all of the run's layers). For a GLM-5.4 run (command.txt
probe_model=glm54) [first, last] are layer PASS positions of the 106-pass schedule (pass_schedule.py), and a chunk's
acks are put in pass order from their ack ids (a loop pass-2 layer L acks as L + 78), so the whole trunk's stages are
G0 0-25, G1 26-53, G2 54-81, G3 82-105. A run's MTP rows (ids 78.. past the model's layers, PREFILL_MTP_LEVELS on the
last rank; decode levels included) follow its last pass as positions 106.. (e.g. G3 with MTP = 82-109 at 4 rows); a
stage given without a last position runs to the run's last ack, MTP rows included. Every ack time is the producer's
clock (time.perf_counter in one process) from the CHUNK_RETIRE lines; a chunk retires on its last ack. Per probe
chunk d (the CI's probe chunks 0, 9, 19, 50, 101, 203 = 5k, 50k, 100k, 255k, 510k, 1020k tokens prefilled through
that chunk) a stage reports:
  ms         ack(last) - ack(first) of chunk d: the stage's first to last layer ack (CI's chunk_time definition)
  period_ms  median over chunks d-4..d+4 of ack(last) minus the event before ack(first): ack(first - 1) of the
             same chunk, or the previous chunk's last ack when the stage starts at the run's first layer (chunks
             are queued back to back). That is the stage's full span of num_layers layers, last ack to last ack:
             its retire period when it runs alone, and the same span when it is cut out of a longer run.
composed: tok/s = 5120 / the slowest stage's period_ms (and / its ms) per probe.

Per layer type: the layer ack is sent after layer i's attention half (indexer, KV write, sparse attention,
output projection), so ack(i+1) - ack(i) = FFN(i) + attention(i+1). The least-squares fit
delta(i) = P[ffn(i)] + Q[attn(i+1)] over a stage's layers and a chunk window gives each layer type's cost
P[ffn] + Q[attn], which is the same for any split of a layer around its ack. A GLM-5.4 run adds a pass-2 attention
type (two-bank, no indexer) and leaves out the deltas that cross a damp gate (the end of each loop pass); those are
reported per crossing as gate_crossing_ms instead.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np

CHUNK = 5120
PROBES = {"5k": 0, "50k": 9, "100k": 19, "255k": 50, "510k": 101, "1020k": 203}
WINDOWS = {"short (chunks 0-2, <=15k)": range(0, 3), "~1M (chunks 195-203, ~1.0M)": range(195, 204)}
_RETIRE = re.compile(
    r"CHUNK_RETIRE slot=(\d+) start=(\d+) end=(\d+) first=([\d.]+) retire=([\d.]+) warmup=([01]) acks=(\d+) layers=(\S*)"
)
_LAYER = re.compile(r"(\d+)-(\d+)@([\d.]+)")
# GLM-5.3 (config.json): layers 0-2 dense, the rest MoE; indexer_types "full" layers run the indexer, the others
# reuse the last full layer's top-k.
FULL = {0, 1, 2} | set(range(6, 78, 4))


def pass_order(cmd_text: str) -> tuple:
    """(model, the run's (layer, loop_pass) per position, ack id per position) from its command.txt. GLM-5.3 runs are
    not looped: a position is its layer, and acks are taken in ascending id (ack ids None)."""
    model = re.search(r"probe_model=(\w+)", cmd_text)
    model = model.group(1) if model else "glm53"
    first, count = (int(x) for x in cmd_text.split()[2:4])
    if model == "glm54":
        from models.demos.deepseek_v3_d_p.reference.glm_5_4.schedule import pass_schedule
        from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import glm_5_4_hf_config

        schedule = pass_schedule(glm_5_4_hf_config())
        return model, schedule.passes[first : first + count], schedule.ack_ids(first, count)
    return model, tuple((layer, 0) for layer in range(first, first + count)), None


def read_run(run_dir: Path) -> dict:
    cmd_text = (run_dir / "command.txt").read_text()
    model, passes, ack_ids = pass_order(cmd_text)
    chunks = []
    for line in (run_dir / "producer.log").read_text(errors="replace").splitlines():
        m = _RETIRE.search(line)
        if not m or m.group(6) != "0":
            continue
        acks = {int(a): float(t) for a, _b, t in _LAYER.findall(m.group(8))}
        if ack_ids is None:  # GLM-5.3: ascending id (older stage-alone runs ack under rank-local ids 0..)
            order = sorted(acks)
        else:
            # MTP rows: the ids right past the model's last layer (only a stage ending on it runs MTP), in id order
            # after the last pass.
            extra = sorted(set(acks) - set(ack_ids))
            mtp_base = max(layer for layer, _ in passes) + 1
            assert sorted(acks) == sorted(ack_ids + extra) and extra == list(
                range(mtp_base, mtp_base + len(extra))
            ), f"{run_dir}: chunk {m.group(2)} acks {sorted(acks)} != {ack_ids} + MTP rows {mtp_base}.."
            order = ack_ids + extra
        # Absolute ms on the producer clock: first is the chunk's first ack, the layers are ms after it. In pass order.
        first_ms = float(m.group(4)) * 1000.0
        chunks.append({"start": int(m.group(2)), "acks": [first_ms + acks[i] for i in order]})
    chunks.sort(key=lambda c: c["start"])
    assert chunks and all(c["start"] == i * CHUNK for i, c in enumerate(chunks)), f"{run_dir}: chunk gaps"
    power = re.search(r"^power: (.*)$", cmd_text, re.M)
    return {
        "dir": run_dir,
        "model": model,
        "passes": passes,
        "first_layer": int(cmd_text.split()[2]),
        "chunks": chunks,
        "sha": re.search(r"sha=(\w+)", cmd_text).group(1),
        "start": re.search(r"start=(\S+)", cmd_text).group(1),
        "power": power.group(1) if power else "not recorded (run predates the launcher's tt-smi read)",
        "command": cmd_text.splitlines()[0],
    }


def stage_probes(run: dict, first: int, last: int) -> dict:
    ch = run["chunks"]
    a, b = first - run["first_layer"], last - run["first_layer"]
    assert 0 <= a <= b < len(ch[0]["acks"]), f"layers [{first}, {last}] outside {run['dir'].name}"

    def span(i):
        before = ch[i]["acks"][a - 1] if a else ch[i - 1]["acks"][-1]
        return ch[i]["acks"][b] - before

    out = {}
    for label, d in PROBES.items():
        if d >= len(ch):
            continue
        lo, hi = max(1, d - 4), min(len(ch) - 1, d + 4)
        out[label] = {
            "ms": round(ch[d]["acks"][b] - ch[d]["acks"][a], 2),
            "period_ms": round(statistics.median(span(i) for i in range(lo, hi + 1)), 2),
            "chunk": d,
            "source": f"{run['dir'].name} layers {first}-{last} (tt-metal {run['sha'][:11]}, bh-glx-120-b06u02)",
        }
    return out


def composed(stages: dict) -> dict:
    out = {}
    for label in PROBES:
        vals = {s: v[label] for s, v in stages.items() if label in v}
        if len(vals) != len(stages):
            continue
        slow_p = max(vals, key=lambda s: vals[s]["period_ms"])
        slow_ms = max(vals, key=lambda s: vals[s]["ms"])
        out[label] = {
            "slowest_stage_by_period": slow_p,
            "slowest_period_ms": vals[slow_p]["period_ms"],
            "tok_s_from_period": round(CHUNK / (vals[slow_p]["period_ms"] / 1000.0)),
            "slowest_stage_by_ms": slow_ms,
            "tok_s_from_ms": round(CHUNK / (vals[slow_ms]["ms"] / 1000.0)),
        }
    return out


def fit_layer_types(run: dict, first: int, last: int, chunk_ids) -> dict:
    """Per layer type cost P[ffn] + Q[attn] (ms) from the ack deltas of positions [first, last] (MTP rows excluded)."""
    passes = run["passes"]
    last = min(last, run["first_layer"] + len(passes) - 1)
    rows, ys, gates = [], [], {}
    for c in (run["chunks"][i] for i in chunk_ids if i < len(run["chunks"])):
        for pos in range(first, last):
            i = pos - run["first_layer"]
            (layer, k), (nxt, k_next) = passes[i], passes[i + 1]
            delta = c["acks"][i + 1] - c["acks"][i]
            if k and k_next != k:  # loop pass k ends here: its damp gate runs between the two acks
                gates.setdefault(f"pass{k}_gate: L{layer}->L{nxt}", []).append(delta)
                continue
            row = [0.0] * 5  # P_dense, P_moe, Q_indexer, Q_reuse, Q_pass2
            row[0 if layer < 3 else 1] = 1.0
            row[4 if k_next >= 2 else (2 if nxt in FULL else 3)] = 1.0
            rows.append(row)
            ys.append(delta)
    if not rows:  # the run has no chunk in this window
        return {"n_deltas": 0}
    a, y = np.array(rows), np.array(ys)
    x, *_ = np.linalg.lstsq(a, y, rcond=None)
    held = passes[first - run["first_layer"] : last - run["first_layer"] + 1]
    cost = {
        "dense+indexer": (x[0] + x[2], any(l < 3 for l, _ in held)),
        "moe+indexer": (x[1] + x[2], any(l >= 3 and l in FULL and k < 2 for l, k in held)),
        "moe+reuse": (x[1] + x[3], any(l >= 3 and l not in FULL and k < 2 for l, k in held)),
        "moe+pass2": (x[1] + x[4], any(k >= 2 for _, k in held)),
    }
    out = {t: round(float(v), 3) for t, (v, present) in cost.items() if present}
    out["fit_rms_ms"] = round(float(np.sqrt(np.mean((a @ x - y) ** 2))), 3)
    out["n_deltas"] = len(ys)
    if gates:
        out["gate_crossing_ms"] = {g: round(statistics.median(v), 3) for g, v in gates.items()}
    return out


def nested(root: dict, path: str) -> dict:
    node = root
    for part in path.split("/"):
        node = node.setdefault(part, {})
    return node


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", action="append", nargs="+", required=True, metavar="NAME_OR_STAGE")
    ap.add_argument("--reference", help="split whose slowest-stage period_ms is the reference")
    ap.add_argument("--provisional", help="why the reference is provisional (omit when it is final)")
    ap.add_argument("--ci", nargs="+", metavar="SPLIT_OR_PROBE", help="<split> <probe>=<tok_s> ...")
    ap.add_argument("--entry-overhead", nargs=2, metavar=("ALONE", "SLICE"))
    ap.add_argument("--note", action="append", default=[], metavar="KEY=TEXT")
    args = ap.parse_args()

    result, fits, runs, stage_index = {}, [], {}, {}
    for spec in args.split:
        name, stage_specs = spec[0], spec[1:]
        stages = {}
        for item in stage_specs:
            stage, run_spec = item.split("=", 1)
            run_dir, _, layer_spec = run_spec.partition("@")
            run = read_run(Path(run_dir))
            runs[run["dir"].name] = {k: run[k] for k in ("sha", "start", "power", "command")}
            lo, _, hi = layer_spec.partition("-")
            first = int(lo) if lo else run["first_layer"]
            last = int(hi) if hi else run["first_layer"] + len(run["chunks"][0]["acks"]) - 1
            stages[stage] = stage_probes(run, first, last)
            fits.append((f"{name}/{stage}", run, first, last))
            stage_index[f"{name}/{stage}"] = stages[stage]
        node = nested(result, name)
        node.update(stages)
        node["layers"] = {s: f"{fits[-len(stages) + k][2]}-{fits[-len(stages) + k][3]}" for k, s in enumerate(stages)}
        node["composed"] = composed(stages)

    if args.reference:
        comp = nested(result, args.reference)["composed"]
        result["reference"] = {
            "split": args.reference,
            "metric": "slowest stage period_ms per probe (last ack to last ack, median over chunks d-4..d+4)",
            "slowest_stage_period_ms": {p: c["slowest_period_ms"] for p, c in comp.items()},
            "slowest_stage": {p: c["slowest_stage_by_period"] for p, c in comp.items()},
            "tok_s": {p: c["tok_s_from_period"] for p, c in comp.items()},
            "compare_with": "a GLM-5.4 stage's period_ms on the same definition (per-stage last-ack-to-last-ack "
            "period, median over chunks d-4..d+4, same probe chunks); met if GLM-5.4 slowest <= 1.54x this",
            "status": f"PROVISIONAL: {args.provisional}" if args.provisional else "final",
        }
    if args.ci:
        comp = nested(result, args.ci[0])["composed"]
        ci = {k: int(v) for k, v in (x.split("=") for x in args.ci[1:])}
        result["ci_comparison"] = {
            p: {
                "ci_tok_s": t,
                "composed_tok_s_from_period": comp[p]["tok_s_from_period"],
                "ratio_to_ci": round(comp[p]["tok_s_from_period"] / t, 3),
            }
            for p, t in ci.items()
            if p in comp
        }
    if args.entry_overhead:
        alone, sliced = (stage_index[k] for k in args.entry_overhead)
        result["entry_overhead_ms"] = {
            "definition": f"period_ms of {args.entry_overhead[0]} minus {args.entry_overhead[1]}: same layers and "
            "inputs; a non-first stage cut out of a longer run does not include it, a stage run alone does "
            "(cause not attributed)",
            **{
                p: round(alone[p]["period_ms"] - sliced[p]["period_ms"], 2)
                for p in PROBES
                if p in alone and p in sliced
            },
        }
    result["per_layer_type_ms"] = {
        w: {key: fit_layer_types(run, lo, hi, ids) for key, run, lo, hi in fits} for w, ids in WINDOWS.items()
    }
    result["per_layer_note"] = (
        "Per layer type cost = P[ffn] + Q[attn] from a least-squares fit of ack(i+1)-ack(i) = P[ffn(i)] + "
        "Q[attn(i+1)] over each stage's layers: the ack follows layer i's attention half (indexer, KV write, sparse "
        "attention), so a delta is FFN(i) + attention(i+1). Producer clock, traced, one user; acks are quantized at "
        "~1.07 ms, so the fit averages many deltas."
    )
    for note in args.note:
        key, text = note.split("=", 1)
        result[key] = text
    result["runs"] = runs
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("reference", "ci_comparison") if k in result}, indent=1))


if __name__ == "__main__":
    main()
