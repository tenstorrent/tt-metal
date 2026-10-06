#!/usr/bin/env python3
"""Machine-checkable definition of 'done' for a model bring-up.

    scorecard.py template [--model NAME] > bringup/<model>/scorecard.json
    scorecard.py run --card bringup/<model>/scorecard.json --log bringup/<model>/LOG.md [--repo .] [--json]

DESIGN (coordinator decision): the CPU reference does NOT run the whole model. Validation stages are per-layer teacher-forced PCC (CPU oracle fed
the captured real hidden state), a short chained-prefix PCC curve (first 4-8 layers or one of each layer type), a prefill-phase check (layer outputs +
KV), a decode-phase check (one step from a reference-prefilled state), then end-to-end acceptance WITHOUT a CPU full-model run: GSM8K and determinism
(both REQUIRED: the only catch for compounding error beyond the CPU prefix), first-token sanity, optional hosted-oracle comparison. Gates are data:
override thresholds per model in the scorecard file, do not edit this code.

A scorecard is a JSON file (YAML too if PyYAML is importable; a .yaml file may also contain JSON): {"model": ..., "gates": [gate, ...]}. A gate:
    {"id": "pcc.layer.moe", "kind": "single_layer_pcc", "required": true, "evidence": ["E261006-1843-a3f"], "threshold": 0.99, ...kind params}
`evidence` lists LOG.md entry ids; the numbers are read from each entry's `data` JSON (log_tool.py add --data '{...}').
Status per gate:  PASS | FAIL (evidence fresh+verified but the number misses the threshold) | MISSING (no evidence id, id not in the log,
evidence is inferred, or its data lacks the needed keys) | STALE (evidence entry is superseded, or its commit/--paths no longer match HEAD: see log_tool.py check).
Exit code 0 only if every required gate is PASS. Optional gates (required=false) are reported but do not block.

Gate kinds and the `data` keys they read (kind params in parentheses):
  single_layer_pcc    pcc                                   (threshold, layer_type)
  chain_pcc_curve     pcc_curve: [pcc after layer 1, 2, ..]  (min_final, max_drop_per_layer)
  first_token         agree, n                              (min_frac)
  gsm8k               n, correct                            (min_n, min_correct)
  determinism         runs, bit_identical                   (min_runs)
  memory_fit          table: [{batch, isl, free_gib}, ...]   (cells: [[batch, isl], ...], min_free_gib)
  perf_recorded       the listed keys are present           (keys: e.g. ["ttft_ms","prefill_tok_s"] or ["decode_ms_tok","tok_s_user"])
  spec_exact          n, identical                          (min_frac)
  prefill_check       layer_pcc_min, kv_pcc_min              (min_layer_pcc, min_kv_pcc)  prefill phase: layer outputs + KV contents vs CPU oracle
  decode_check        pcc                                   (threshold)  one decode step from a reference-prefilled state
  hosted_oracle       n, prompts_file                       (min_n)  optional behavioural comparison vs a hosted model; evidence = logged prompts/outputs file
  metric              any key                               (key, op in >= <= == >  <, value)
"""
import argparse
import json
import operator
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import log_tool  # noqa: E402

OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq, ">": operator.gt, "<": operator.lt}


class Missing(Exception):
    pass


def need(d, *keys):
    for k in keys:
        if k not in d:
            raise Missing(f"evidence data lacks '{k}'")
    return [d[k] for k in keys]


# each checker: (gate, data) -> (ok, detail); raises Missing
def c_single_layer_pcc(g, d):
    (v,) = need(d, "pcc")
    return v >= g["threshold"], f"pcc {v} vs >= {g['threshold']}"


def c_chain(g, d):
    (curve,) = need(d, "pcc_curve")
    drops = [curve[i] - curve[i + 1] for i in range(len(curve) - 1)]
    md = max(drops) if drops else 0.0
    ok = curve[-1] >= g["min_final"] and md <= g["max_drop_per_layer"]
    return ok, f"final {curve[-1]} (min {g['min_final']}), max drop/layer {md:.4f} (max {g['max_drop_per_layer']})"


def c_first_token(g, d):
    a, n = need(d, "agree", "n")
    return n > 0 and a / n >= g["min_frac"], f"{a}/{n} agree (min {g['min_frac']})"


def c_gsm8k(g, d):
    n, c = need(d, "n", "correct")
    return (
        n >= g["min_n"] and c >= g["min_correct"],
        f"{c}/{n} correct (need n>={g['min_n']}, correct>={g['min_correct']})",
    )


def c_determinism(g, d):
    r, b = need(d, "runs", "bit_identical")
    return r >= g.get("min_runs", 2) and bool(b), f"{r} runs, bit_identical={b}"


def c_memory(g, d):
    (table,) = need(d, "table")
    have = {(r["batch"], r["isl"]): r["free_gib"] for r in table}
    miss = [c for c in g["cells"] if tuple(c) not in have]
    if miss:
        raise Missing(f"memory table lacks cells {miss}")
    low = {c: have[tuple(c)] for c in map(tuple, g["cells"]) if have[c] < g["min_free_gib"]}
    return (
        not low,
        f"{len(g['cells'])} cells, min free {min(have[tuple(c)] for c in g['cells'])} GiB (need >= {g['min_free_gib']})"
        + (f"; below: {low}" if low else ""),
    )


def c_perf(g, d):
    vals = need(d, *g["keys"])
    return True, ", ".join(f"{k}={v}" for k, v in zip(g["keys"], vals))


def c_spec(g, d):
    n, i = need(d, "n", "identical")
    return n > 0 and i / n >= g["min_frac"], f"{i}/{n} identical to plain (min {g['min_frac']})"


def c_prefill(g, d):
    a, b = need(d, "layer_pcc_min", "kv_pcc_min")
    return (
        a >= g["min_layer_pcc"] and b >= g["min_kv_pcc"],
        f"layer pcc min {a} (>= {g['min_layer_pcc']}), kv pcc min {b} (>= {g['min_kv_pcc']})",
    )


def c_decode(g, d):
    (v,) = need(d, "pcc")
    return v >= g["threshold"], f"decode-step pcc {v} vs >= {g['threshold']}"


def c_hosted(g, d):
    n, f = need(d, "n", "prompts_file")
    return (
        n >= g.get("min_n", 1) and os.path.exists(f),
        f"{n} prompts compared, file {f} {'exists' if os.path.exists(f) else 'MISSING'}",
    )


def c_metric(g, d):
    (v,) = need(d, g["key"])
    return OPS[g["op"]](v, g["value"]), f"{g['key']}={v} {g['op']} {g['value']}"


CHECKERS = {
    "single_layer_pcc": c_single_layer_pcc,
    "chain_pcc_curve": c_chain,
    "first_token": c_first_token,
    "gsm8k": c_gsm8k,
    "determinism": c_determinism,
    "prefill_check": c_prefill,
    "decode_check": c_decode,
    "hosted_oracle": c_hosted,
    "memory_fit": c_memory,
    "perf_recorded": c_perf,
    "spec_exact": c_spec,
    "metric": c_metric,
}


def evaluate_gate(g, entries, state_fn):
    """-> (status, detail). state_fn(entry) -> ('FRESH'|'STALE'|'SUPERSEDED', reasons)."""
    ids = g.get("evidence") or []
    if isinstance(ids, str):
        ids = [ids]
    if not ids:
        return "MISSING", "no evidence id"
    by_id = {e["id"]: e for e in entries}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        return "MISSING", f"evidence id(s) not in LOG: {unknown}"
    data = {}
    for i in ids:
        e = by_id[i]
        s, why = state_fn(e)
        if s != "FRESH":
            return "STALE", f"{i}: {s} ({'; '.join(why)})"
        if e.get("status") != "verified":
            return "MISSING", f"{i} is inferred, not verified: re-measure"
        data.update(e["data_obj"])
    if g["kind"] not in CHECKERS:
        return "MISSING", f"unknown gate kind {g['kind']}"
    try:
        ok, detail = CHECKERS[g["kind"]](g, data)
    except Missing as m:
        return "MISSING", str(m)
    except KeyError as k:
        return "MISSING", f"gate parameter {k} missing in scorecard"
    return ("PASS" if ok else "FAIL"), detail


def run(card, entries, state_fn):
    rows = []
    for g in card["gates"]:
        s, d = evaluate_gate(g, entries, state_fn)
        rows.append({"id": g["id"], "kind": g["kind"], "required": g.get("required", True), "status": s, "detail": d})
    ok = all(r["status"] == "PASS" for r in rows if r["required"])
    return ok, rows


def load_card(path):
    txt = open(path).read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # optional dependency
        except ImportError:
            return json.loads(txt)  # JSON is valid YAML: ship the card as JSON if PyYAML is missing
        return yaml.safe_load(txt)
    return json.loads(txt)


def template(model, layer_types=("dense", "moe")):
    R = {"required": True, "evidence": []}
    g = [
        {
            "id": f"layer_pcc.{t}",
            "kind": "single_layer_pcc",
            "threshold": 0.99,
            "layer_type": t,
            **R,
            "note": "teacher-forced: CPU oracle fed the captured REAL hidden state of the device run, one layer at a time",
        }
        for t in layer_types
    ]
    g += [
        {
            "id": "chain_prefix_pcc",
            "kind": "chain_pcc_curve",
            "min_final": 0.97,
            "max_drop_per_layer": 0.01,
            **R,
            "note": "short chained prefix: first 4-8 layers or one of each layer type (NOT the full model)",
        },
        {
            "id": "prefill_phase",
            "kind": "prefill_check",
            "min_layer_pcc": 0.99,
            "min_kv_pcc": 0.99,
            **R,
            "note": "layer outputs + KV cache contents after prefill vs CPU oracle",
        },
        {
            "id": "decode_phase",
            "kind": "decode_check",
            "threshold": 0.98,
            **R,
            "note": "one decode step starting from a reference-prefilled state",
        },
        {
            "id": "first_token",
            "kind": "first_token",
            "min_frac": 0.9,
            **R,
            "note": "sanity: argmax of the first generated token vs reference on real prompts",
        },
        {
            "id": "gsm8k",
            "kind": "gsm8k",
            "min_n": 64,
            "min_correct": 55,
            **R,
            "note": "REQUIRED. closed-loop greedy decode, full model; catches compounding error beyond the CPU prefix. set n/min_correct per model",
        },
        {
            "id": "determinism",
            "kind": "determinism",
            "min_runs": 2,
            **R,
            "note": "REQUIRED. two identical runs, bit-identical outputs",
        },
        {
            "id": "memory_fit",
            "kind": "memory_fit",
            "min_free_gib": 1.0,
            "cells": [[4, 4096], [16, 4096], [64, 4096]],
            **R,
            "note": "free DRAM per chip after prefill",
        },
        {"id": "perf_prefill", "kind": "perf_recorded", "keys": ["ttft_ms", "prefill_tok_s"], **R},
        {"id": "perf_decode", "kind": "perf_recorded", "keys": ["decode_ms_tok", "tok_s_user"], **R},
        {
            "id": "hosted_oracle",
            "kind": "hosted_oracle",
            "min_n": 16,
            "required": False,
            "evidence": [],
            "note": "optional behavioural comparison vs a hosted model; evidence data.prompts_file = logged prompts/outputs",
        },
        {
            "id": "spec_exact",
            "kind": "spec_exact",
            "min_frac": 0.95,
            "required": False,
            "evidence": [],
            "note": "only if speculative decoding is shipped",
        },
    ]
    return {"model": model, "version": 2, "gates": g}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("template")
    s.add_argument("--model", default="NEW_MODEL")
    s.add_argument(
        "--layer-types",
        default="dense,moe",
        help="comma list, one single-layer gate each (e.g. dense,moe,swa,full_attn)",
    )
    s.add_argument("--yaml", action="store_true", help="print YAML (needs PyYAML)")
    s = sub.add_parser("run")
    s.add_argument("--card", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--repo", default=".")
    s.add_argument("--ref", default="HEAD")
    s.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "template":
        c = template(a.model, tuple(a.layer_types.split(",")))
        if a.yaml:
            import yaml

            print(yaml.safe_dump(c, sort_keys=False))
        else:
            print(json.dumps(c, indent=1))
        return 0
    entries = log_tool.load(a.log)
    ok, rows = run(load_card(a.card), entries, lambda e: log_tool.entry_state(e, a.repo, entries, a.ref))
    if a.json:
        print(json.dumps({"done": ok, "gates": rows}, indent=1))
    else:
        for r in rows:
            print(f"{r['status']:<8} {'*' if r['required'] else ' '} {r['id']:<26} {r['detail']}")
        n = sum(r["status"] == "PASS" for r in rows)
        print(f"\n{n}/{len(rows)} gates PASS (* = required). DONE: {'YES' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
