# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Render a model's bring-up dashboard (the ERNIE dashboard's look and sections, from any model's records).

    python -m models.demos.common.bringup.dashboard.export --spec S [--style standard|teletext|both] [--out PATH]
    -> <bringup_dir>/dashboard/index.html (standard) and/or teletext.html (a 90s teletext service on a CRT),
       self-contained, data inlined. The spec's ``dashboard.styles`` picks the default; the /bringup skill asks.

Reads: tasks.yaml, state.json, results/*.json (metrics, plan_memory.json, block_graphs.json, <task>_profile.json),
components.yaml, plan.yaml (optional ``chips``, ``ccl_per_layer``, ``profile_sections``), the HF config.
Sections: progress, gate ladder, model graph + layer strip, op coverage, chunk timing, sharding (memory
from the plan gate), where the time goes (warm per-section, per-chip profile), PCC trail by layer; with a spec
``prior``, a vs-prior view (accuracy, chunk time and TTFT, task status next to the prior bring-up's). Steps deferred to
op-gen (F46): a "Deferred to op-gen" section (task, block, layers, request status, evidence), a banner that the results
include N CPU steps, and timing rows and profile sections marked when the CPU bridge ran in them. Last, "Final tests":
the model smoke and the runner smoke (<task>_smoke.json, <task>_runner_smoke.json: prompt, answer, pass, time; the
runner's migration boundary and records), the full-target ladder rung and the contract tests (contract_tests.yaml and
the gates that ran them, <task>_contract.json per test where recorded); what never ran shows as "not run".
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import yaml

from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core.ledger import Ledger
from models.demos.common.bringup.core.spec import parse_layers
from models.demos.common.bringup.reference.golden import load_spec
from models.demos.common.bringup.testing.harness import DEFAULT_THRESHOLDS
from models.demos.common.bringup.testing.profiler import phase_of

HERE = Path(__file__).resolve().parent
COMPUTE = ("sdpa", "matmul", "linear", "proj", "qkv", "experts", "mlp", "shared", "ffn", "lm_head", "attention", "rope")
COMM = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all", "ccl")
MEMORY = ("kv_write", "dispatch", "combine", "fill_cache", "copy", "gather", "scatter", "reshape", "permute")


def bound_of(section: str) -> str:
    s = section.lower()
    for words, b in ((COMM, "comm"), (MEMORY, "memory"), (COMPUTE, "compute")):
        if any(w in s for w in words):
            return b
    return "other"


def git(repo: Path, *args) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout.strip()


def load_yaml(p: Path) -> dict:
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def hf_arch(spec) -> str:
    try:
        from models.demos.common.bringup.reference.golden import hf_path

        cfg = json.loads((Path(hf_path(spec)) / "config.json").read_text())
    except Exception:
        return ""
    c = cfg.get("text_config", cfg)
    parts = [f"{c.get('num_hidden_layers', '?')} layers", f"d={c.get('hidden_size', '?')}"]
    if c.get("num_attention_heads"):
        parts.append(f"GQA {c['num_attention_heads']}/{c.get('num_key_value_heads', '?')}×{c.get('head_dim', '?')}")
    ne = c.get("num_experts") or c.get("n_routed_experts") or c.get("moe_num_experts") or c.get("num_local_experts")
    if ne:
        k = c.get("top_k_experts") or c.get("num_experts_per_tok") or c.get("moe_k")
        parts.append(f"{ne} experts top-{k}")
    if c.get("sliding_window"):
        parts.append(f"sliding window {c['sliding_window']}")
    return " · ".join(parts)


def kt(n) -> str:
    """Tokens in 1024-token k: 56320 -> '55k', 51200 -> '50k'."""
    k = n / 1024
    if n < 1024:
        return str(int(n))
    return f"{k:.0f}k" if abs(k - round(k)) < 0.05 else f"{k:.1f}k"


def timing_rows(spec, task: dict, m: dict) -> list[dict]:
    """Headline timing rows of one task: what range of tokens, how, on which model, how long."""
    v = lambda k, d=None: m[k]["value"] if k in m else d  # noqa: E731
    hyb = v("device_model_hybrid")
    model = "hybrid harness" if hyb == 1 else "all-device" if hyb == 0 else "model not recorded"
    if v("deferred_cpu_steps"):  # F46: steps deferred to op-gen ran on the host inside these times
        model += f" + {v('deferred_cpu_steps')} CPU steps (deferred to op-gen, {v('deferred_cpu_ms', 0):.0f} ms host)"
    rows = []
    ch = sorted((int(k.rsplit("_c", 1)[1]), x["value"]) for k, x in m.items() if k.startswith("chunk_seconds_c"))
    if ch:
        rung = None
        if task["id"].startswith("L."):
            try:
                rung = spec.rung(task["id"][2:])
            except Exception:
                rung = None
        seq = v("rung_seq", rung and rung["seq"])
        chunk = v("rung_chunk", rung and rung["chunk"])
        start = v("rung_start", ((seq // chunk - 1) * chunk) if rung and rung.get("prefix_from_golden") else 0)
        secs = [x for _, x in ch]
        head = f"{kt(start)}->{kt(seq)}" if seq else task["id"]
        rows.append(
            {
                "task": task["id"],
                "headline": head,
                "chunks": ch,
                "tokens": (seq - start) if seq else None,
                "chunk": chunk,
                "how": "ladder: reads back every layer",
                "model": model,
                "seconds": sum(secs),
            }
        )
    full = sorted(
        (int(k.rsplit("_c", 1)[1]), x["value"] / 1e3) for k, x in m.items() if k.startswith("prefill_chunk_ms_c")
    )
    if "prefill_ms_full" in m:
        seq, chunk = v("prefill_seq"), v("prefill_chunk")
        rows.append(
            {
                "task": task["id"],
                "headline": f"0->{kt(seq)}" if seq else "full prefill",
                "chunks": full,
                "tokens": seq,
                "chunk": chunk,
                "how": "warm, no readback, one sync (TTFT without LM head)",
                "model": model,
                "seconds": v("prefill_ms_full") / 1e3,
            }
        )
    if "chunk_wall_ms" in m:
        start, n = v("chunk_start"), v("chunk_len")
        rows.append(
            {
                "task": task["id"],
                "headline": f"{kt(start)}->{kt(start + n)}" if n else "one chunk",
                "chunks": [[0, v("chunk_wall_ms") / 1e3]],
                "tokens": n,
                "chunk": n,
                "how": "warm, no readback (on the golden prefix)",
                "model": model,
                "seconds": v("chunk_wall_ms") / 1e3,
            }
        )
    return rows


def build(spec) -> dict:
    led = Ledger(spec.bringup_dir)
    tasks_spec, state = led.tasks(), led.state()
    res = led.results_dir
    comp_doc = load_yaml(spec.bringup_dir / "components.yaml")
    plan_doc = load_yaml(spec.bringup_dir / "plan.yaml")

    commits = {}
    for line in git(spec.repo, "log", "--format=%h|%s", f"--grep=[{spec.tag}][", "--fixed-strings").splitlines():
        sha, subj = line.split("|", 1)
        m = re.match(rf"\[{re.escape(spec.tag)}\]\[([^\]]+)\] ", subj)
        if m and m.group(1) not in commits:
            commits[m.group(1)] = sha

    tasks = []
    for tid in led.topo_order():
        t, s = tasks_spec[tid], state.get(tid, {})
        fr = t.get("frozen") or {}
        runs = s.get("agent_runs") or []
        tasks.append(
            {
                "id": tid,
                "title": t["title"],
                "step": t.get("step", "other"),
                "deps": t.get("deps", []),
                "cmd": t["gate"]["cmd"],
                "thresholds": t["gate"].get("metrics") or {},
                "status": s.get("status", "TODO"),
                "last_run": s.get("last_run"),
                "duration_s": s.get("duration_s"),
                "metrics": s.get("metrics", {}),
                "commit": commits.get(tid),
                "waiting": s.get("waiting") if s.get("status") != "PASS" else None,
                "reason": "; ".join(s.get("reason") or []) or None,
                "frozen": (
                    f"{len(fr.get('files', {}))} files, reference {fr.get('reference', 'n/a')}, stub {fr.get('stub', 'n/a')}"
                    if fr
                    else None
                ),
                "agent": (
                    (
                        f"{len(runs)} runs ({', '.join(sorted({r['role'] for r in runs}))}), debugger {s.get('debugger_attempts', 0)}, "
                        f"last session {runs[-1].get('session_id')}"
                    )
                    if runs
                    else None
                ),
            }
        )
    status = {t["id"]: t["status"] for t in tasks}

    block_types = list(spec.data.get("block_types", {}))
    layer_types = [spec.block_type_of(i) for i in range(spec.num_layers)]
    graphs_path = res / "block_graphs.json"
    graphs = json.loads(graphs_path.read_text()) if graphs_path.exists() else {}
    for bt, steps in graphs.items():
        for st in steps:
            st["task"] = f"C.{bt}.{st['name']}"
            st["state"] = "device" if status.get(st["task"]) == "PASS" else "cpu"
            st["deferred"] = status.get(st["task"]) == "DEFERRED"
    first_rung = spec.data["ladder"][0]["name"] if spec.data.get("ladder") else None
    comps = []
    for c in comp_doc.get("components") or []:
        key = f"{c['block_type']}/{c['step']}"
        gate = f"C.{c['block_type']}.{c['step']}" if c["block_type"] != "model" else f"L.{first_rung}"
        on = status.get(gate) == "PASS" and c.get("tag") != "CPU"
        comps.append(
            {
                "key": key,
                "ttnn": c.get("ttnn"),
                "tag": c.get("tag"),
                "reuse": c.get("reuse"),
                "task": gate,
                "state": "device" if on else "cpu",
                "deferred": status.get(gate) == "DEFERRED",
            }
        )

    trails = []
    r2 = M.load("R.2", res)
    pts = sorted((int(k.rsplit("L", 1)[1]), v["value"]) for k, v in r2.items() if re.match(r"pcc_hidden_L\d+$", k))
    if pts:
        trails.append({"task": "R.2", "label": "CPU reference vs HF", "points": pts, "device": False})
    timing, positions = [], None
    for t in tasks:
        m = M.load(t["id"], res)
        pts = sorted((int(k.rsplit("L", 1)[1]), v["value"]) for k, v in m.items() if re.match(r"pcc_layer_L\d+$", k))
        if pts:
            trails.append(
                {"task": t["id"], "label": f"{t['id']} {t['title'].split(':')[0]}", "points": pts, "device": True}
            )
        # performance only: the ladder is an accuracy run (reads every layer back) and never appears here
        timing += [r for r in timing_rows(spec, t, m) if not r["how"].startswith("ladder")]
        pos = sorted((int(k[len("pos_ms_") :]), x["value"]) for k, x in m.items() if re.match(r"pos_ms_\d+$", k))
        if pos:  # the latest task's sweep wins (tasks are in ledger order)
            hyb = m.get("device_model_hybrid", {}).get("value")
            model = "hybrid harness" if hyb == 1 else "all-device" if hyb == 0 else "model not recorded"
            if m.get("deferred_cpu_steps", {}).get("value"):
                model += f" + {m['deferred_cpu_steps']['value']} CPU steps (deferred to op-gen)"
            positions = {"task": t["id"], "chunk": m.get("pos_chunk", {}).get("value"), "points": pos, "model": model}

    plan_mem = res / "plan_memory.json"
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "branch": git(spec.repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "head": git(spec.repo, "rev-parse", "--short", "HEAD"),
        "title": f"{spec.data.get('hf_id', spec.model).split('/')[-1]} chunked prefill",
        "page_title": f"{spec.get('display_name') or spec.model} Prefill Bring-up",
        "short_name": spec.get("display_name") or spec.model,
        "model": spec.data.get("hf_id", spec.model),
        "target": (
            f"{spec.get('target.seq'):,} tokens in {spec.get('target.chunk'):,}-token chunks, {spec.get('target.dtype', 'bf16')}"
            if spec.get("target.seq")
            else ""
        ),
        "box": spec.get("box.name") or f"mesh {'x'.join(map(str, spec.mesh))}, {spec.mesh[0] * spec.mesh[1]} chips",
        "arch": hf_arch(spec),
        "mesh": spec.mesh,
        "chip_name": spec.get("box.chip", "chip"),
        "tasks": tasks,
        "block_types": block_types,
        "layer_types": layer_types,
        "selected_layers": spec.layers(),
        "subset": spec.layers() != list(range(spec.num_layers)),
        "graphs": graphs,
        "components": comps,
        "trails": trails,
        "timing": timing,
        "positions": positions,
        "plan": json.loads(plan_mem.read_text()) if plan_mem.exists() else None,
        "plan_chips": plan_doc.get("chips") or [],
        "plan_ccl": plan_doc.get("ccl_per_layer") or [],
        "profile": load_profile(spec, res, plan_doc),
        "prior": load_prior(spec),
        "deferred": deferred_rows(spec, led, state),
        "final": final_tests(spec, led, state),
        "thresholds": {k: spec.threshold(k, v) for k, v in DEFAULT_THRESHOLDS.items()},
        "source": str(spec.bringup_dir.relative_to(spec.repo)),
    }


def final_tests(spec, led, state) -> dict:
    """The "Final tests" section: the two end-to-end smokes (model smoke L.smoke, runner smoke in a contract gate),
    the full-target ladder rung and the contract tests, all from the ledger's recorded results. Missing = not run."""
    from models.demos.common.bringup.testing import serving as SV

    res, tasks = led.results_dir, led.tasks()
    order = led.topo_order()
    st = lambda tid: (state.get(tid) or {}).get("status", "TODO") if tid else None  # noqa: E731

    def covering(test: str) -> list[str]:
        """Tasks whose gate runs this contract test (by path, or through testing.serving --run <gate|all>)."""
        t = next((x for x in SV.tests(spec) if x["test"] == test), {})
        out = []
        for tid in order:
            cmd = tasks[tid]["gate"]["cmd"]
            runs = re.findall(r"testing\.serving --run (\S+)", cmd)
            if test.split("::")[0] in cmd or any(r in ("all", t.get("gates")) for r in runs):
                out.append(tid)
        return out

    # the smokes: the newest <task>_*smoke.json of each mode
    smokes = {}
    for p in res.glob("*_*smoke.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("mode") in ("model", "runner") and d.get("t", "") >= smokes.get(d["mode"], {}).get("t", ""):
            smokes[d["mode"]] = d
    rs = SV.runner_smoke(spec)
    want = {
        "model": next((t for t in order if t == "L.smoke"), None),
        # the gate that runs the runner smoke itself (K.1, or a model's own task), else any that runs it
        "runner": (
            (
                [t for t in covering(rs["test"]) if rs["test"].split("::")[0] in tasks[t]["gate"]["cmd"]]
                or covering(rs["test"])
            )
            or [None]
        )[-1]
        if rs
        else None,
    }
    rows = []
    for mode, label in (("model", "Model smoke"), ("runner", "Runner smoke")):
        d = smokes.get(mode)
        tid = (d or {}).get("task") or want[mode]
        sm = (spec.data.get("intake") or {}).get("smoke") or {}
        rows.append(
            {
                "mode": mode,
                "label": label,
                "task": tid,
                "status": st(tid),
                "ran": bool(d),
                "prompt": (d or {}).get("prompt", sm.get("prompt")),
                "expected": (d or {}).get("expected", sm.get("expect")),
                **{
                    k: d.get(k)
                    for k in ("answer", "ok", "seconds", "t", "prompt_len", "boundary", "records", "prompt_variant")
                    if d
                },
            }
        )

    # the full-target rung (ladder[-1], as X.3 runs it): the newest task that recorded that rung
    ladder = spec.data.get("ladder") or []
    lad = None
    if ladder:
        r = ladder[-1]
        start = (r["seq"] // r["chunk"] - 1) * r["chunk"] if r.get("prefix_from_golden") else 0
        best = None
        for tid in order:
            m = M.load(tid, res)
            v = lambda k: m[k]["value"] if k in m else None  # noqa: E731
            same = v("rung_seq") == r["seq"] and v("rung_chunk") == r["chunk"] and v("rung_start") in (start, None)
            pcc = [x["value"] for k, x in m.items() if re.match(r"pcc_layer_L\d+$", k)]
            if not pcc or not (same or tid == f"L.{r['name']}"):
                continue
            t = max(x.get("t", "") for x in m.values())
            if best is None or t >= best["t"]:
                best = {
                    "task": tid,
                    "t": t,
                    "status": st(tid),
                    "min_pcc": min(pcc),
                    "top1": v("top1_match"),
                    "top5": v("top5_overlap"),
                    "final_hidden": v("pcc_final_hidden"),
                }
        lad = {"rung": r["name"], "seq": r["seq"], "chunk": r["chunk"], "ran": bool(best), **(best or {})}

    # contract tests: per test, the newest task that ran it (a per-test record where the gate wrote one)
    cts = SV.tests(spec)
    con = None
    if cts:
        per = []
        for i, t in enumerate(cts):
            got = None
            for tid in covering(t["test"]):
                s = state.get(tid) or {}
                if s.get("status") in (None, "TODO", "RUNNING") or not s.get("last_run"):
                    continue
                side = res / f"{tid}_contract.json"
                rec = json.loads(side.read_text()).get("results", {}) if side.exists() else {}
                if t["test"] in rec:
                    ok = rec[t["test"]]
                elif t["test"].split("::")[0] in tasks[tid]["gate"]["cmd"]:
                    ok = s["status"] == "PASS"
                else:  # a --run gate without a per-test record: it ran the first contract_tests_run tests listed
                    n = (M.load(tid, res).get("contract_tests_run") or {}).get("value")
                    if n is None or i >= n:
                        continue
                    ok = s["status"] == "PASS"
                if got is None or s["last_run"] >= got["t"]:
                    got = {"t": s["last_run"], "ok": ok, "task": tid}
            per.append({"test": t["test"].rsplit("/", 1)[-1], **(got or {"ok": None})})
        con = {
            "total": len(per),
            "passed": sum(1 for p in per if p["ok"]),
            "failed": sum(1 for p in per if p["ok"] is False),
            "not_run": sum(1 for p in per if p["ok"] is None),
            "tests": per,
        }
    return {"smokes": rows, "ladder": lad, "contract": con}


def deferred_rows(spec, led, state) -> list[dict]:
    """One row per task deferred to op-gen, with its request's status and evidence (F46)."""
    from models.demos.common.bringup.plan import op_request as OR

    reqs = {tid: (d, req) for d, req in OR.all_requests(spec) for tid in req.get("tasks") or []}
    rows = []
    for tid in led.deferred():
        b = led.task(tid).get("brief") or {}
        d, req = reqs.get(tid, (None, {}))
        ev = req.get("evidence") or {}
        rows.append(
            {
                "task": tid,
                "block_type": b.get("block_type"),
                "step": b.get("step"),
                "layers": layer_label(req["layers"]) if req.get("layers") else "",
                "op": req.get("op") or (state.get(tid, {}).get("deferred") or {}).get("op"),
                "status": req.get("status", "missing"),
                "request": str(d.relative_to(spec.repo)) if d else None,
                "searched": [str(x) for x in ev.get("searched") or []],
                "tried": [f"{x.get('what')}: {x.get('outcome')}" for x in ev.get("tried") or [] if isinstance(x, dict)],
                "why_not_fork": ev.get("why_not_fork", ""),
            }
        )
    return rows


def layer_label(lays: list[int]) -> str:
    """[1, 2, 3, 4] -> 'L1-4', [5] -> 'L5', [0, 5, 11] -> 'L0,5,11'."""
    if len(lays) > 1 and lays == list(range(lays[0], lays[-1] + 1)):
        return f"L{lays[0]}\u2013{lays[-1]}"
    return "L" + ",".join(map(str, lays))


def load_profile(spec, res: Path, plan_doc: dict) -> dict | None:
    """The newest profile (a picked perf task's, else X.1) with X.1 as the before-comparison."""
    profs = sorted(res.glob("*_profile.json"), key=lambda p: p.stat().st_mtime)
    if not profs:
        return None
    prof_p = profs[-1]
    prof = json.loads(prof_p.read_text())
    base_p = res / "X.1_profile.json"
    base = json.loads(base_p.read_text()) if base_p.exists() and base_p != prof_p else None
    meta = plan_doc.get("profile_sections") or {}
    nch = spec.mesh[0] * spec.mesh[1]
    ops = prof.get("ops") or {}

    bridged = set(prof.get("bridged_steps") or [])  # F46: steps deferred to op-gen, on the CPU bridge

    def step(key, ms, pc, op_rows, programs):
        m = meta.get(key) or {}
        b = m.get("bound") or bound_of(key)
        cpu = key.split(".")[0] in bridged
        return {
            "key": key,
            "name": m.get("name", key) + (" (CPU bridge, deferred to op-gen)" if cpu else ""),
            "cpu_bridge": cpu,
            "part": m.get("part", phase_of(key)),
            "bound": b,
            "pat": m.get("pat", "ring" if b == "comm" else "local"),
            "what": m.get("what", ""),
            "inp": m.get("inp", ""),
            "out": m.get("out", ""),
            "cells": m.get("cells") or [[f"chip {c}", ""] for c in range(nch)],
            "ms": round(ms, 2),
            "per_chip": [round(pc.get(str(c), 0.0), 2) for c in range(nch)],
            "programs": programs,
            "ops": op_rows,
            **({"before_ms": round(base["sections_ms"][key], 2)} if base and key in base["sections_ms"] else {}),
        }

    def merged_ops(layers, key, n):
        """The section's ttnn ops over these layers, in the first layer's execution order, per layer (sum / n)."""
        rows = {}
        for li in layers:
            for r in ops.get(f"L{li}.{key}", []):
                k = (r["op"], r["shape"], r.get("mem", ""))
                acc = rows.setdefault(
                    k,
                    {
                        "op": r["op"],
                        "shape": r["shape"],
                        "mem": r.get("mem", ""),
                        "calls": 0,
                        "ms": 0.0,
                        "pc": {},
                        "tl": {},
                    },
                )
                acc["calls"] += r["calls"]
                acc["ms"] += r["ms"]
                for f in ("gap_ms", "host_ms", "slot_ms"):
                    if f in r:
                        acc["tl"][f] = acc["tl"].get(f, 0.0) + r[f]
                for c, v in r.get("ms_per_chip", {}).items():
                    acc["pc"][c] = acc["pc"].get(c, 0.0) + v
        return [
            {
                "op": r["op"],
                "shape": r["shape"],
                "mem": r["mem"],
                "calls": round(r["calls"] / n, 2),
                "ms": round(r["ms"] / n, 3),
                "per_chip": [round(r["pc"].get(str(c), 0.0) / n, 3) for c in range(nch)],
                **{f: round(v / n, 4) for f, v in r["tl"].items()},
            }
            for r in rows.values()
            if r["ms"] > 0 or r["calls"]
        ]

    layers_all = prof.get("layers", [])
    steps = [
        step(
            key,
            ms,
            prof["sections_ms_per_chip"].get(key, {}),
            merged_ops(layers_all, key, 1),
            prof.get("programs", {}).get(key, 0),
        )
        for key, ms in prof["sections_ms"].items()
    ]
    steps.sort(key=lambda s: (s["part"], -s["ms"]))
    views = [{"id": "all", "label": f"All {len(layers_all)} layers", "layers": layers_all, "n": 1, "steps": steps}]
    if ops:  # one tab per block type: that type's sections, per layer (mean over its profiled layers)
        for bt, info in (spec.data.get("block_types") or {}).items():
            lays = [i for i in parse_layers(info["layers"], spec.num_layers) if i in layers_all]
            if not lays:
                continue
            secs = []
            for key in prof["sections_ms"]:
                rows = [r for li in lays for r in ops.get(f"L{li}.{key}", [])]
                if not rows:
                    continue
                pc = {}
                for li in lays:
                    for r in ops.get(f"L{li}.{key}", []):
                        for c, v in r.get("ms_per_chip", {}).items():
                            pc[c] = pc.get(c, 0.0) + v / len(lays)
                ms = sum(r["ms"] for r in rows) / len(lays)
                st = step(
                    key, ms, pc, merged_ops(lays, key, len(lays)), round(sum(r["programs"] for r in rows) / len(lays))
                )
                st.pop("before_ms", None)
                secs.append(st)
            secs.sort(key=lambda s: (s["part"], -s["ms"]))
            views.append(
                {"id": bt, "label": f"{bt} · {layer_label(lays)}", "layers": lays, "n": len(lays), "steps": secs}
            )
    a, b = prof.get("chunk", [0, 0])
    return {
        "wall_ms": round(prof["wall_ms"], 1),
        "steps": steps,
        "views": views,
        "timeline": prof.get("timeline"),
        "chunk": [a, b],
        "source": prof_p.name,
        "heading": f"Where the time goes: one {a:,}→{b:,} chunk",
        "note": f"Device kernel time per section, summed over {len(prof.get('layers', []))} layers of one {b - a:,}-token chunk "
        f"at positions {a:,}-{b - 1:,} (golden prefix loaded, warm run). Per chip = that chip's own programs; the bar "
        f"uses the slowest chip per section. Source {prof_p.name}"
        + (f"; before = X.1 (chunk {base['wall_ms']:.0f} ms)." if base else ".")
        + (
            f" The wall time includes {prof.get('deferred_cpu_ms', 0):.0f} ms of host work for steps deferred to op-gen "
            f"({', '.join(sorted(bridged))}); their sections hold only the bridge's transfers."
            if bridged
            else ""
        ),
    }


ACC_KEY = re.compile(r"^(pcc_|text_top|top[15]_|final_hidden)")
PER_LAYER = re.compile(r"^pcc_(layer|state_\w+|hidden)_L\d+$")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _delta(a, b):
    """this - prior, or None when either side is missing."""
    return round(a - b, 9) if _num(a) and _num(b) else None


def _side(spec) -> dict:
    """What the vs-prior view needs from one bring-up: per-task result metrics, the current profile, the latest TTFT."""
    led = Ledger(spec.bringup_dir)
    res = led.results_dir
    tids = led.topo_order()
    metrics = {tid: {k: v["value"] for k, v in M.load(tid, res).items()} for tid in tids}
    ttft = None
    for tid in tids:  # the latest task's full prefill wins (ledger order), as for the position sweep
        for r in timing_rows(spec, {"id": tid}, M.load(tid, res)):
            if r["headline"].startswith("0->") and r["how"].startswith("warm"):
                ttft = r
    return {
        "led": led,
        "tids": tids,
        "metrics": metrics,
        "profile": load_profile(spec, res, load_yaml(spec.bringup_dir / "plan.yaml")),
        "ttft": ttft,
    }


def load_prior(spec) -> dict | None:
    """This run next to its prior bring-up (spec ``prior``): accuracy per matching task and layer, the current
    profile's chunk time per section and per block type, the 0->seq TTFT, and the runs.compare task rows."""
    if not spec.data.get("prior"):
        return None
    from models.demos.common.bringup.core.runs import compare

    ps = spec.prior_spec() if (spec.prior / "bringup" / "spec.yaml").exists() else None
    mesh = lambda s: "x".join(map(str, s.mesh))  # noqa: E731
    head = {"this": {"label": mesh(spec), "model": spec.model}, "source": spec.data["prior"]}
    if ps is None:
        return {**head, "prior": {"label": "?", "model": spec.data["prior"]}, "error": "no bringup/spec.yaml"}
    head["prior"] = {"label": mesh(ps), "model": ps.model}
    if head["prior"]["label"] == head["this"]["label"]:  # same mesh: tell them apart by model name
        head["this"]["label"], head["prior"]["label"] = spec.model, ps.model
    A, B = _side(spec), _side(ps)
    order = A["tids"] + [t for t in B["tids"] if t not in A["tids"]]

    layers, gate = [], []
    for tid in order:
        ma, mb = A["metrics"].get(tid, {}), B["metrics"].get(tid, {})
        lays = sorted(
            {int(k.rsplit("L", 1)[1]) for k in (*ma, *mb) if re.match(r"pcc_layer_L\d+$", k)}
        )  # per-layer PCC of the device runs (ladder, perf runs)
        for li in lays:
            k = f"pcc_layer_L{li:02d}"
            a, b = ma.get(k), mb.get(k)
            layers.append({"task": tid, "layer": li, "this": a, "prior": b, "delta": _delta(a, b)})
        for k in sorted(set(ma) | set(mb)):
            if ACC_KEY.match(k) and not PER_LAYER.match(k):
                a, b = ma.get(k), mb.get(k)
                if _num(a) or _num(b):
                    gate.append({"task": tid, "metric": k, "this": a, "prior": b, "delta": _delta(a, b)})

    def prof_sum(P):
        if not P:
            return {}
        out = {
            "device_ms": round(sum(s["ms"] for s in P["steps"]), 2),
            "wall_ms": P["wall_ms"],
            "source": P["source"],
            "sections": {s["key"]: s["ms"] for s in P["steps"]},
            "blocks": {v["id"]: round(sum(s["ms"] for s in v["steps"]), 2) for v in P["views"][1:]},
            "block_labels": {v["id"]: v["label"] for v in P["views"][1:]},
        }
        a, b = P.get("chunk") or [0, 0]
        out["chunk"] = f"{kt(a)}->{kt(b)}" if b else ""
        return out

    pa, pb = prof_sum(A["profile"]), prof_sum(B["profile"])
    chunk = pa.get("chunk") or pb.get("chunk") or "one chunk"
    if pa.get("chunk") and pb.get("chunk") and pa["chunk"] != pb["chunk"]:
        chunk = f"{pa['chunk']} vs {pb['chunk']}"

    def row(what, a, b, unit, src_a=None, src_b=None, short=None):
        r = {"what": what, "this": a, "prior": b, "delta": _delta(a, b), "unit": unit, "src": [src_a, src_b]}
        return {**r, "short": short} if short else r

    ta, tb = A["ttft"], B["ttft"]
    tt = lambda t: round(t["seconds"] * 1e3, 1) if t else None  # noqa: E731
    perf = [
        row(
            f"{chunk} chunk, device",
            pa.get("device_ms"),
            pb.get("device_ms"),
            "ms",
            pa.get("source"),
            pb.get("source"),
            "chunk device",
        ),
        row(
            f"{chunk} chunk, wall",
            pa.get("wall_ms"),
            pb.get("wall_ms"),
            "ms",
            pa.get("source"),
            pb.get("source"),
            "chunk wall",
        ),
        row(
            f"{(ta or tb or {}).get('headline', '0->seq')} TTFT (warm, no LM head)",
            tt(ta),
            tt(tb),
            "ms",
            ta and ta["task"],
            tb and tb["task"],
            f"TTFT {(ta or tb or {}).get('headline', '')}",
        ),
    ]
    sec_keys = list(pa.get("sections", {})) + [k for k in pb.get("sections", {}) if k not in pa.get("sections", {})]
    sections = [
        row(k, pa.get("sections", {}).get(k), pb.get("sections", {}).get(k), "ms")
        for k in sorted(
            sec_keys, key=lambda k: -max(pa.get("sections", {}).get(k, 0), pb.get("sections", {}).get(k, 0))
        )
    ]
    bts = list(spec.data.get("block_types") or {}) + [
        b for b in ps.data.get("block_types") or {} if b not in (spec.data.get("block_types") or {})
    ]
    blocks = [
        row(
            (pa.get("block_labels") or pb.get("block_labels") or {}).get(bt, bt),
            pa.get("blocks", {}).get(bt),
            pb.get("blocks", {}).get(bt),
            "ms/layer",
            short=bt,
        )
        for bt in bts
        if bt in pa.get("blocks", {}) or bt in pb.get("blocks", {})
    ]

    in_a, in_b = set(A["tids"]), set(B["tids"])
    tasks = []
    for r in compare(B["led"], A["led"]) + [
        {k: tuple(reversed(v)) if isinstance(v, tuple) else v for k, v in r.items()}
        for r in compare(A["led"], B["led"])
        if r["task"] not in in_a
    ]:
        tid = r["task"]
        pick = lambda v: [v[1] if tid in in_a else None, v[0] if tid in in_b else None]  # noqa: E731  (this, prior)
        tasks.append(
            {
                "task": tid,
                "status": pick(r["status"]),
                "attempts": pick(r["attempts"]),
                "debugger": pick(r["debugger"]),
                "duration_s": pick(r["duration_s"]),
                "agent_defs_changed": len(r["agent_defs_changed"]),
                "metric_deltas": len(r["metric_deltas"]),
            }
        )
    return {
        **head,
        "chunk": chunk,
        "layers": layers,
        "gate": gate,
        "perf": perf,
        "sections": sections,
        "blocks": blocks,
        "tasks": tasks,
    }


STYLES = {"standard": ("template.html", "index.html"), "teletext": ("teletext_template.html", "teletext.html")}


def render(data: dict, style: str) -> str:
    tpl = (HERE / STYLES[style][0]).read_text()
    title = data["page_title"] if style == "standard" else f"{data['short_name']} Ceefax"
    return tpl.replace("__TITLE__", title, 1).replace("/*__DATA__*/null", json.dumps(data), 1)


def styles_of(spec, requested: str | None) -> list[str]:
    """--style wins; else spec dashboard.styles (a list or one name); default: standard."""
    want = requested or spec.get("dashboard.styles") or ["standard"]
    want = [want] if isinstance(want, str) else list(want)
    want = list(STYLES) if want == ["both"] else want
    bad = [w for w in want if w not in STYLES]
    if bad:
        raise SystemExit(f"unknown dashboard style {bad}; choose from {sorted(STYLES)} or both")
    return want


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--style", help="standard | teletext | both (default: spec dashboard.styles, else standard)")
    ap.add_argument("--out", help="output file (one style only) or directory")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    data = build(spec)
    styles = styles_of(spec, a.style)
    outs = []
    for style in styles:
        html = render(data, style)
        if a.out and len(styles) == 1 and a.out.endswith(".html"):
            out = Path(a.out)
        else:
            out = (Path(a.out) if a.out else spec.bringup_dir / "dashboard") / STYLES[style][1]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        print(f"wrote {out} ({len(html) // 1024} KB, {style})")
        outs.append(out)
    return outs[0] if len(outs) == 1 else outs


if __name__ == "__main__":
    main()
