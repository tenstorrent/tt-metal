# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Render a model's bring-up dashboard (the ERNIE dashboard's look and sections, from any model's records).

    python -m models.demos.common.bringup.dashboard.export --spec S [--style standard|teletext|both] [--out PATH]
    -> <bringup_dir>/dashboard/index.html (standard) and/or teletext.html (a 90s teletext service on a CRT),
       self-contained, data inlined. The spec's ``dashboard.styles`` picks the default; the /bringup skill asks.

Reads: tasks.yaml, state.json, results/*.json (metrics, plan_memory.json, block_graphs.json, <task>_profile.json),
components.yaml, plan.yaml (optional ``chips``, ``ccl_per_layer``, ``profile_sections``), findings.yaml, the HF config.
Sections: progress, gate ladder, model graph + layer strip, op coverage, findings, chunk timing, sharding (memory
from the plan gate), where the time goes (warm per-section, per-chip profile), PCC trail by layer.
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
    return f"{k:.0f}k" if abs(k - round(k)) < 0.05 else f"{k:.1f}k"


def timing_rows(spec, task: dict, m: dict) -> list[dict]:
    """Headline timing rows of one task: what range of tokens, how, on which model, how long."""
    v = lambda k, d=None: m[k]["value"] if k in m else d  # noqa: E731
    hyb = v("device_model_hybrid")
    model = "hybrid harness" if hyb == 1 else "all-device" if hyb == 0 else "model not recorded"
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
    findings = load_yaml(spec.bringup_dir / "findings.yaml").get("findings", [])

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
                "frozen": f"{len(fr.get('files', {}))} files, reference {fr.get('reference', 'n/a')}, stub {fr.get('stub', 'n/a')}"
                if fr
                else None,
                "agent": (
                    f"{len(runs)} runs ({', '.join(sorted({r['role'] for r in runs}))}), debugger {s.get('debugger_attempts', 0)}, "
                    f"last session {runs[-1].get('session_id')}"
                )
                if runs
                else None,
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
            }
        )

    trails = []
    r2 = M.load("R.2", res)
    pts = sorted((int(k.rsplit("L", 1)[1]), v["value"]) for k, v in r2.items() if re.match(r"pcc_hidden_L\d+$", k))
    if pts:
        trails.append({"task": "R.2", "label": "CPU reference vs HF", "points": pts, "device": False})
    timing = []
    for t in tasks:
        m = M.load(t["id"], res)
        pts = sorted((int(k.rsplit("L", 1)[1]), v["value"]) for k, v in m.items() if re.match(r"pcc_layer_L\d+$", k))
        if pts:
            trails.append(
                {"task": t["id"], "label": f"{t['id']} {t['title'].split(':')[0]}", "points": pts, "device": True}
            )
        timing += timing_rows(spec, t, m)

    plan_mem = res / "plan_memory.json"
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "branch": git(spec.repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "head": git(spec.repo, "rev-parse", "--short", "HEAD"),
        "title": f"{spec.data.get('hf_id', spec.model).split('/')[-1]} chunked prefill",
        "page_title": f"{spec.get('display_name') or spec.model} Prefill Bring-up",
        "short_name": spec.get("display_name") or spec.model,
        "model": spec.data.get("hf_id", spec.model),
        "target": f"{spec.get('target.seq'):,} tokens in {spec.get('target.chunk'):,}-token chunks, {spec.get('target.dtype', 'bf16')}"
        if spec.get("target.seq")
        else "",
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
        "plan": json.loads(plan_mem.read_text()) if plan_mem.exists() else None,
        "plan_chips": plan_doc.get("chips") or [],
        "plan_ccl": plan_doc.get("ccl_per_layer") or [],
        "profile": load_profile(spec, res, plan_doc),
        "findings": findings,
        "thresholds": {k: spec.threshold(k, v) for k, v in DEFAULT_THRESHOLDS.items()},
        "source": str(spec.bringup_dir.relative_to(spec.repo)),
    }


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
    steps = []
    for key, ms in prof["sections_ms"].items():
        m = meta.get(key) or {}
        pc = prof["sections_ms_per_chip"].get(key, {})
        b = m.get("bound") or bound_of(key)
        steps.append(
            {
                "key": key,
                "name": m.get("name", key),
                "part": m.get("part", phase_of(key)),
                "bound": b,
                "pat": m.get("pat", "ring" if b == "comm" else "local"),
                "what": m.get("what", ""),
                "inp": m.get("inp", ""),
                "out": m.get("out", ""),
                "cells": m.get("cells") or [[f"chip {c}", ""] for c in range(nch)],
                "ms": round(ms, 2),
                "per_chip": [round(pc.get(str(c), 0.0), 2) for c in range(nch)],
                "programs": prof.get("programs", {}).get(key, 0),
                **({"before_ms": round(base["sections_ms"][key], 2)} if base and key in base["sections_ms"] else {}),
            }
        )
    steps.sort(key=lambda s: (s["part"], -s["ms"]))
    a, b = prof.get("chunk", [0, 0])
    return {
        "wall_ms": round(prof["wall_ms"], 1),
        "steps": steps,
        "source": prof_p.name,
        "heading": f"Where the time goes: one {a:,}→{b:,} chunk",
        "note": f"Device kernel time per section, summed over {len(prof.get('layers', []))} layers of one {b - a:,}-token chunk "
        f"at positions {a:,}-{b - 1:,} (golden prefix loaded, warm run). Per chip = that chip's own programs; the bar "
        f"uses the slowest chip per section. Source {prof_p.name}"
        + (f"; before = X.1 (chunk {base['wall_ms']:.0f} ms)." if base else "."),
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
