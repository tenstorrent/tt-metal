# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hang-hunt driver for the GLM-5.3-Flash token-split experts: runs a queue of workload specs back to back through
scripts/run_safe_pytest.sh (one at a time, the device lock), one JSONL row per run, and stops at the first HANG so the
case can be narrowed by hand (triage saved, the hung op mapped to a layer / pass).

    python models/demos/glm53_flash_d_p_lb/tests/hang/campaign.py run --out /localdev/$USER/hang_campaign/<day> \\
        --queue <out>/queue.json --until 06:45
    python models/demos/glm53_flash_d_p_lb/tests/hang/campaign.py report --out <dir>

The queue (JSON list, re-read before every run, so it can be edited while the driver runs) holds entries
{"name": ..., "workload": "dvp" | "chunk" | "block" | "raw", "reps": n, "opts": {...}}; the first entry with fewer
than ``reps`` finished runs goes next. Workloads (the options each one takes: WORKLOADS below):
  dvp    decode via prefill (zz_scratch_babilong_think.py): eager, the host sampling between passes - what hung
  chunk  the whole model's chunk (test_layer_perf.py::test_chunk_perf), eager iterations then traced replays
  block  the token-split block's routed() alone (zz_scratch_experts_ts_stress.py), eager or traced
  raw    {"test": <pytest node>, "env": {...}}: anything else
An entry's "hook_args" (and optionally "hooks": another hook file) adds ops / host gaps / syncs / garbage around every
MoE call (std_hooks.py: matmul, sleep, sync, l1_garbage, dram_garbage, all_gather, py), e.g.
  {"name": "dvp_ts_mm_before", "workload": "dvp", "reps": 2, "opts": {"max_new": 400},
   "hook_args": {"layers": [20], "before": [{"op": "matmul", "m": 2048, "k": 4096, "n": 4096}]}}
Every run gets GLM_TS_OPLOG=1 (each MoE call's device op ids: the call count, and the hung op's layer / pass).
Run rows: runs.jsonl (spec, result, calls, wall, git state, hang location, triage summary); per run runs/<id>/
(log.txt, triage.txt, spec.json)."""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
TESTS = "models/demos/glm53_flash_d_p_lb/tests"
BASE_ENV = {
    "TT_METAL_HOME": str(REPO),
    "PYTHONPATH": f"{REPO}:{REPO}/ttnn:{REPO}/tools",
    "TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES": "0",
    "FLAT_CACHE_NO_ADOPT": "1",
    "GLM_TS_OPLOG": "1",
}


def _b(v):
    return "1" if v else "0"


def _dvp(o):
    env = {
        "BRINGUP_SPEC": "models/demos/glm53_flash_d_p_lb/bringup/spec_babilong.yaml",
        "GLM_EXPERTS_MODE": o.get("experts", "ts"),
        "GLM_INDEXER_RING": "0",
        "GLM_DVP_CHUNK": str(o.get("chunk", 2048)),
        "GLM_DVP_MAX_SEQ": str(o.get("max_seq", 8192)),
        "GLM_BL_TASKS": o.get("task", "qa1"),
        "GLM_BL_LENS": o.get("len", "4k"),
        "GLM_BL_IDX": ",".join(str(i) for i in o.get("idx", [1, 0, 2, 3, 4])),
        "GLM_BLT_MAX_NEW": str(o.get("max_new", 1536)),
    }
    return f"{TESTS}/zz_scratch_babilong_think.py", env, ["--run-all"]


def _chunk(o):
    env = {
        "BRINGUP_SPEC": "models/demos/glm53_flash_d_p_lb/bringup/spec.yaml",
        "BRINGUP_FABRIC_PAYLOAD": "8704",
        "BRINGUP_TRACE_REGION_BYTES": "400000000",
        "GLM_EXPERTS_MODE": o.get("experts", "ts"),
        "GLM_LP_REAL": _b(o.get("real", True)),
        "GLM_LP_CHUNK": str(o.get("chunk", 2048)),
        "GLM_LP_START": str(o.get("start", 2048)),
        "GLM_LP_ITERS": str(o.get("eager_iters", 3)),
        "GLM_CP_TRACE_ITERS": str(o.get("replays", 2000)),
    }
    return f"{TESTS}/test_layer_perf.py::test_chunk_perf", env, []


def _block(o):
    env = {
        "BRINGUP_SPEC": "models/demos/glm53_flash_d_p_lb/bringup/spec.yaml",
        "BRINGUP_FABRIC_PAYLOAD": "8704",
        "BRINGUP_TRACE_REGION_BYTES": "100000000",
        "GLM_TSS_CHUNK": str(o.get("chunk", 2048)),
        "GLM_TSS_ROUTINGS": ",".join(o.get("routings", ["uniform", "hot", "golden"])),
        "GLM_TSS_REPLAYS": str(o.get("replays", 20000)),
        "GLM_TSS_BATCH": str(o.get("batch", 500)),
        "GLM_TSS_EAGER": _b(o.get("eager", False)),
        "GLM_TSS_ROTATE": str(o.get("rotate", 0)),
        "GLM_TSS_REBUILD": str(o.get("rebuild", 0)),
        "GLM_TSG_LAYER": str(o.get("golden_layer", 4)),
    }
    return f"{TESTS}/zz_scratch_experts_ts_stress.py", env, []


def _raw(o):
    return o["test"], dict(o.get("env", {})), list(o.get("args", []))


WORKLOADS = {"dvp": _dvp, "chunk": _chunk, "block": _block, "raw": _raw}


def _git():
    def g(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout

    diff = g("diff", "HEAD")
    return {"sha": g("rev-parse", "--short=11", "HEAD").strip(), "dirty": hashlib.sha1(diff.encode()).hexdigest()[:10]}


def _rows(out):
    p = out / "runs.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


OP_RE = re.compile(r"\[(ts|ag)-op\] layer (\d+): ops (\d+)\.\.(\d+)")


def _calls(log):
    """MoE calls the run made: eager calls from the op log, plus traced replays x MoE layers in the trace."""
    ops = OP_RE.findall(log)
    eager = len(ops)
    layers = len({int(L) for _, L, _, _ in ops})
    rep = re.findall(r"\[cp\] traced replays (\d+) /", log)
    traced = int(rep[-1]) * layers if rep else 0
    tss = re.findall(r"TSS \w+: (\d+) / \d+ calls", log)  # (block stress: cumulative per routing)
    block = 0
    if tss:
        for name in dict.fromkeys(re.findall(r"TSS (\w+): \d+ / \d+ calls", log)):
            done = re.findall(rf"TSS {name}: (\d+) / \d+ calls", log)
            block += int(done[-1])
    st = re.findall(r"\[stress\] ([\d,]+) / [\d,]+ calls", log)  # (test_flat_stress: the op alone, mesh calls)
    op = int(st[-1].replace(",", "")) if st else 0
    return {
        "moe_eager": eager,
        "moe_traced": traced,
        "block_calls": block,
        "op_calls": op,
        "total": eager + traced + block + op,
    }


def _hang_site(log, triage):
    """The running op (triage) -> the MoE call holding it: layer, and which pass (that layer's n-th call)."""
    m = re.search(r"dump_running_operations\.py:\n[^\n]*\n(\d+),([A-Za-z]+)", triage)
    if not m:
        return {}
    op, name = int(m.group(1)), m.group(2)
    seen = Counter()
    for kind, L, a, b in OP_RE.findall(log):
        seen[L] += 1
        if int(a) <= op <= int(b):
            return {"op": op, "op_name": name, "layer": int(L), "layer_call": seen[L], "in": f"{kind} {a}..{b}"}
    for L, when, a, b in re.findall(r"\[hook-op\] layer (\d+) (\w+): ops (\d+)\.\.(\d+)", log):
        if int(a) <= op <= int(b):
            return {"op": op, "op_name": name, "layer": int(L), "in": f"hook {when} {a}..{b}"}
    return {"op": op, "op_name": name, "layer": None}


FRAME_RE = re.compile(r"#\d+ 0x[0-9A-Fa-f]+ in (.*?) (?:\(\) )?at ")
KDIR_RE = re.compile(r"kernels/([A-Za-z0-9_]+)/\d+/")


def _short(f):
    while re.search(r"<[^<>]*>", f):
        f = re.sub(r"<[^<>]*>", "", f)
    return f.replace("kernel_main::", "")


def _triage_summary(triage, top=14):
    """Callstacks grouped: (kernel, risc, innermost frames) -> cores, devices; the stuck sites at a glance."""
    sec = re.search(r"dump_callstacks\.py:\n(.*?)\n\n", triage, re.S)
    if not sec:
        return []
    groups, devs = Counter(), {}
    for r in csv.DictReader(io.StringIO(sec.group(1))):
        cs = (r.get("Kernel Callstack") or "").replace("\\n", "\n")
        name = r.get("Kernel Name", "")
        if name.startswith("cq_") or "idle_erisc" in name:
            continue
        kd = KDIR_RE.findall(cs)
        frames = [_short(f) for f in FRAME_RE.findall(cs) if f not in ("kernel_main", "main")][:2] or ["kernel_main"]
        k = (f"{r.get('Kernel ID')}:{kd[0] if kd else name}", r.get("RiscV"), " < ".join(frames))
        groups[k] += 1
        devs.setdefault(k, set()).add(r.get("Dev"))
    return [
        {"kernel": k, "risc": rv, "frames": fr, "cores": n, "devices": len(devs[(k, rv, fr)])}
        for (k, rv, fr), n in groups.most_common(top)
    ]


def _pause(out, entry):
    """A gap in the queue (e.g. installing a rebuilt ttnn): the device stays idle while entry["path"] exists (<= 1 h)."""
    t0 = time.time()
    path = Path(entry["path"])
    while path.exists() and time.time() - t0 < 3600:
        (out / "heartbeat").write_text(f"{dt.datetime.utcnow().isoformat()} paused on {path}\n")
        time.sleep(10)
    row = {
        "id": f"{dt.datetime.utcnow():%m%d-%H%M%S}-{entry['name']}",
        "name": entry["name"],
        "workload": "pause",
        "opts": {},
        "start": dt.datetime.utcfromtimestamp(t0).isoformat(timespec="seconds"),
        "wall_s": round(time.time() - t0),
        "result": "PASS",
        "calls": {"total": 0},
        "git": _git(),
        "errors": [],
    }
    with open(out / "runs.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def run_one(out, entry, idx):
    if entry["workload"] == "pause":
        return _pause(out, entry)
    rid = f"{dt.datetime.utcnow():%m%d-%H%M%S}-{entry['name']}"
    rdir = out / "runs" / rid
    rdir.mkdir(parents=True)
    test, env, args = WORKLOADS[entry["workload"]](entry.get("opts", {}))
    env = {**BASE_ENV, **env, **entry.get("env", {})}
    if entry.get("block_opts"):  # (TtK3TokenSplitMoE variants: acc_on_ports, deterministic)
        env["GLM_TS_BLOCK_OPTS"] = json.dumps(entry["block_opts"])
    if entry.get("hooks") or entry.get("hook_args"):  # (ops / gaps / syncs around every MoE call: std_hooks.py)
        env["GLM_HANG_HOOKS"] = str(REPO / (entry.get("hooks") or f"{TESTS}/hang/std_hooks.py"))
        env["GLM_HANG_HOOK_ARGS"] = json.dumps(entry.get("hook_args", {}))
    (rdir / "spec.json").write_text(json.dumps({"entry": entry, "test": test, "env": env, "args": args}, indent=1))
    full = {k: v for k, v in os.environ.items() if k != "TT_MESH_GRAPH_DESC_PATH"} | env
    full["PATH"] = f"{Path(sys.executable).parent}:{full.get('PATH', '')}"
    cmd = ["timeout", str(entry.get("timeout", 7200)), "scripts/run_safe_pytest.sh", "--mesh", "--no-precompile"]
    cmd += args + [test, "-s"]
    t0 = time.time()
    with open(rdir / "log.txt", "w") as f:
        rc = subprocess.run(cmd, cwd=REPO, env=full, stdout=f, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    log = (rdir / "log.txt").read_text(errors="replace")
    res = re.findall(r"SAFE_PYTEST_RESULT: ([A-Z]+)", log)
    result = res[-1] if res else ("TIMEOUT" if rc == 124 else f"RC{rc}")
    row = {
        "id": rid,
        "name": entry["name"],
        "workload": entry["workload"],
        "opts": entry.get("opts", {}),
        "start": dt.datetime.utcfromtimestamp(t0).isoformat(timespec="seconds"),
        "wall_s": round(wall),
        "result": result,
        "calls": _calls(log),
        "git": _git(),
        "errors": [l[:300] for l in re.findall(r"^E   +[A-Z].*$", log, re.M)[:3]],
    }
    if result == "HANG":
        tri = re.search(r"=== TRIAGE LOG ===.*?=== END TRIAGE LOG ===", log, re.S)
        triage = tri.group(0) if tri else ""
        (rdir / "triage.txt").write_text(triage)
        row["hang"] = _hang_site(log, triage)
        row["triage"] = _triage_summary(triage)
        row["triage_lines"] = triage.count("\n")
    with open(out / "runs.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def cmd_run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    queue = Path(a.queue or out / "queue.json")
    until = None
    if a.until:
        h, m = map(int, a.until.split(":"))
        now = dt.datetime.utcnow()
        until = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if until <= now:
            until += dt.timedelta(days=1)
    me = Path(__file__).resolve()
    mtime = me.stat().st_mtime
    while True:
        if me.stat().st_mtime != mtime:  # (edited: re-exec at the run boundary, same arguments)
            print("[campaign] campaign.py changed: re-exec", flush=True)
            os.execv(sys.executable, [sys.executable, str(me), *sys.argv[1:]])
        if until and dt.datetime.utcnow() >= until:
            print(f"[campaign] deadline {until:%H:%M} UTC reached", flush=True)
            return 0
        done = Counter(r["name"] for r in _rows(out) if r["result"] in ("PASS", "HANG", "FAIL"))
        entries = json.loads(queue.read_text())
        nxt = next((e for e in entries if done[e["name"]] < e.get("reps", 1)), None)
        if nxt is None:
            print("[campaign] queue done", flush=True)
            return 0
        (out / "heartbeat").write_text(
            f"{dt.datetime.utcnow().isoformat()} running {nxt['name']} #{done[nxt['name']] + 1}\n"
        )
        row = run_one(out, nxt, done[nxt["name"]])
        print(
            f"[campaign] {row['id']}: {row['result']} in {row['wall_s']} s, {row['calls']['total']} MoE calls"
            + (f" | hang {row.get('hang')}" if row["result"] == "HANG" else "")
            + (f" | {row['errors'][:1]}" if row["result"] not in ("PASS", "HANG") else ""),
            flush=True,
        )
        if row["result"] == "HANG":
            for g in row.get("triage", [])[:8]:
                print(
                    f"[campaign]   {g['cores']:4d} cores / {g['devices']} chips: {g['kernel']} {g['risc']}: {g['frames']}",
                    flush=True,
                )
            if nxt.get("continue_on_hang"):  # (a repro being repeated for its rate: keep going, triage saved)
                print("[campaign] HANG: continuing (continue_on_hang)", flush=True)
                continue
            print("[campaign] HANG: stopping for narrowing", flush=True)
            return 3
        if row["result"] not in ("PASS",) and a.stop_on_fail:
            return 4


def cmd_report(a):
    rows = _rows(Path(a.out))
    by = {}
    for r in rows:
        by.setdefault(r["name"], []).append(r)
    print(
        f"{'config':34s} {'runs':>4s} {'pass':>4s} {'hang':>4s} {'fail':>4s} {'MoE calls':>11s} {'hangs / 1M calls':>22s}"
    )
    for name, rs in by.items():
        c = Counter(r["result"] for r in rs)
        calls = sum(r["calls"]["total"] for r in rs)
        h = c["HANG"]
        if calls:
            rate = f"{h / calls * 1e6:.1f}" if h else f"0 (< {3 / calls * 1e6:.1f} at 95%)"
        else:
            rate = "-"
        print(f"{name:34s} {len(rs):4d} {c['PASS']:4d} {h:4d} {len(rs) - c['PASS'] - h:4d} {calls:11d} {rate:>22s}")
    for r in rows:
        if r["result"] == "HANG":
            print(f"HANG {r['id']}: {r.get('hang')}")
    out = Path(a.out)
    perf = []
    for r in rows:  # (scratch_flat_combine_rf runs: flat / fused us per routing)
        log = out / "runs" / r["id"] / "log.txt"
        if r["name"].startswith("perf") and log.exists():
            for m in re.finditer(
                r"FC_SUMMARY (\S+)\s+pipelined: flat (\d+) \| fused (\d+) us", log.read_text(errors="replace")
            ):
                perf.append((r["name"], m.group(1), int(m.group(2)), int(m.group(3))))
    if perf:
        print(f"\n{'perf run':28s} {'routing':16s} {'flat us':>8s} {'fused us':>9s}")
        for n, rt, f, fu in perf:
            print(f"{n:28s} {rt:16s} {f:8d} {fu:9d}")


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--queue")
    r.add_argument("--until", help="HH:MM UTC")
    r.add_argument("--stop-on-fail", action="store_true")
    q = sp.add_parser("report")
    q.add_argument("--out", required=True)
    a = p.parse_args()
    sys.exit(cmd_run(a) if a.cmd == "run" else cmd_report(a))


if __name__ == "__main__":
    main()
