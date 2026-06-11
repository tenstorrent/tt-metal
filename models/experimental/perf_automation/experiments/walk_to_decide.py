"""Full ROUTE -> DECIDE pass with ALL real handlers, leaves injected — no key/hw.

    python experiments/walk_to_decide.py

Uses the REAL GUIDELINES routing (so ROUTE finds real matmul levers) and the real
VERIFY/GATE_PCC/REMEASURE/DECIDE logic; only the four edge leaves are faked
(editor, lever-picker, pcc, tracy). Persists everything under
experiments/route_to_decide/ so you can inspect the brief, profiles, ledger, and
the DECIDE verdict, then revert/tweak any stage.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # perf_automation on path

from agent import engine, states
from agent.looplog import make_logger
from agent.handlers import build_handlers
from agent.loop_context import LoopContext
from agent.run import Run

EXP = Path(__file__).parent / "route_to_decide"
if EXP.exists():
    shutil.rmtree(EXP)
EXP.mkdir(parents=True)

# dummy model git repo (APPLY records its clean SHA; VERIFY parses/imports it)
model = EXP / "model"
model.mkdir()
(model / "model.py").write_text("x = 1\n")
for c in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
    subprocess.run(["git", *c], cwd=model, check=True)
subprocess.run(["git", "add", "."], cwd=model, check=True)
subprocess.run(["git", "commit", "-qm", "init"], cwd=model, check=True)

run = Run.create(
    EXP / "runs",
    config={"config": {"model_root": str(model)}, "env": {}, "pathmap": {"model_files": ["model.py"]}},
    run_id="PASS",
)
run.state_path.write_text(
    json.dumps(
        {
            "run_id": "PASS",
            "state": "BEFORE_LOOP_DONE",
            "iteration": 0,
            "metric": {
                "name": "device_ms",
                "unit": "ms",
                "direction": "min",
                "baseline": 12.091,
                "current": 12.091,
                "target": 11.0,
            },
            "max_iter": 25,
            "budget_usd": 5.0,
            "cost_usd": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
            "git_sha_clean": None,
            "candidates": [],
            "tried": [],
            "crash_retries": 0,
            "code_fix_attempts": 0,
            "pcc_fix_attempts": 0,
            "current_profile": None,
            "last_error": None,
        }
    )
)
# real BGE-M3-style matmul bottleneck so ROUTE matches real GUIDELINES sections
(run.profiles_dir / "baseline_profile.json").write_text(
    json.dumps(
        {
            "device_ms": 12.091,
            "wall_ms": 13291,
            "buckets": [
                {
                    "id": "matmul",
                    "device_ms": 6.741,
                    "pct": 55.7,
                    "count": 96,
                    "tags": {
                        "op_class": "matmul",
                        "fidelity": "hifi2",
                        "bound": "slow",
                        "rank": "time",
                        "grid": "full",
                        "dispatch": "ok",
                        "memory": "dram_interleaved",
                        "regime": "na",
                    },
                },
                {"id": "reduction", "device_ms": 2.052, "pct": 16.9, "count": 50, "tags": {"op_class": "reduction"}},
            ],
        }
    )
)

ctx = LoopContext.from_run(run)  # REAL GUIDELINES index
ctx.deps["edit_runner"] = lambda **k: {
    "files": ["model.py"],
    "summary": "fake edit",
    "model": "mock",
    "usage": {"tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0},
}
ctx.deps["select_runner"] = lambda **k: {
    "lever": k["candidates"][0],
    "reasoning": "fake: first candidate",
    "model": "mock",
    "usage": {"tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0},
}
ctx.deps["pcc_runner"] = lambda c: {"status": "ok", "pcc": 0.997}


def _fake_measure(c):
    prof = json.loads(json.dumps(c.current_profile()))
    for b in prof["buckets"]:
        if b["id"] == c.state.get("current_bucket"):
            b["device_ms"] = round(b["device_ms"] * 0.5, 4)
    prof["device_ms"] = 11.4
    return [prof, {**prof, "device_ms": 11.5}, {**prof, "device_ms": 11.45}]  # 3 runs -> median 11.45


ctx.deps["measure_runner"] = _fake_measure

reached = engine.run(ctx, build_handlers(), stop_after={states.DECIDE}, log=make_logger())

print("\n=== ROUTE -> DECIDE PASS ===")
for ln in (run.dir / "events.jsonl").read_text().splitlines():
    e = json.loads(ln)
    if e["status"] == "done":
        print(f"  {e['stage']:12} -> {e['detail'].replace('-> ', '')}")
d = ctx.state["last_decision"]
print(f"\nparked at: {reached}  (ran through DECIDE)")
print(f"selected lever : {ctx.state['selected_lever']}   ({ctx.state.get('select_reasoning')})")
print(f"clean SHA      : {ctx.state['git_sha_clean'][:12]}...")
print(f"PCC verdict    : {ctx.state['last_verdict']}")
print(f"REMEASURE      : before {d['before']} -> after {d['after']} ms  (spread {d['spread']}, {d['runs']} runs)")
print(f"DECIDE         : {d['result']}" + (f" ({d.get('reason')})" if d.get("reason") else ""))
print(f"\nartifacts saved under: {EXP}")
for f in sorted(run.dir.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(EXP)}")
