"""Dummy end-to-end walk of the Agent Loop — NO API key, NO hardware.

    python demo_walk.py

Sets up a throwaway model git repo, a fake baseline profile, and a fake editor,
then drives the real engine (real ROUTE / APPLY / LOG / CHECK_EXIT; mock leaves
for the stages not built yet) and prints the state trace + ledger. Use it to see
how the loop moves before any real handlers/keys are in play.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from agent import engine
from agent.handlers import build_handlers
from agent.loop_context import LoopContext
from agent.run import Run

tmp = Path(tempfile.mkdtemp())
try:
    # 1. a dummy "model" that is a real git repo (APPLY records its clean SHA)
    model = tmp / "model"
    model.mkdir()
    (model / "model.py").write_text("x = 1\n")
    for c in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *c], cwd=model, check=True)
    subprocess.run(["git", "add", "."], cwd=model, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=model, check=True)

    # 2. a run dir at BEFORE_LOOP_DONE with a dummy baseline profile + manifest
    run = Run.create(
        tmp / "runs",
        config={"config": {"model_root": str(model)}, "env": {}, "pathmap": {"model_files": ["model.py"]}},
        run_id="DEMO",
    )
    run.state_path.write_text(
        json.dumps(
            {
                "run_id": "DEMO",
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
                        "tags": {"op_class": "matmul", "fidelity": "hifi2"},
                    },
                    {
                        "id": "reduction",
                        "device_ms": 2.052,
                        "pct": 16.9,
                        "count": 50,
                        "tags": {"op_class": "reduction"},
                    },
                ],
            }
        )
    )

    # 3. a tiny playbook index (matmul levers) + a FAKE editor (no SDK, no key)
    idx = [
        {
            "id": a,
            "title": a,
            "file": "f",
            "lever_type": "single-shot",
            "op_class": ["matmul"],
            "bound": ["*"],
            "rank": ["*"],
            "fidelity": ["*"],
            "grid": ["*"],
            "dispatch": ["*"],
            "memory": ["*"],
            "regime": ["*"],
        }
        for a in ("mlp-fidelity-walk", "subblock-unlock", "fuse-activation-matmul")
    ]
    ctx = LoopContext.from_run(run, index=idx)
    ctx.deps["edit_runner"] = lambda **k: {
        "files": ["model.py"],
        "summary": "fake edit",
        "model": "mock",
        "usage": {"tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0, "latency_s": 0.0},
    }

    # 4. drive the real engine
    final = engine.run(ctx, build_handlers())

    print("\n--- STATE TRACE ---")
    for ln in (run.dir / "events.jsonl").read_text().splitlines():
        e = json.loads(ln)
        if e["status"] == "done":
            print(f"  iter {e['iteration']}  {e['stage']:16} -> {e['detail'].replace('-> ', '')}")
    m = ctx.state["metric"]
    print(f"\nFINAL: {final}   device_ms {m['baseline']} -> {m['current']}  (target {m['target']})")
    print(f"APPLY recorded clean SHA: {ctx.state['git_sha_clean'][:12]}... (len {len(ctx.state['git_sha_clean'])})")
    print(f"current_profile now: {ctx.state.get('current_profile')}")
    print("\n--- LEDGER (one row per experiment) ---")
    for ln in (run.dir / "ledger.jsonl").read_text().splitlines():
        r = json.loads(ln)
        print(f"  iter {r['iteration']}: {r['lever']:20} {r['before']} -> {r['after']} ms  [{r['result']}]")
finally:
    shutil.rmtree(tmp)
