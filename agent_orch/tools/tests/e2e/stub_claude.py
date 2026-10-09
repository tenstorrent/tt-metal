#!/usr/bin/env python3
"""Stand-in for `claude -p` in the offline end-to-end test. Behaves like the three session kinds:
worker (edit src/speed.txt, eval, commit), policy developer (writes nothing: keeps the policy),
round summary (writes summary.md + insights.json). Node r01-b02-a02 never commits (tests 'lost');
r01-b01-a02 writes a non-positive speed (tests an invalid attempt)."""
import hashlib, json, re, subprocess, sys
from pathlib import Path

prompt = sys.argv[sys.argv.index("-p") + 1]


def result(text, usd):
    print(json.dumps({"type": "result", "result": text, "total_cost_usd": usd, "num_turns": 3}), flush=True)


import os

MODE = os.environ.get("ROUND_ROOT", "origin")
if "NODE_ID=" in prompt or "policy-development" in prompt:
    assert f"ROUND_ROOT={MODE}" in prompt, "prompt does not state the round_root mode"
if "NODE_ID=" in prompt:
    import os, time

    time.sleep(float(os.environ.get("STUB_DELAY", "0")))  # lets the test stop the campaign mid-step
    node = re.search(r"NODE_ID=(\S+)", prompt).group(1)
    camp = re.search(r"CAMPAIGN=(\S+)", prompt).group(1)
    if node == "r01-b02-a02":
        result("gave up", 0.3)
        sys.exit(0)
    # isolation: worktrees must not see the user's repo refs; r01-b03.. is never used, so r02-b01-a01 peeks outside
    seen = subprocess.run(["git", "for-each-ref", "--format=%(refname)"], capture_output=True, text=True).stdout.split()
    assert all(r.startswith(f"refs/dream/{camp}/") for r in seen), f"worktree sees foreign refs: {seen}"
    if node == "r02-b01-a01":
        main = subprocess.run(
            ["git", "config", "--get", "dream.mainRepo"], capture_output=True, text=True
        ).stdout.strip()
        cmd = f"git -C {main} log --oneline --all"
        print(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": cmd}}]},
                }
            ),
            flush=True,
        )
    nd = Path(f"agent_orch/campaigns/{camp}/attempts/{node}")
    cur = float(Path("src/speed.txt").read_text())
    h = int(hashlib.sha1(node.encode()).hexdigest(), 16) % 1000 / 1000
    new = -1.0 if node == "r01-b01-a02" else round(cur * (0.80 + 0.25 * h), 3)
    Path("src/speed.txt").write_text(f"{new}\n")
    meta = json.loads((nd / "node.json").read_text())
    meta.update(mechanism=f"scale speed {cur} -> {new}", tags=["toy"], worker="stub")
    (nd / "node.json").write_text(json.dumps(meta, indent=2))
    (nd / "proposal.md").write_text(f"# {node}\nscale\n")
    (nd / "context.md").write_text("## Files read\n- src/speed.txt\n")
    subprocess.run(["agent_orch/bin/dream", "eval", "--campaign", camp, "--node", node], check=True)
    (nd / "reflection.md").write_text(f"# {node} result\n## What a child of this node should try next\n- scale again\n")
    out = subprocess.run(
        ["agent_orch/bin/dream", "commit", "--campaign", camp, "--node", node],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    result(out, 0.5)
elif "Summarize round" in prompt:
    paths = re.findall(r"(/\S+/rounds/r\d+/(?:summary\.md|insights\.json))", prompt)
    for p in paths:
        p = Path(p.rstrip(":"))
        if p.name == "summary.md":
            p.write_text("# summary\nstub summary\n")
        else:
            p.write_text(
                json.dumps(
                    {
                        "worked": ["scaling down (stub)"],
                        "dead_ends": ["negative speed (r01-b01-a02)"],
                        "open_leads": ["scale more"],
                    }
                )
            )
    result("done", 0.05)
else:  # policy developer
    result(json.dumps({"current": "v0", "winner": "v0", "summary": "stub kept the policy"}), 0.2)
