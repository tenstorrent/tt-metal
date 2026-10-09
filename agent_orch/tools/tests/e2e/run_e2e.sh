#!/usr/bin/env bash
# Offline end-to-end test of a whole campaign on this machine: toy repo, toy eval, stub claude.
#   run_e2e.sh [scratch dir]
set -euo pipefail
T="$(cd "$(dirname "$0")" && pwd)"
SRC="$(git -C "$T" rev-parse --show-toplevel)/agent_orch"
S="${1:-$(mktemp -d)}"; rm -rf "$S"; mkdir -p "$S"
export DREAM_REGISTRY="$S/registry.json" DREAM_CLAUDE="$T/stub_claude.py" DREAM_PY="${DREAM_PY:-$(command -v python3)}"
R="$S/toy"; mkdir -p "$R/src"; cd "$R"; git init -q; git config user.email t@t; git config user.name t
cp -r "$SRC" agent_orch; rm -rf agent_orch/campaigns/*; find agent_orch -name __pycache__ -prune -exec rm -rf {} +
printf "__pycache__/\n" > .gitignore
echo 10.0 > src/speed.txt
mkdir -p agent_orch/campaigns/toy && cp "$T/toy_eval.py" agent_orch/campaigns/toy/eval.py
cat > agent_orch/campaigns/toy/dream.yaml <<YAML
name: toy
description: toy campaign for the offline end-to-end test
machine: local
dream_home: $S/home
editable: ["src/*"]
rules: ["Keep speed.txt a single number"]
forbidden_patterns: ['FORBIDDEN']
eval: {command: python3 agent_orch/campaigns/toy/eval.py, direction: minimize, unit: ms, gates: ["check >= 1"],
       timeout_s: 60, baseline_runs: 2, drift_check: true}
build: {command: "", check_file: null}
resource: {reset_command: "true"}
budget: {max_attempts: 9, max_hours: 1, max_usd: 100}
search: {policy: fresh, W: 2, R: 2, max_rounds: 3}
YAML
echo "# toy brief" > agent_orch/campaigns/toy/brief.md
git add -A; git commit -qm "toy repo"
D="$R/agent_orch/bin/dream"
"$D" check toy
STUB_DELAY=12 "$D" start toy
# interrupt mid-step, then resume: the unfinished step must be re-run, nothing lost or duplicated
sleep 6
"$D" stop toy
st=$("$D" status toy --json | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['state'], d['alive'])")
[[ "$st" == "stopped False" ]] || { echo "expected stopped, got $st"; exit 1; }
[[ -z "$(ls "$S/home/toy/sessions" 2>/dev/null)" ]] || { echo "sessions left behind"; exit 1; }
echo "stopped OK; resuming"
"$D" resume toy
grep -q "resuming unfinished step" "$S/home/toy/logs/driver.log" || { echo "resume did not re-run the interrupted step"; exit 1; }
for i in $(seq 1 120); do
  st=$("$D" status toy --json | python3 -c "import json,sys;print(json.load(sys.stdin)['state'])")
  [[ "$st" == running ]] || break; sleep 2
done
"$D" status toy
"$D" report toy --out "$S/report.html" >/dev/null
python3 - "$S" <<'PY'
import json, subprocess, sys
from pathlib import Path
S = Path(sys.argv[1]); H = S / "home/toy"
st = json.loads((H / "status.json").read_text())
assert st["state"] == "finished", st
data = json.loads((H / "report/data.json").read_text())
nodes = [n for r in data["rounds"] for n in r["nodes"]]
lost = [x for r in data["rounds"] for x in r["lost"]]
print(f"rounds={len(data['rounds'])} nodes={len(nodes)} lost={lost} best={data['best']['node']} {data['best']['score']}"
      f" budget={data['budget']} reason={st['message']!r}")
assert data["budget"]["attempts"] == 9, "hard stop at max_attempts"
assert "r01-b02-a02" in lost, "lost worker recorded"
inv = [n for n in nodes if not n["valid"]]
assert any(n["id"] == "r01-b01-a02" and n["fail"] == "accuracy_fail" for n in inv), inv
assert data["insights"]["worked"], "round summary insights"
assert len(data["rounds"]) >= 2 and data["rounds"][1]["root"].endswith("/n/" + data["rounds"][1]["root"].split("/")[-1])
refs = subprocess.run(["git", "-C", str(S / "toy"), "for-each-ref", "--format=%(refname)", "refs/dream/toy/"],
                      capture_output=True, text=True).stdout.split()
assert "refs/dream/toy/ledger" in refs and any("/n/" in r for r in refs), refs
tags = subprocess.run(["git", "-C", str(S / "toy"), "tag"], capture_output=True, text=True).stdout.strip()
branches = subprocess.run(["git", "-C", str(S / "toy"), "branch", "--list"], capture_output=True, text=True).stdout
assert tags == "", "no tags created"
assert "dream/toy/best" in branches and branches.count("\n") == 2, branches
diff = subprocess.run(["git", "-C", str(S / "toy"), "diff", "--stat", "HEAD", "dream/toy/best"], capture_output=True,
                      text=True).stdout
assert "src/speed.txt" in diff and "agent_orch" not in diff, diff
best_val = subprocess.run(["git", "-C", str(S / "toy"), "show", "dream/toy/best:src/speed.txt"], capture_output=True,
                          text=True).stdout.strip()
print("best branch speed:", best_val, "| diff:", diff.strip().splitlines()[-1])
print("e2e: OK")
PY
