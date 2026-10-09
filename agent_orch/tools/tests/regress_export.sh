#!/usr/bin/env bash
# Regression on the real exported rmsnorm-prefill campaign (58 attempts, policies v0-v5):
# load its git bundle into a scratch clone, move the legacy tags under refs/dream/, and check that
# the new replay reproduces every recorded policy's objective V, and that history.md + the report render.
#   regress_export.sh [<results ref>] [<scratch dir>]
set -euo pipefail
T="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$T" rev-parse --show-toplevel)"
REF="${1:-origin/opgen_hackathon/nstamatovic_dream_rsi_v1_results}"
S="${2:-$(mktemp -d)}"
E=agent_orch/campaigns/rmsnorm-prefill/export
mkdir -p "$S"
[[ -d "$S/repo" ]] || git clone -q --shared --no-checkout "$REPO" "$S/repo"
git -C "$REPO" show "$REF:$E/dream_rmsnorm-prefill.bundle" > "$S/bundle"
git -C "$S/repo" fetch -q "$S/bundle" 'refs/tags/dream/*:refs/legacy/dream/*'
git -C "$S/repo" for-each-ref --format='%(refname) %(objectname)' refs/legacy/dream/rmsnorm-prefill/ |
  while read -r ref sha; do git -C "$S/repo" update-ref "refs/dream/${ref#refs/legacy/dream/}" "$sha"; done
# the old campaign has no base ref: the root's parent is its base
git -C "$S/repo" update-ref refs/dream/rmsnorm-prefill/base "$(git -C "$S/repo" rev-parse refs/dream/rmsnorm-prefill/root~1)"
rm -rf "$S/home/rmsnorm-prefill/ledger"; mkdir -p "$S/home/rmsnorm-prefill"
git -C "$REPO" archive "$REF" "$E/ledger" | tar -x -C "$S" && mv "$S/$E/ledger" "$S/home/rmsnorm-prefill/ledger"
echo "scratch: $S ($(git -C "$S/repo" for-each-ref refs/dream/rmsnorm-prefill/n/ | wc -l) nodes)"
DREAM_HOME="$S/home" PYTHONPATH="$REPO/agent_orch/tools" python3 - "$S" "$REPO" <<'PY'
import json, sys
from pathlib import Path
from dream.campaign import Campaign, load_spec
from dream.replay import replay_policy
from dream.tree import load_round, recorded_rounds
from dream.history import write_md
from dream import report

S, REPO = Path(sys.argv[1]), Path(sys.argv[2])
cfg = load_spec(REPO / "agent_orch/campaigns/rmsnorm-prefill/dream.yaml")
c = Campaign("rmsnorm-prefill", cfg, S / "repo")
d = {"W": 4, "R": 4, "beta": 0.6}
noise = json.loads((c.ledger / "baseline.json").read_text())["noise_pct"]
ok = True
# v3's replay.json was written before the 3 HiFi2 attempts of r04 were overridden (rule added 2026-10-08),
# so it matches only when the overrides are not applied; every other version was scored with them.
pre_rule = {"v3"}
for v in ["v0", "v1", "v2", "v3", "v4", "v5"]:
    pdir = c.ledger / "policies" / v
    rec = json.loads((pdir / "replay.json").read_text())
    rounds = [load_round(c, r, apply_overrides=v not in pre_rule) for r in rec["rounds"]]
    got = replay_policy(pdir / "policy.py", rounds, noise, d, cfg["dreaming"])
    same = abs(got["objective_V"] - rec["objective_V"]) < 1e-9
    ok &= same
    print(f"{v}: recorded V {rec['objective_V']:.4f} on rounds {rec['rounds']}, replayed {got['objective_V']:.4f}  {'OK' if same else 'MISMATCH'}")
write_md(c, S / "history.md")
data = report.collect(c)
assert data["best"]["node"] == "r05-b01-a01" and abs(data["best"]["score"] - 1.5696) < 1e-9, data["best"]
assert sum(len(r["nodes"]) for r in data["rounds"]) == 58, "expected 58 attempts"
print("report:", report.build(c, S / "report"))
print("history:", S / "history.md")
sys.exit(0 if ok else 1)
PY
echo "export regression: OK"
