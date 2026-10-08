#!/usr/bin/env bash
# Replay the walkthrough example and check the replay scores (v0 1.23, A 1.24, B 1.11).
set -euo pipefail
T="$(cd "$(dirname "$0")" && pwd)"
PY="${DREAM_PY:-python3}"
out=$(mktemp -d)
"$PY" "$T/../replay.py" --fixture "$T/fixture_example.fixture" --cost 0.005 --bonus 0.01 --beta-sweep 0.6 \
    --policy "$T/../../policies/v0/policy.py" --policy "$T/policies/revision_a.py" --policy "$T/policies/revision_b.py" \
    --out "$out/replay.json" --traces "$out/traces.jsonl"
"$PY" - "$out/replay.json" <<'PY'
import json, sys
res = json.load(open(sys.argv[1]))
want = {"v0/policy.py": (1.23, 16), "revision_a.py": (1.24, 12), "revision_b.py": (1.11, 8)}
for path, r in res.items():
    key = next(k for k in want if path.endswith(k))
    V, n = want[key]
    ep = r["episodes"][0]
    assert abs(r["objective_V"] - V) < 1e-9 and ep["attempts"] == n, (key, r["objective_V"], ep["attempts"])
print("replay fixture: OK")
PY
