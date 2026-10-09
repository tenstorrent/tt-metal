#!/usr/bin/env bash
# Offline tests for the Dream-RSI tools (no device, no model calls).
#   1. replay of the walkthrough fixture: V = 1.23 / 1.24 / 1.11
#   2. (optional) regression on the exported rmsnorm-prefill campaign: tools/tests/regress_export.sh
set -euo pipefail
T="$(cd "$(dirname "$0")" && pwd)"
D="$T/../../bin/dream"
out=$(mktemp -d)
"$D" replay --fixture "$T/fixture_example.fixture" --cost 0.005 --bonus 0.01 --beta-sweep 0.6 \
    --policy "$T/../../policies/library/parallel-refine/policy.py" --policy "$T/policies/revision_a.py" \
    --policy "$T/policies/revision_b.py" --out "$out/replay.json" --traces "$out/traces.jsonl"
python3 - "$out/replay.json" <<'PY'
import json, sys
res = json.load(open(sys.argv[1]))
want = {"parallel-refine/policy.py": (1.23, 16), "revision_a.py": (1.24, 12), "revision_b.py": (1.11, 8)}
for path, r in res.items():
    key = next(k for k in want if path.endswith(k))
    V, n = want[key]
    ep = r["episodes"][0]
    assert abs(r["objective_V"] - V) < 1e-9 and ep["attempts"] == n, (key, r["objective_V"], ep["attempts"])
print("replay fixture: OK")
PY
