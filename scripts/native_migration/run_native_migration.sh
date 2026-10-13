#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Port a Python ttnn.generic_op operation to a C++ device operation with the native-migration flow
# (tt_metal/third_party/tt_ops_code_gen), in place on this checkout, and time it.
#
#   scripts/native_migration/run_native_migration.sh <op> <python-op-dir> <tests.yaml>
#   scripts/native_migration/run_native_migration.sh fabric_reduce_scatter \
#       ttnn/ttnn/operations/examples/fabric_reduce_scatter scripts/native_migration/fabric_reduce_scatter.yaml
#
# Before the first run: ./build_metal.sh and ./create_venv.sh (the flow rebuilds incrementally in build_Release),
# the `claude` CLI on PATH and logged in, and the devices the op's tests need (fabric_reduce_scatter: a 2x2 or larger
# Wormhole mesh). The flow commits its work on this branch as `[codegen-port]` commits.
#
# Output in $WORK (default generated/native_migration/<op>): result.json (the outcome), work/logs (every step and
# agent turn), and timing.txt: wall time and the agent time of translation, review and fix turns.
set -euo pipefail

OP=${1:?op name}
PY_DIR=${2:?python op dir}
TESTS=$(realpath "${3:?tests file}")
ROOT=$(git rev-parse --show-toplevel)
FLOW=$ROOT/tt_metal/third_party/tt_ops_code_gen
WORK=${WORK:-$ROOT/generated/native_migration/$OP}

git -C "$ROOT" submodule update --init --checkout tt_metal/third_party/tt_ops_code_gen
mkdir -p "$WORK"
cd "$ROOT"
# shellcheck disable=SC1091
source python_env/bin/activate
export TT_METAL_HOME=$ROOT PYTHONPATH=$ROOT TT_METAL_ENV=dev

start=$(date +%s)
set +e
"$FLOW/native_migrate/bin/native-migrate" --checkout "$ROOT" --op "$OP" --python-op-dir "$PY_DIR" \
    --tests-file "$TESTS" --work-dir "$WORK/work" --result-json "$WORK/result.json" \
    --model "${NM_MODEL:-opus}" --effort "${NM_EFFORT:-medium}" --timeout-s "${NM_TIMEOUT_S:-10800}" \
    2>&1 | tee "$WORK/flow.log"
rc=${PIPESTATUS[0]}
set -e
wall=$(($(date +%s) - start))

python3 - "$WORK" "$wall" "$rc" <<'PY' | tee "$WORK/timing.txt"
import json, re, sys
from datetime import datetime
from pathlib import Path

work, wall, rc = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
turns = []
for log in sorted((work / "work" / "logs").glob("*-nm-*.jsonl")):
    for line in log.open():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") != "claude":
            continue
        end = datetime.fromisoformat(e["ts"].replace("Z", "+00:00")).timestamp()
        dur = ((e.get("raw") or {}).get("duration_ms") or 0) / 1000
        turns.append((e.get("invocation_label") or "", end - dur, end, dur))
tr = [t for t in turns if re.match(r"(translate|interface)", t[0])]
translate = (max(t[2] for t in tr) - min(t[1] for t in tr)) if tr else 0.0
review, cur = 0.0, None  # reviews running at the same time count once
for _, s, e, _ in sorted(t for t in turns if t[0].startswith("review")):
    if cur is None or s > cur[1]:
        review += (cur[1] - cur[0]) if cur else 0.0
        cur = [s, e]
    else:
        cur[1] = max(cur[1], e)
review += (cur[1] - cur[0]) if cur else 0.0
fix = sum(t[3] for t in turns if t[0].startswith("fix"))
try:
    result = json.loads((work / "result.json").read_text())
except (OSError, ValueError):
    result = {}
print(f"outcome            {result.get('outcome')} (exit {rc})")
print(f"wall               {wall / 60:.1f} min")
print(f"translate          {translate / 60:.1f} min")
print(f"review             {review / 60:.1f} min ({sum(t[0].startswith('review') for t in turns)} turn(s))")
print(f"fix                {fix / 60:.1f} min")
print(f"translate+review   {(translate + review + fix) / 60:.1f} min (with fix turns)")
for label, _, _, dur in turns:
    print(f"  {label:24s} {dur / 60:5.1f} min")
PY
exit "$rc"
