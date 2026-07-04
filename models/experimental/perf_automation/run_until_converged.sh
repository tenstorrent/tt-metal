#!/usr/bin/env bash
# Auto-iterate optimization PASSES until convergence — ONE job, no manual restarts.
#
# A single `agent.loop` run is one greedy pass: it tries each lever once and stops
# when all buckets are swept. But optimizations interact, so a lever that was a
# no-gain early in a pass can become a gain once other gains are applied. This
# wrapper therefore re-profiles from the (now more-optimized, committed) model and
# sweeps again, repeating until a whole pass keeps ZERO gains = converged.
#
# Usage:  DEVICES=0,1 ./run_until_converged.sh [<demo_dir>] [<task>]
set -u
cd /home/ttuser/tt-metal/models/experimental/perf_automation || exit 1
export TT_METAL_HOME=/home/ttuser/tt-metal PYTHONPATH=/home/ttuser/tt-metal
export PATH=/home/ttuser/tt-metal/python_env/bin:/home/ttuser/.local/bin:$PATH
set -a; source .env.agent 2>/dev/null; set +a

DEMO="${1:-/home/ttuser/tt-metal/models/demos/hf_seamless_m4t_medium}"
TASK="${2:-t2t}"
DEVICES="${DEVICES:-0,1}"
MAX_PASSES="${MAX_PASSES:-8}"          # safety cap; convergence is usually 2-4 passes
RELDEMO="${DEMO#*/tt-metal/}"
PERF="$RELDEMO/tests/e2e/test_${TASK}_perf.py::test_${TASK}_perf"

pass=0
while [ "$pass" -lt "$MAX_PASSES" ]; do
    pass=$((pass + 1))
    echo "================= $(date) : PASS $pass / $MAX_PASSES ================="
    # re-profile the CURRENT committed model as this pass's baseline, then sweep once
    python -m agent.before_loop "$DEMO" --metric device_ms --devices "$DEVICES" \
        --perf-test "$PERF" -k in0 --budget-usd 1000000000 --max-iter 1000 2>&1 || {
        echo "PASS $pass: before_loop FAILED -> stopping"; break; }
    python -m agent.loop runs 2>&1 || echo "  [loop returned non-zero for pass $pass]"

    # count gains KEPT this pass (the just-finished run dir is the newest)
    LATEST=$(ls -td runs/2026-* 2>/dev/null | head -1)
    # grep -c prints "0" AND exits 1 on no-match; a "|| echo 0" would append a 2nd "0"
    # and break the integer test below (the bug that made it skip convergence). Capture
    # stdout only and default empties to 0.
    KEEPS=$(grep -c '"result": "keep"' "$LATEST/ledger.jsonl" 2>/dev/null); KEEPS=${KEEPS:-0}
    FINAL=$(python3 -c "import json;rows=[json.loads(l) for l in open('$LATEST/ledger.jsonl')];ap=[r['after'] for r in rows if r.get('after') is not None];print(ap[-1] if ap else 'n/a')" 2>/dev/null)
    echo "----- PASS $pass RESULT: kept=$KEEPS gains, device_ms=$FINAL (run $LATEST) -----"
    if [ "$KEEPS" -eq 0 ]; then
        echo "================= CONVERGED after pass $pass (0 new gains) ================="
        break
    fi
done
echo "================= $(date) : ALL PASSES DONE (ran $pass) ================="
