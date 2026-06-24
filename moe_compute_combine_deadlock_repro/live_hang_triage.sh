#!/bin/bash
# Run INSIDE the tt-xla-ird container. Reproduces the moe_compute fused-combine hang
# while keeping the host process ALIVE, then runs tt-triage in parallel so it can read
# live Inspector data + stuck-core callstacks. Kills only its own smoke PID at the end
# (no kill-all). Does NOT reset the galaxy.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/.env.sh"

TS=$(date +%Y%m%d_%H%M%S)
SMOKE_LOG="$HERE/logs/livehang_smoke_$TS.txt"
TRIAGE_LOG="$HERE/logs/livehang_triage_$TS.txt"
TRIAGE_CALLSTACK_LOG="$HERE/logs/livehang_triage_callstacks_$TS.txt"

echo "runtime HEAD: $(cd "$TT_METAL_RUNTIME_ROOT" && git rev-parse --short HEAD)"
echo "smoke log:  $SMOKE_LOG"
echo "triage log: $TRIAGE_LOG"
echo "callstack triage log: $TRIAGE_CALLSTACK_LOG"

# 1) Launch the pinpoint smoke in the background (NO timeout -> it parks at the hung
#    post-moe_compute sync and stays alive so triage can attach).
SMOKE_PINPOINT=1 stdbuf -oL -eL python3 moe_compute_smoke.py >"$SMOKE_LOG" 2>&1 &
SMOKE_PID=$!
echo "smoke PID: $SMOKE_PID"

# 2) Wait until the combine has been enqueued (the line right before the hanging sync).
echo "waiting for 'moe_compute ok' (combine enqueued)..."
for i in $(seq 1 120); do
    grep -q "moe_compute ok" "$SMOKE_LOG" && break
    if ! kill -0 "$SMOKE_PID" 2>/dev/null; then echo "smoke exited early; see $SMOKE_LOG"; exit 1; fi
    sleep 2
done

# 3) Confirm it is actually HUNG: combine enqueued but the post-sync line never appears.
echo "combine enqueued; confirming the post-moe_compute sync hangs (60s dwell)..."
sleep 60
if grep -q "SMOKE SYNC after moe_compute combine OK" "$SMOKE_LOG"; then
    echo "NOT HUNG: sync returned -> this run drained. Tail:"; tail -5 "$SMOKE_LOG"
    wait "$SMOKE_PID"; exit 0
fi
echo ">>> CONFIRMED HUNG (sync did not return, PID $SMOKE_PID still alive). Running tt-triage in parallel."

# 4) tt-triage against the LIVE hung runtime. Write outputs to files.
echo "running tt-triage default -> $TRIAGE_LOG"
timeout 300 python3 "$TT_METAL_RUNTIME_ROOT/tools/tt-triage.py" --skip-version-check >"$TRIAGE_LOG" 2>&1
echo "running tt-triage dump_callstacks -> $TRIAGE_CALLSTACK_LOG"
timeout 300 python3 "$TT_METAL_RUNTIME_ROOT/tools/tt-triage.py" --skip-version-check --run=dump_callstacks --all-cores -vv >"$TRIAGE_CALLSTACK_LOG" 2>&1

# 5) Clean up ONLY our smoke process (no kill-all). Reset is left to the operator.
echo ">>> triage captured. Killing smoke PID $SMOKE_PID (SIGTERM, then SIGKILL)."
kill -TERM "$SMOKE_PID" 2>/dev/null; sleep 5; kill -KILL "$SMOKE_PID" 2>/dev/null
echo ">>> done. NOTE: galaxy still needs a reset before the next run."
