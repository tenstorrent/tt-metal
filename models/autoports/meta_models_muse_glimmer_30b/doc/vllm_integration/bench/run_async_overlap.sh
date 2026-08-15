#!/usr/bin/env bash
# The --async-scheduling overlap validation $vllm-integration requires.
#
# With supports_async_decode=True, sample_on_device_mode=all, a captured decode
# trace and reset_batch=False, vLLM may build and submit decode step N+1 before
# sampled token N has reached host scheduler state. The adapter's answer is to
# read nothing from host on such a step. This run is the end-to-end check that
# the answer holds: if it did not, the output would show doubled subwords or
# repeated control tokens.
#
# Artifacts go to a SEPARATE directory so they cannot overwrite the stage's
# committed non-overlap evidence.
set -u
REPO=/home/ttuser/dev/muse-glimmer/tt-metal
MODEL_DIR=models/autoports/meta_models_muse_glimmer_30b
DOC=$REPO/$MODEL_DIR/doc/vllm_integration
LOGS=$DOC/logs
OUT=$DOC/async_overlap
URL=http://localhost:8000
cd "$REPO"; mkdir -p "$LOGS" "$OUT"

echo "=== launching server WITH --async-scheduling $(date -u +%H:%M:%S) ==="
OUT_DIR="$OUT" EXTRA_SERVER_ARGS=--async-scheduling \
  bash "$DOC/bench/serve.sh" hold > "$LOGS/async_serve_hold.log" 2>&1 &
SERVE_PID=$!

ready=0
for i in $(seq 1 240); do
  curl -sf "$URL/health" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 $SERVE_PID 2>/dev/null || { echo "LAUNCHER EXITED"; break; }
  grep -qE "EngineCore encountered a fatal error|EngineDeadError|EngineCore failed to start" \
      "$OUT/server.log" 2>/dev/null && { echo "FATAL MARKER"; break; }
  sleep 10
done
[ "$ready" = 1 ] || { echo "SERVER_NOT_READY"; tail -40 "$OUT/server.log" 2>/dev/null; kill -TERM $SERVE_PID 2>/dev/null; exit 1; }
echo "=== server ready $(date -u +%H:%M:%S) ==="

# The capability must have been ACCEPTED, not silently refused.
if grep -q "Disabling async scheduling" "$OUT/server.log"; then
  echo "ASYNC_REFUSED: the plugin disabled async scheduling"
else
  echo "ASYNC_ACCEPTED: no 'Disabling async scheduling' in the server log"
fi
grep -oE "async_scheduling=[A-Za-z]+" "$OUT/server.log" | sort -u | head -3

echo "=== qualitative under overlap $(date -u +%H:%M:%S) ==="
python "$DOC/bench/qualitative_vllm.py" --server-url "$URL" --out-dir "$OUT/qualitative" \
    > "$LOGS/async_qualitative.log" 2>&1
echo "STEP async_qualitative rc=$?"

echo "=== shutting down $(date -u +%H:%M:%S) ==="
kill -TERM $SERVE_PID 2>/dev/null
for i in $(seq 1 30); do kill -0 $SERVE_PID 2>/dev/null || break; sleep 2; done
kill -9 $SERVE_PID 2>/dev/null; sleep 5

echo "=== degenerate check over the overlap artifacts $(date -u +%H:%M:%S) ==="
python models/common/readiness_check/check_degenerate_output.py \
    --model-dir "$OUT" --missing-artifacts critical --scope vllm \
    > "$LOGS/async_degenerate.log" 2>&1
echo "STEP async_degenerate rc=$?"
tail -3 "$LOGS/async_degenerate.log"
echo "ASYNC_OVERLAP_DONE $(date -u +%H:%M:%S)"
