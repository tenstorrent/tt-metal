#!/usr/bin/env bash
# The whole serving-evidence sweep against ONE server, so the 52-layer model is
# loaded once (~3 min) instead of once per gate.
#
#   1. hold a server open  (run_vllm_server --stages serve)
#   2. sampling   -- the canonical TT plugin pytest suite, --sampling-profile full
#   3. qualitative-- the runner's own raw-completion arm (labelled continuation
#                    stress coverage; this checkpoint has a chat template)
#   4. qualitative-- the prompt-correct chat arm on the pinned token ids, which is
#                    the arm the verdict is read from ($qualitative-check)
#   5. determinism-- run-to-run, cross-batch-position, standalone baseline, and
#                    nine non-aligned prompt lengths through the OpenAI API
#   6. benchmark  -- primary single-user 128/128/1 plus the CI serving-burst
#                    100/100/32 profile
#   7. shut the server down and audit for leftovers holding the chips
#
# Every stage's exit status is echoed and the sweep continues, so one failing gate
# does not cost the others their evidence.
set -u

REPO=/home/ttuser/dev/muse-glimmer/tt-metal
MODEL_DIR=models/autoports/meta_models_muse_glimmer_30b
DOC=$REPO/$MODEL_DIR/doc/vllm_integration
LOGS=$DOC/logs
HF_MODEL=meta-models/Muse-Glimmer-30B
PORT=${PORT:-8000}
URL=http://localhost:$PORT
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}

cd "$REPO"
mkdir -p "$LOGS"
export MAX_NUM_SEQS PORT

echo "=== launching server $(date -u +%H:%M:%S) ==="
bash "$DOC/bench/serve.sh" hold > "$LOGS/serve_hold.log" 2>&1 &
SERVE_PID=$!

# The runner writes readiness_vllm/server.log; wait on /health, and fail fast if
# the launcher dies or the engine writes a fatal marker.
ready=0
for i in $(seq 1 240); do
  if curl -sf "$URL/health" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 $SERVE_PID 2>/dev/null; then echo "LAUNCHER EXITED"; break; fi
  if grep -qE "EngineCore encountered a fatal error|EngineDeadError|EngineCore failed to start" \
        "$REPO/$MODEL_DIR/readiness_vllm/server.log" 2>/dev/null; then echo "FATAL MARKER"; break; fi
  sleep 10
done
if [ "$ready" != 1 ]; then
  echo "SERVER_NOT_READY"
  tail -60 "$REPO/$MODEL_DIR/readiness_vllm/server.log" 2>/dev/null
  kill -TERM $SERVE_PID 2>/dev/null
  exit 1
fi
echo "=== server ready $(date -u +%H:%M:%S) ==="

step () {
  name=$1; shift
  echo "=== $name $(date -u +%H:%M:%S) ==="
  "$@" > "$LOGS/$name.log" 2>&1
  echo "STEP $name rc=$?"
}

step sampling_full bash "$DOC/bench/serve.sh" checks sampling
step qualitative_runner bash "$DOC/bench/serve.sh" checks qualitative
step qualitative_chat python "$DOC/bench/qualitative_vllm.py" --server-url "$URL"
step determinism python "$DOC/bench/determinism_vllm.py" --server-url "$URL"
step benchmark bash "$DOC/bench/serve.sh" checks benchmark

echo "=== shutting down $(date -u +%H:%M:%S) ==="
kill -TERM $SERVE_PID 2>/dev/null
for i in $(seq 1 30); do kill -0 $SERVE_PID 2>/dev/null || break; sleep 2; done
kill -9 $SERVE_PID 2>/dev/null
sleep 5

step audit python "$DOC/bench/audit_serving.py" \
    --server-log "$REPO/$MODEL_DIR/readiness_vllm/server.log" \
    --out "$DOC/serving_audit.json"

echo "SERVING_EVIDENCE_DONE $(date -u +%H:%M:%S)"
