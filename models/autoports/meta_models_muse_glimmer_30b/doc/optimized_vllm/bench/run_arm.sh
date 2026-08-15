#!/usr/bin/env bash
# One before/after arm of the optimized-vLLM sweep, against ONE server so the
# 52-layer model is loaded once for the whole arm.
#
#   run_arm.sh <arm> <step,step,...>
#
# arm    -- "before" or "after" (or any name); artifacts land in
#           doc/optimized_vllm/<arm>/ and logs in doc/optimized_vllm/logs/<arm>_*.log
# steps  -- comma-separated, from:
#           bench<N>               the runner's benchmark stage (primary single-user
#                                  128/128/1 AND the CI serving-burst 100/100/32) into
#                                  <arm>/run<N>/.  Repeat it: the first requests after a
#                                  server start are measurably slower than the fourth
#                                  (the before arm's TTFT runs 89.5 -> 80.5 -> 75.5 ms
#                                  over three back-to-back invocations), so a single
#                                  sample per arm would compare two different points on
#                                  a warm-up curve.  Every arm runs the same count in the
#                                  same position, before any other traffic.
#           sampling               canonical TT plugin pytest suite, --sampling-profile full
#           qualitative            the runner's raw-completion arm
#           qualchat               the prompt-correct chat arm on the pinned token ids
#           determinism            run-to-run, cross-batch-position, non-aligned lengths
#           probe                  adapter stale-input / async-split probe against the
#                                  live serving build's own generator (in-process, separate)
#
# Every step's exit status is echoed and the sweep continues, so one failing gate
# does not cost the others their evidence.
#
# MUSE_GLIMMER_VLLM_PREFILL_TRACE is exported from the caller's environment into
# the server process by run_vllm_server (it inherits the environment), which is
# what selects the arm.
set -u

REPO=/home/ttuser/dev/muse-glimmer/tt-metal
MODEL_DIR=models/autoports/meta_models_muse_glimmer_30b
DOC=$REPO/$MODEL_DIR/doc/optimized_vllm
VDOC=$REPO/$MODEL_DIR/doc/vllm_integration
LOGS=$DOC/logs
PORT=${PORT:-8000}
URL=http://localhost:$PORT
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}

ARM=${1:?"usage: run_arm.sh <arm> <steps>"}
STEPS=${2:?"usage: run_arm.sh <arm> <steps>"}
ARM_DIR=$DOC/$ARM

cd "$REPO"
mkdir -p "$LOGS" "$ARM_DIR"
export MAX_NUM_SEQS PORT

echo "=== arm=$ARM steps=$STEPS prefill_trace=${MUSE_GLIMMER_VLLM_PREFILL_TRACE:-unset} ==="
echo "=== launching server $(date -u +%H:%M:%S) ==="
OUT_DIR=$ARM_DIR/server bash "$DOC/bench/serve.sh" hold > "$LOGS/${ARM}_serve_hold.log" 2>&1 &
SERVE_PID=$!
SERVER_LOG=$ARM_DIR/server/server.log

launch_t0=$(date +%s)
ready=0
for i in $(seq 1 240); do
  if curl -sf "$URL/health" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 $SERVE_PID 2>/dev/null; then echo "LAUNCHER EXITED"; break; fi
  if grep -qE "EngineCore encountered a fatal error|EngineDeadError|EngineCore failed to start" \
        "$SERVER_LOG" 2>/dev/null; then echo "FATAL MARKER"; break; fi
  sleep 5
done
if [ "$ready" != 1 ]; then
  echo "SERVER_NOT_READY"
  tail -80 "$SERVER_LOG" 2>/dev/null
  kill -TERM $SERVE_PID 2>/dev/null
  exit 1
fi
echo "=== server ready $(date -u +%H:%M:%S) after $(( $(date +%s) - launch_t0 ))s ==="

step () {
  name=$1; shift
  echo "=== $name $(date -u +%H:%M:%S) ==="
  "$@" > "$LOGS/${ARM}_$name.log" 2>&1
  echo "STEP $name rc=$?"
}

for s in ${STEPS//,/ }; do
  case "$s" in
    bench*)
      n=${s#bench}
      mkdir -p "$ARM_DIR/run$n"
      step "$s" env OUT_DIR="$ARM_DIR/run$n" bash "$DOC/bench/serve.sh" checks benchmark
      # Where the measured window ends, in bytes of server log.  The audit needs it to
      # separate "a degraded path was taken while the benchmark was running" -- which
      # would invalidate the numbers -- from one taken by a later check stage, which
      # does not.  Last bench step wins.
      wc -c < "$SERVER_LOG" > "$ARM_DIR/bench_window_end_bytes.txt" 2>/dev/null || true
      ;;
    sampling)
      step sampling env OUT_DIR="$ARM_DIR" bash "$DOC/bench/serve.sh" checks sampling
      ;;
    samplingrep*)
      # Repeat the canonical suite against the SAME server, into its own directory.
      # The suite has a known nondeterministic class on this port (seeded
      # reproducibility at batch > 1), so a single run cannot tell a regression from
      # that class re-rolling; repeating it against one server can.
      n=${s#samplingrep}
      mkdir -p "$ARM_DIR/sampling$n"
      step "$s" env OUT_DIR="$ARM_DIR/sampling$n" bash "$DOC/bench/serve.sh" checks sampling
      ;;
    qualitative)
      step qualitative env OUT_DIR="$ARM_DIR" bash "$DOC/bench/serve.sh" checks qualitative
      ;;
    qualitativerep*)
      # The runner's raw-completion arm, repeatable within one server.  Twelve requests
      # per round with real prompts and real sampling; the cheapest way to put sustained
      # traffic through a traced-prefill server and see whether it stays healthy.
      n=${s#qualitativerep}
      mkdir -p "$ARM_DIR/runner_qual$n"
      step "$s" env OUT_DIR="$ARM_DIR/runner_qual$n" bash "$DOC/bench/serve.sh" checks qualitative
      ;;
    qualchatrep*)
      # The prompt-correct chat arm, repeatable within one server so "was it broken
      # from the first request or did something earlier in the sweep break it" is
      # answerable without a second four-minute model load.
      n=${s#qualchatrep}
      mkdir -p "$ARM_DIR/qualitative$n"
      step "$s" python "$VDOC/bench/qualitative_vllm.py" --server-url "$URL" \
          --out-dir "$ARM_DIR/qualitative$n"
      ;;
    qualchat)
      mkdir -p "$ARM_DIR/qualitative"
      step qualchat python "$VDOC/bench/qualitative_vllm.py" --server-url "$URL" \
          --out-dir "$ARM_DIR/qualitative"
      step qualcompare python "$VDOC/bench/qualitative_vllm.py" --compare \
          --out-dir "$ARM_DIR/qualitative"
      ;;
    determinism)
      step determinism python "$VDOC/bench/determinism_vllm.py" --server-url "$URL" \
          --out "$ARM_DIR/determinism_vllm.json"
      ;;
    *)
      echo "STEP $s rc=SKIPPED_UNKNOWN"
      ;;
  esac
done

echo "=== shutting down $(date -u +%H:%M:%S) ==="
# Where teardown starts, in bytes of server log.  vLLM's API server logs an
# EngineDeadError traceback when the EngineCore it was talking to goes away, which is
# what SIGTERM does; without this offset the audit reads that as a serving-path
# degradation instead of the shutdown it is.
wc -c < "$SERVER_LOG" > "$ARM_DIR/shutdown_window_start_bytes.txt" 2>/dev/null || true
kill -TERM $SERVE_PID 2>/dev/null
for i in $(seq 1 30); do kill -0 $SERVE_PID 2>/dev/null || break; sleep 2; done
kill -9 $SERVE_PID 2>/dev/null
sleep 5

bench_end=$(cat "$ARM_DIR/bench_window_end_bytes.txt" 2>/dev/null || echo "")
shutdown_start=$(cat "$ARM_DIR/shutdown_window_start_bytes.txt" 2>/dev/null || echo "")
step audit python "$VDOC/bench/audit_serving.py" \
    --server-log "$SERVER_LOG" \
    ${bench_end:+--benchmark-window-end-bytes "$bench_end"} \
    ${shutdown_start:+--shutdown-window-start-bytes "$shutdown_start"} \
    --out "$ARM_DIR/serving_audit.json"

# The server log is 80+ MB; keep a committable excerpt and drop the raw file.
if [ -f "$SERVER_LOG" ]; then
  { head -400 "$SERVER_LOG"; echo "... [truncated] ..."; \
    grep -nE "prefill trace|prefill warmup|captured|trace|DEGRADED|Sampling|sample_on_device|max_model_len|max_num_seqs|blocks" \
      "$SERVER_LOG" | head -400; echo "... [tail] ..."; tail -200 "$SERVER_LOG"; } \
      > "$ARM_DIR/server_excerpt.log" 2>/dev/null
  du -h "$SERVER_LOG" > "$ARM_DIR/server_log_size.txt"
fi

echo "ARM_${ARM}_DONE $(date -u +%H:%M:%S)"
