#!/usr/bin/env bash
# Drive one tt-inference-server workflow as a CLIENT of the already-running
# autoport vLLM server (doc/tti_release/bench/serve_release.sh).
#
#   run_tti.sh smoke      # tiny benchmark, trace capture disabled
#   run_tti.sh evalsmoke  # 1%-sampled evals: proves the lm-eval chat path works
#                         # before committing an hour to the unrestricted suite
#   run_tti.sh release    # the full release workflow (evals+benchmarks+spec_tests)
#
# There is deliberately no reports-only mode: this tt-inference-server has no
# workflows/run_reports.py, the report is built in-process from the Blocks the
# run itself accumulates, and there is no path that re-aggregates an old
# workflow_logs tree. Regenerating a release report means rerunning the release.
#
# No --docker-server, no --local-server: TTI never launches a server here.
# Nothing in this script prints or persists a token.
set -uo pipefail

REPO=/home/ttuser/dev/muse-glimmer/tt-metal
MODEL_DIR=models/autoports/meta_models_muse_glimmer_30b
DOC=$REPO/$MODEL_DIR/doc/tti_release
WORK_ROOT=/home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b
TTI=$WORK_ROOT/tt-inference-server
SPECS=$WORK_ROOT/specs

TTI_MODEL=Muse-Glimmer-30B
TTI_DEVICE=p300x2
SERVICE_PORT=${SERVICE_PORT:-8000}

mode=${1:?"usage: run_tti.sh {smoke|evalsmoke|release}"}

# The server must already be up: TTI is a pure client here.
if ! curl -fsS "http://127.0.0.1:$SERVICE_PORT/health" >/dev/null 2>&1; then
  echo "autoport vLLM server is not answering on port $SERVICE_PORT" >&2
  exit 2
fi

case "$mode" in
  smoke)     workflow=benchmarks; spec_name=smoke;     extra=(--limit-samples-mode smoke-test --disable-trace-capture); disable_trace=--disable-trace-capture ;;
  evalsmoke) workflow=evals;      spec_name=evalsmoke; extra=(--limit-samples-mode smoke-test); disable_trace= ;;
  release)   workflow=release;    spec_name=release;   extra=() ; disable_trace= ;;
  *) echo "usage: run_tti.sh {smoke|evalsmoke|release}" >&2; exit 2 ;;
esac

SPEC=$SPECS/muse_glimmer_30b_autoport_${spec_name}.json
python3 "$DOC/bench/export_runtime_spec.py" \
  --tti-root "$TTI" --workflow "$workflow" --service-port "$SERVICE_PORT" \
  --device "$TTI_DEVICE" ${disable_trace:+--disable-trace-capture} \
  --out "$SPEC" || exit 1

export HF_HOME=${HF_HOME:-/home/ttuser/.cache/huggingface}
export HOST_HF_HOME=$HF_HOME
export MODEL_SOURCE=huggingface
export CACHE_ROOT=$WORK_ROOT/cache_root
export PERSISTENT_VOLUME_ROOT=$WORK_ROOT/persistent_volume
export SERVICE_PORT
mkdir -p "$CACHE_ROOT" "$PERSISTENT_VOLUME_ROOT"
# Read from the on-disk credential; never echoed.
if [ -f "$HF_HOME/token" ]; then
  HF_TOKEN=$(cat "$HF_HOME/token"); export HF_TOKEN
fi

LOG=$DOC/logs/tti_${mode}_$(date -u +%Y%m%dT%H%M%SZ).log
echo "log: $LOG"
cd "$TTI" || exit 1

set -x
python3 run.py \
  --model "$TTI_MODEL" \
  --runtime-model-spec-json "$SPEC" \
  --tt-device "$TTI_DEVICE" \
  --workflow "$workflow" \
  --service-port "$SERVICE_PORT" \
  --no-auth \
  --skip-system-sw-validation \
  "${extra[@]}" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set +x

echo "=== run_tti.sh $mode exit=$status ==="
echo "log: $LOG"
exit "$status"
