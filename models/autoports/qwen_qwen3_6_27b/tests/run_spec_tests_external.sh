#!/usr/bin/env bash
# Spec tests / API conformance against an ALREADY-RUNNING server.
#
# `--workflow spec_tests --local-server` cannot work for this model: it launches the server
# and runs the conformance suite ~2 seconds later, against a server that needs ~15 minutes to
# load. Measured: server log created 15:37:11, tests ran 15:37:13, every case failing with
# ConnectionRefusedError on 127.0.0.1:8000. base_test.py:446 defines wait_for_server_ready but
# vllm_param_conformance_test.py never calls it, and the workflow only *warns* when its
# /tt-liveness probe fails instead of waiting.
#
# So: bring the server up ourselves with the release serving flags, wait for health, then run
# the workflow in external-server mode (--server-url, no --local-server) -- the same shape the
# earlier release invocation used.
set -uo pipefail
TTSMI=$HOME/tt-metal/python_env/bin/tt-smi
OUT=$HOME/_qwen_spec_external
mkdir -p "$OUT"
exec >"$OUT/run.log" 2>&1
echo "=== spec-tests (external server) start $(date -u +%FT%TZ)"

echo "--- reset + settle"
sleep 20
ok=0
for attempt in 1 2; do
  "$TTSMI" -r >/dev/null 2>&1 && echo "    tt-smi -r ok" || echo "    tt-smi -r FAILED"
  if bash /tmp/mesh_smoke.sh >/dev/null 2>&1; then echo "    mesh smoke ok"; ok=1; break; fi
  echo "    mesh smoke FAILED (attempt $attempt)"
done
[ $ok -eq 1 ] || { echo "ABORT: mesh will not come up"; exit 1; }

# Release serving flags, matching the spec (trace region lowered, as everywhere here).
cd "$HOME/tt-metal" || exit 2
source python_env/bin/activate
export PYTHONPATH="$HOME/vllm:$HOME/tt-metal:${PYTHONPATH:-}"
export VLLM_PLUGINS=tt,tt_model_registry
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-27B \
  --block_size 64 --max_model_len 262144 --max_num_batched_tokens 262144 \
  --max_num_seqs 32 --seed 9472 --port 8000 \
  --reasoning_parser qwen3 --tool_call_parser qwen3_coder --enable-auto-tool-choice \
  --additional-config '{"tt":{"sample_on_device_mode":"decode_only","trace_region_size":200000000,"fabric_config":"FABRIC_1D","l1_small_size":24576}}' \
  > "$OUT/server.log" 2>&1 &
SRV=$!
echo "--- server pid $SRV"
for i in $(seq 1 160); do
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; grep -iE "TT_FATAL|RuntimeError|Error" "$OUT/server.log" | grep -viE "leaked|nanobind" | tail -4; exit 1; }
  sleep 15
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "SERVER NOT READY"; kill $SRV; exit 1; }
echo "--- server healthy $(date -u +%FT%TZ)"
curl -sf -m 5 http://127.0.0.1:8000/v1/models 2>/dev/null | head -c 200; echo

cd "$HOME/tt-inference-server" || exit 2
echo "--- tti at: $(git rev-parse --short HEAD)"
timeout 5400 python3 run.py \
  --model Qwen3.6-27B \
  --workflow spec_tests \
  --tt-device p300x2 \
  --server-url http://127.0.0.1 \
  --service-port 8000 \
  --no-auth \
  --skip-system-sw-validation \
  --limit-samples-mode ci-nightly \
  --ci-mode \
  > "$OUT/runner.log" 2>&1
echo "RUNNER_EXIT=$?"

echo "--- conformance case results:"
grep -oE "test_[a-z0-9_]+.{0,20}(PASSED|FAILED)|(PASSED|FAILED).{0,10}test_[a-z0-9_]+" "$OUT/runner.log" 2>/dev/null | sort -u | head -30 | sed 's/^/    /'
echo "--- summary lines:"
grep -iE "Total Tests|Passed|Failed|Success Rate|Acceptance status|Spec Tests:" "$OUT/runner.log" 2>/dev/null | tail -12 | sed 's/^/    /'

kill $SRV 2>/dev/null
for j in $(seq 1 24); do kill -0 $SRV 2>/dev/null || break; sleep 5; done
kill -9 $SRV 2>/dev/null
echo "=== spec-tests (external server) done $(date -u +%FT%TZ)"
