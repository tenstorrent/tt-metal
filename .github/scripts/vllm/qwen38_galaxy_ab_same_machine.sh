#!/bin/bash
# Serve two settings one after the other on one machine, in the order A B A B, and run the
# same benchmark and greedy client against each server. The repeat of each setting gives
# the run-to-run spread that the A/B difference has to exceed.
#
# Inputs (environment):
#   AB_MODEL           hub id the server is served under
#   HF_MODEL_LOCAL     local snapshot dir of that model
#   AB_COMMON_ENV      KEY=VALUE pairs for both settings
#   AB_COMMON_ARGS     vLLM server args for both settings
#   AB_ENV_A/B         KEY=VALUE pairs of setting A/B
#   AB_ARGS_A/B        vLLM server args of setting A/B
#   AB_TT_CONFIG       JSON for --additional-config '{"tt": ...}'
#   AB_CHAT_KWARGS     JSON for --default-chat-template-kwargs
#   AB_SERVER_TIMEOUT  minutes to wait for /health
set -u
OUT=output
mkdir -p "$OUT"
FAILED=0

wait_ready() {
  local pid=$1 log=$2 limit=$(( AB_SERVER_TIMEOUT * 60 )) t=0
  while [ $t -lt $limit ]; do
    curl -sf http://localhost:8000/health >/dev/null && { echo "ready after ${t}s"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "server exited after ${t}s"; return 1; }
    if grep -q -F -m1 -e "EngineCore failed to start" -e "EngineDeadError" -e "Engine core initialization failed" -e "Failed core proc" "$log" 2>/dev/null; then
      echo "fatal marker in server log after ${t}s"; return 1
    fi
    sleep 20; t=$(( t + 20 ))
  done
  echo "not ready after ${limit}s"; return 1
}

stop_server() {
  local pid=$1 t=0
  kill -TERM -- "-$pid" 2>/dev/null
  while kill -0 -- "-$pid" 2>/dev/null && [ $t -lt 180 ]; do sleep 5; t=$(( t + 5 )); done
  kill -KILL -- "-$pid" 2>/dev/null
  # Let the device handles close before the next server opens the mesh.
  sleep 20
}

run_arm() {
  local tag=$1 envs=$2 args=$3
  local log="$OUT/ab_${tag}_server.log"
  echo "=== $tag: start $(date -u +%H:%M:%S)"
  # shellcheck disable=SC2086
  env $AB_COMMON_ENV $envs MESH_DEVICE=BH-Galaxy HF_MODEL="$HF_MODEL_LOCAL" VLLM_RPC_TIMEOUT=300000 \
    setsid python3 "$PLUGIN_DIR/examples/server_example_tt.py" \
      --model "$HF_MODEL_LOCAL" --served-model-name "$AB_MODEL" \
      $AB_COMMON_ARGS $args \
      --default-chat-template-kwargs "$AB_CHAT_KWARGS" \
      --additional-config "{\"tt\": $AB_TT_CONFIG}" > "$log" 2>&1 &
  local pid=$!
  local t0=$SECONDS
  if ! wait_ready "$pid" "$log"; then
    echo "=== $tag: server did not start"; tail -60 "$log"; FAILED=1; stop_server "$pid"; return
  fi
  echo "$tag startup_s $(( SECONDS - t0 ))" >> "$OUT/ab_startup.txt"
  for c in 32 8; do
    # The spec-decode counters around each benchmark give its draft acceptance.
    curl -sf http://localhost:8000/metrics > "$OUT/ab_${tag}_metrics_pre_c${c}.txt" || true
    vllm bench serve --backend vllm --model "$AB_MODEL" --tokenizer "$HF_MODEL_LOCAL" \
      --dataset-name random --random-input-len 2000 --random-output-len 200 \
      --num-prompts $(( c * 4 )) --max-concurrency "$c" --ignore-eos \
      --percentile-metrics ttft,tpot,itl,e2el --save-result \
      --result-filename "$OUT/ab_${tag}_bench_c${c}.json" > "$OUT/ab_${tag}_bench_c${c}.log" 2>&1 \
      || { echo "=== $tag: benchmark c$c failed"; tail -20 "$OUT/ab_${tag}_bench_c${c}.log"; FAILED=1; }
    curl -sf http://localhost:8000/metrics > "$OUT/ab_${tag}_metrics_post_c${c}.txt" || true
  done
  python3 .github/scripts/vllm/qwen38_galaxy_ab_client.py --model "$AB_MODEL" \
    --out "$OUT/ab_${tag}_greedy.json" || { echo "=== $tag: greedy client reported failures"; FAILED=1; }
  stop_server "$pid"
  echo "=== $tag: done $(date -u +%H:%M:%S)"
}

run_arm A1 "$AB_ENV_A" "$AB_ARGS_A"
run_arm B1 "$AB_ENV_B" "$AB_ARGS_B"
run_arm A2 "$AB_ENV_A" "$AB_ARGS_A"
run_arm B2 "$AB_ENV_B" "$AB_ARGS_B"

python3 .github/scripts/vllm/qwen38_galaxy_ab_summary.py "$OUT" | tee "$OUT/ab_summary.md"
[ -n "${GITHUB_STEP_SUMMARY:-}" ] && cat "$OUT/ab_summary.md" >> "$GITHUB_STEP_SUMMARY"
exit $FAILED
