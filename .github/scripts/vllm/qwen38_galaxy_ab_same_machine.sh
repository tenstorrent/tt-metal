#!/bin/bash
# Serve several settings one after the other on one machine and run the same benchmark (and,
# per arm, the greedy client) against each server. A setting run twice gives the run-to-run
# spread that a difference between settings has to exceed.
#
# Inputs (environment):
#   AB_MODEL           hub id the server is served under
#   HF_MODEL_LOCAL     local snapshot dir of that model
#   AB_COMMON_ENV      KEY=VALUE pairs for every arm
#   AB_COMMON_ARGS     vLLM server args for every arm
#   AB_ARMS            arms in run order, separated by ';', each TAG|ENV|ARGS|GREEDY
#                      (GREEDY 1 runs the greedy client); empty: A1 B1 A2 B2 from the four below
#   AB_ENV_A/B         KEY=VALUE pairs of setting A/B
#   AB_ARGS_A/B        vLLM server args of setting A/B
#   AB_BENCH           benchmark runs, separated by spaces, each CONCURRENCY:PROMPTS
#   AB_TT_CONFIG       JSON for --additional-config '{"tt": ...}'
#   AB_CHAT_KWARGS     JSON for --default-chat-template-kwargs
#   AB_SERVER_TIMEOUT  minutes to wait for /health
#   AB_MESH            MESH_DEVICE of the server (default BH-Galaxy)
set -u
OUT=output
mkdir -p "$OUT"
FAILED=0
: > "$OUT/ab_arms.txt"

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

# Processes of the server's session that are not zombies. The server runs under setsid, so its
# pid is the session id; a zombie that is reaped later must not hold up the next arm.
session_alive() {
  ps -o pid=,stat=,etime=,args= -s "$1" 2>/dev/null | awk '$2 !~ /^Z/'
}

stop_server() {
  local pid=$1 t=0
  kill -TERM -- "-$pid" 2>/dev/null
  if ps -o pid= -s "$$" >/dev/null 2>&1; then
    while [ -n "$(session_alive "$pid")" ] && [ $t -lt 180 ]; do sleep 5; t=$(( t + 5 )); done
    if [ -n "$(session_alive "$pid")" ]; then
      echo "still running ${t}s after SIGTERM:"; session_alive "$pid" | cut -c1-200
    else
      echo "server session exited ${t}s after SIGTERM"
    fi
  else
    while kill -0 -- "-$pid" 2>/dev/null && [ $t -lt 180 ]; do sleep 5; t=$(( t + 5 )); done
  fi
  kill -KILL -- "-$pid" 2>/dev/null
  # Let the device handles close before the next server opens the mesh.
  sleep 20
}

run_arm() {
  local tag=$1 envs=$2 args=$3 greedy=$4
  local log="$OUT/ab_${tag}_server.log"
  echo "$tag" >> "$OUT/ab_arms.txt"
  echo "=== $tag: start $(date -u +%H:%M:%S) env [$envs] args [$args]"
  # shellcheck disable=SC2086
  env $AB_COMMON_ENV $envs MESH_DEVICE="${AB_MESH:-BH-Galaxy}" HF_MODEL="$HF_MODEL_LOCAL" VLLM_RPC_TIMEOUT=300000 \
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
  for run in ${AB_BENCH:-32:128 8:32}; do
    local c=${run%%:*} n=${run##*:}
    # The spec-decode counters around each benchmark give its draft acceptance.
    curl -sf http://localhost:8000/metrics > "$OUT/ab_${tag}_metrics_pre_c${c}.txt" || true
    vllm bench serve --backend vllm --model "$AB_MODEL" --tokenizer "$HF_MODEL_LOCAL" \
      --dataset-name random --random-input-len 2000 --random-output-len 200 \
      --num-prompts "$n" --max-concurrency "$c" --ignore-eos \
      --percentile-metrics ttft,tpot,itl,e2el --save-result \
      --result-filename "$OUT/ab_${tag}_bench_c${c}.json" > "$OUT/ab_${tag}_bench_c${c}.log" 2>&1 \
      || { echo "=== $tag: benchmark c$c failed"; tail -20 "$OUT/ab_${tag}_bench_c${c}.log"; FAILED=1; }
    curl -sf http://localhost:8000/metrics > "$OUT/ab_${tag}_metrics_post_c${c}.txt" || true
  done
  if [ "$greedy" = "1" ]; then
    python3 .github/scripts/vllm/qwen38_galaxy_ab_client.py --model "$AB_MODEL" \
      --out "$OUT/ab_${tag}_greedy.json" || { echo "=== $tag: greedy client reported failures"; FAILED=1; }
  fi
  stop_server "$pid"
  echo "=== $tag: done $(date -u +%H:%M:%S)"
}

if [ -n "${AB_ARMS:-}" ]; then
  IFS=';' read -r -a ARMS <<< "$AB_ARMS"
  for arm in "${ARMS[@]}"; do
    IFS='|' read -r tag envs args greedy <<< "$arm"
    run_arm "$(echo "$tag" | xargs)" "$envs" "$args" "$(echo "${greedy:-1}" | xargs)"
  done
else
  run_arm A1 "$AB_ENV_A" "$AB_ARGS_A" 1
  run_arm B1 "$AB_ENV_B" "$AB_ARGS_B" 1
  run_arm A2 "$AB_ENV_A" "$AB_ARGS_A" 1
  run_arm B2 "$AB_ENV_B" "$AB_ARGS_B" 1
fi

python3 .github/scripts/vllm/qwen38_galaxy_ab_summary.py "$OUT" | tee "$OUT/ab_summary.md"
[ -n "${GITHUB_STEP_SUMMARY:-}" ] && cat "$OUT/ab_summary.md" >> "$GITHUB_STEP_SUMMARY"
exit $FAILED
