#!/bin/bash
# One scored serving run of Llama-3.3-70B at TP32 on an 8x4 WH Galaxy (4 DP lanes x 8 users).
# Same server flags, benchmark, scoring and correctness gates as the reference runs in README.md.
#
# usage: [MMLU=full|limit|0] [LOGDIR=/dev/shm/llama70b_tp32] bash run_serving.sh <tag>
# Run inside the container that holds ~/teja/{tt-metal,vllm-tt-plugin,runs}, with tt-metal on this branch.
# Exit code: 0 = every scored row passes and every gate passes. Bits: 1 SLA row, 8 MMLU, 16 needle, 32 prefix needle;
# 3 reset failed, 4 server exited at start, 5 not ready after 2 h, 6 smoke failed, 7 stale env var set.
set -u
TAG=${1:?usage: run_serving.sh <tag>}
HERE=$(cd "$(dirname "$0")" && pwd)
TEJA=${TEJA:-$HOME/teja}
R=$TEJA/runs
MMLU=${MMLU:-full}
LOGDIR=${LOGDIR:-/dev/shm/llama70b_tp32}   # server log and bench output on tmpfs: disk writes caused latency tails
S=$R/case_$TAG.log
log() { echo "$(date -u +%F_%T) $*" | tee -a "$S"; }
finish() { log "__case_exit:$1"; exit "$1"; }
mkdir -p "$R/bench" "$LOGDIR"

# 1. No switches: the setup is the config file plus the branches. Any of these left in the shell changes the run.
stale=$(env | cut -d= -f1 | grep -E '^(LLAMA_GALAXY_|TT_PLUGIN_EXP_|TRACE_REGION_SIZE$)' | tr '\n' ' ')
[ -n "$stale" ] && { log "unset these first: $stale"; finish 7; }

TM=$TEJA/tt-metal; PL=$TEJA/vllm-tt-plugin
log "case $TAG; tt-metal $(git -C "$TM" rev-parse --short HEAD) ($(git -C "$TM" rev-parse --abbrev-ref HEAD)) dirty=$(git -C "$TM" status --porcelain -- models | wc -l)"
log "plugin $(git -C "$PL" rev-parse --short HEAD) ($(git -C "$PL" rev-parse --abbrev-ref HEAD)) dirty=$(git -C "$PL" status --porcelain -- src | wc -l)"
log "config $(tr -d ' \n' < "$HERE/serving_tt_config.json")"

cd "$TM" && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_MODEL=${HF_MODEL:-/home/ubuntu/teja/weights/Llama-3.3-70B-Instruct}
export TT_CACHE_PATH=${TT_CACHE_PATH:-/home/ubuntu/teja/cache/llama70b}
export MESH_DEVICE=TG TT_LLAMA_TEXT_VER=llama3_70b_galaxy VLLM_RPC_TIMEOUT=900000

stop_server() {
  [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null
  for i in $(seq 1 30); do pgrep -f server_example_tt.py > /dev/null || break; sleep 5; done
  pkill -9 -f server_example_tt.py; pkill -9 -f 'vllm|EngineCore'; sleep 5
  tt-smi -glx_reset_auto >> "$R/reset_$TAG.log" 2>&1
  cp "$LOGDIR/vllm_server_$TAG.log" "$R/" 2>/dev/null
  [ -d "$LOGDIR/bench_$TAG" ] && cp -r "$LOGDIR/bench_$TAG" "$R/"
}

tt-smi -glx_reset_auto > "$R/reset_$TAG.log" 2>&1 || { log "reset failed"; finish 3; }

# 2. Server. The first start after the precision change builds a new weight cache (bfp4) and takes longer.
cd "$PL"
python examples/server_example_tt.py --model "$HF_MODEL" --served-model-name llama70b \
  --data_parallel_size 4 --max_num_seqs 8 --async-scheduling --host 127.0.0.1 --port 8000 \
  --max-model-len 36864 --enable-prompt-tokens-details --override-generation-config '{"top_k": 32}' \
  --additional-config "$(cat "$HERE/serving_tt_config.json")" > "$LOGDIR/vllm_server_$TAG.log" 2>&1 &
SRV=$!
log "server starting (pid $SRV, log $LOGDIR/vllm_server_$TAG.log)"
for i in $(seq 1 720); do
  curl -sf localhost:8000/v1/models > /dev/null && break
  kill -0 $SRV 2>/dev/null || { log "server exited during start"; stop_server; finish 4; }
  sleep 10
done
curl -sf localhost:8000/v1/models > /dev/null || { log "server not ready after 2 h"; stop_server; finish 5; }
log "server ready"
for k in "Experimental keep_decode_chain" "Prefill admission cap" "Prefill output flush" "Decode interleave disabled" "Experimental arrival wait"; do
  log "server log: $(grep -m1 -c "$k" "$LOGDIR/vllm_server_$TAG.log") x '$k'"
done

smoke=$(curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"llama70b","prompt":"The capital of France is","max_tokens":20,"temperature":0,"top_k":1}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])' 2>&1)
log "smoke: $(printf '%q' "$smoke")"
printf '%s' "$smoke" | python3 -c 'import re,sys; t=sys.stdin.read(); sys.exit(0 if len(re.findall(r"[A-Za-z]{2,}",t))>=3 and not re.search(r"\\\\{2,}",t) else 1)' \
  || { log "smoke failed"; stop_server; finish 6; }

# 3. Benchmark (vllm bench serve, fresh seeds) and the SLA score.
rc=0
OUT=$LOGDIR/bench_$TAG HIT_TAG=${TAG}_rpm60_40pct bash "$TEJA/bench_tis.sh" > "$R/bench_$TAG.driver.log" 2>&1
log "bench done: $(tail -1 "$R/bench_$TAG.driver.log")"
python3 "$HERE/slide5.py" "$LOGDIR/bench_$TAG" "$R/bench/hit_${TAG}_rpm60_40pct.json" | tee "$R/score_$TAG.txt" | tee -a "$S"
[ "${PIPESTATUS[0]}" -ne 0 ] && rc=$((rc | 1))
python3 - "$LOGDIR/bench_$TAG/rpm60_prefix40.json" <<'EOF' | tee -a "$S"
import json, sys
d = json.load(open(sys.argv[1]))
print(f"60 RPM + prefix TTFT, 4 decimals: mean {d['mean_ttft_ms'] / 1000:.4f} s, P99 {d['p99_ttft_ms'] / 1000:.3f} s")
EOF

# 4. Correctness gates: long-prompt needle, prefix-cache-hit needle, MMLU.
cd "$TM"
python3 "$HERE/needle_probe.py" "$R/needle_$TAG.json" > "$R/needle_$TAG.txt" 2>&1
n=$?; log "needle: $(tail -1 "$R/needle_$TAG.txt")"; [ $n -ne 0 ] && rc=$((rc | 16))
python3 "$HERE/prefix_needle_probe.py" "$R/pneedle_$TAG.json" > "$R/pneedle_$TAG.txt" 2>&1
p=$?; log "prefix needle: $(tail -1 "$R/pneedle_$TAG.txt")"; [ $p -ne 0 ] && rc=$((rc | 32))

if [ "$MMLU" != 0 ]; then
  # lm-eval queues every request at once and counts the queue wait against the client timeout:
  # the full set (14042 requests, ~75 min) needs a timeout longer than the whole run.
  lim=""; tmo=86400; [ "$MMLU" = limit ] && lim="--limit 20" && tmo=600
  MODEL=llama70b NC=8 TMO=$tmo bash "$TEJA/run_mmlu.sh" "$TAG" $lim
  python3 "$HERE/mmlu_check.py" "$R/mmlu_$TAG" | tee -a "$S"
  [ "${PIPESTATUS[0]}" -ne 0 ] && rc=$((rc | 8))
fi

stop_server
log "server stopped, Galaxy reset; outputs in $R/{case,score,needle,pneedle,mmlu,vllm_server}_$TAG*, $R/bench_$TAG/"
finish $rc
