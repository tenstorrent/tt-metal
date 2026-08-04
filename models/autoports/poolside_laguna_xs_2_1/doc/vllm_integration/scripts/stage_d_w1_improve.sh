#!/bin/bash
# Latency pass showing the W1 fix (+PREFILL_FAST). Same config as stage_d_c1c8 BUT deliberately NO
# client-side warmup pass — the W1 code fix (warmup_model_prefill pre-warms (1..max_num_seqs, serve_w))
# should make the server warm-by-construction. If the fix holds, concurrent (C=8) prefills produce ZERO
# "prefill page-table alloc AT SERVING" warnings and no stall. vllm bench serve ONLY; measured E2EL.
set +e
LOCAL=/home/ttuser/.local/lib/model-bringup/tt-metal
BASE=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1
OUTD=$BASE/doc/vllm_integration/stage_d_w1
mkdir -p "$OUTD"
LOG=$OUTD/sweep.log; TSV=$OUTD/sweep.tsv; SRVLOG=$BASE/readiness_vllm/server.log
BV=/home/ttuser/.local/lib/tt-inference-server/.workflow_venvs/.venv_benchmarks_vllm/bin
VBIN=/home/ttuser/.tenstorrent-venv/bin
SERVE_PP=/home/ttuser/dev/tt-metal:$LOCAL/vllm:$LOCAL/vllm/plugins/vllm-tt-plugin/src
: > "$LOG"; : > "$SRVLOG" 2>/dev/null
echo -e "# server\tISL\tOSL\tCONC\tt/s/u\tagg_tok/s\tE2EL_s(meas)\tTTFT_ms\tMeanITL\tP99ITL" > "$TSV"
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

start_server(){
  log "START W1-fixed server (k64 + PREFILL_FAST; NO client warmup pass; max-num-seqs 8; 131072; APC OFF)"
  cd /tmp
  setsid bash -c "
    export TT_METAL_HOME=$LOCAL PYTHONPATH=$SERVE_PP TT_LAGUNA_PIPE_CHUNK=2048 \
           TT_LAGUNA_WEIGHT_CACHE_DISABLE=1 TT_LAGUNA_DECODE_SDPA_PC=1 TT_LAGUNA_PREFILL_FAST=1
    exec $VBIN/python -u -m models.common.readiness_check.run_vllm_server \
      --model-dir $BASE --hf-model poolside/Laguna-XS-2.1 --mesh-device P150x4 --stages serve \
      --max-num-seqs 8 --block-size 64 --max-model-len 131072 \
      --tt-config '{\"trace_region_size\":1500000000,\"fabric_config\":\"FABRIC_1D_RING\"}' \
      --additional-server-args='--trust-remote-code --max-num-batched-tokens 131072 --no-enable-prefix-caching'
  " >> "$LOG" 2>&1 &
  echo $! > /tmp/laguna_srv_w1
  local i; for i in $(seq 1 400); do sleep 5
    curl -sf -m3 http://localhost:8000/health >/dev/null 2>&1 && { log "healthy ~$((i*5))s"; return 0; }
    grep -qiE "out of memory|Traceback|EngineDeadError" "$LOG" 2>/dev/null && log "boot issue? still waiting"
  done; log "FAILED health"; return 1; }

stop_server(){ local g; g=$(cat /tmp/laguna_srv_w1 2>/dev/null)
  [ -n "$g" ] && kill -TERM -"$g" 2>/dev/null; sleep 12; [ -n "$g" ] && kill -KILL -"$g" 2>/dev/null; sleep 5
  tt-smi -r all >/dev/null 2>&1; sleep 10; log "STOPPED + reset"; }

vbench(){ # isl osl conc np
  local out=$OUTD/r_${1}_${2}_${3}.json
  log ">>> bench ISL=$1 OSL=$2 conc=$3 np=$4"
  timeout 3000 $BV/vllm bench serve --backend vllm --base-url http://localhost:8000 --endpoint /v1/completions \
    --model poolside/Laguna-XS-2.1 --dataset-name random --num-prompts $4 \
    --random-input-len $1 --random-output-len $2 --max-concurrency $3 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --save-result --result-filename "$out" 2>&1 | tee -a "$LOG"
  $VBIN/python - "$out" k64 "$1" "$2" "$3" >> "$TSV" <<'PY'
import json,sys
f,srv,isl,osl,conc=sys.argv[1:6]
try:
  d=json.load(open(f)); tpot=d["mean_tpot_ms"]; ttft=d["mean_ttft_ms"]
  agg=d.get("output_throughput") or 0.0; tsu=1000.0/tpot if tpot else 0.0
  e2=d.get("mean_e2el_ms"); e2s=f"{e2/1000.0:.1f}" if e2 else "n/a"
  mitl=d.get("mean_itl_ms") or 0.0; p99=d.get("p99_itl_ms") or 0.0
  print(f"{srv}\t{isl}\t{osl}\t{conc}\t{tsu:.2f}\t{agg:.1f}\t{e2s}\t{ttft:.0f}\t{mitl:.1f}\t{p99:.1f}")
except Exception as e:
  print(f"{srv}\t{isl}\t{osl}\t{conc}\tERR\tERR\tERR\tERR\tERR\t{e}")
PY
  log "row: $(tail -1 "$TSV")"; }

if start_server; then
  for ISL in 1024 16384 32768 130048; do vbench $ISL 1024 1 1; done          # C=1 single-user, OSL 1024
  for ISL in 1024 16384 32768; do vbench $ISL 128 8 16; done                 # C=8 concurrency, OSL 128
  # W1 PROOF: with the fix, concurrent (N,w) prefills never allocate under the resident decode trace.
  W1WARN=$(grep -c "prefill page-table alloc.*AT SERVING" "$SRVLOG" 2>/dev/null)
  log "W1-PROOF: prefill-pt allocations AT SERVING during the whole sweep = $W1WARN (want 0)"
  stop_server
else stop_server; fi
log "=== W1 IMPROVE SWEEP DONE ==="; echo; cat "$TSV" | tee -a "$LOG"
