#!/bin/bash
# Latency sweep to 128k @ concurrency 1 AND 8 (Laguna-XS-2.1, P150x4). vllm bench serve ONLY.
# One server: shipped k64 decode (TT_LAGUNA_DECODE_SDPA_PC=1) + PREFILL_FAST (cold long-ctx TTFT).
# APC OFF (random seed=0 => identical prompts; APC on = cache-hit garbage). Weight-cache OFF (unvalidated).
# Metrics per row: ISL / OSL / CONC / t/s/u (1000/mean_tpot) / agg tok/s / E2EL_s(measured) / TTFT_ms. Never ms/tok.
set +e
LOCAL=/home/ttuser/.local/lib/model-bringup/tt-metal
BASE=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1
OUTD=$BASE/doc/vllm_integration/stage_d_c1c8
mkdir -p "$OUTD"
LOG=$OUTD/sweep.log
TSV=$OUTD/sweep.tsv
BV=/home/ttuser/.local/lib/tt-inference-server/.workflow_venvs/.venv_benchmarks_vllm/bin
VBIN=/home/ttuser/.tenstorrent-venv/bin
SERVE_PP=/home/ttuser/dev/tt-metal:$LOCAL/vllm:$LOCAL/vllm/plugins/vllm-tt-plugin/src
: > "$LOG"
echo -e "# server\tISL\tOSL\tCONC\tt/s/u\tagg_tok/s\tE2EL_s(meas)\tTTFT_ms" > "$TSV"
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

start_server(){
  log "START server (k64 + PREFILL_FAST; max-num-seqs 8; max-model-len 131072; APC OFF)"
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
  echo $! > /tmp/laguna_srv_c1c8
  local i; for i in $(seq 1 400); do sleep 5
    curl -sf -m3 http://localhost:8000/health >/dev/null 2>&1 && { log "healthy ~$((i*5))s"; return 0; }
    grep -qiE "out of memory|Traceback|FATAL|EngineDeadError" "$LOG" 2>/dev/null && log "BOOT issue? still waiting"
  done; log "FAILED health"; return 1; }

stop_server(){ local g; g=$(cat /tmp/laguna_srv_c1c8 2>/dev/null)
  [ -n "$g" ] && kill -TERM -"$g" 2>/dev/null; sleep 12
  [ -n "$g" ] && kill -KILL -"$g" 2>/dev/null; sleep 5
  tt-smi -r all >/dev/null 2>&1; sleep 10; log "STOPPED + board reset"; }

warm(){ # $1 isl $2 conc — warm the EXACT measured shape (ISL + concurrency) so the measured run excludes
        # first-call compile/alloc (the W1 recompile-under-resident-trace artifact). Matches tt-inference-
        # server's always-warm-before-sweep behavior. Results discarded. Short OSL to keep it quick.
  log "warm ISL=$1 conc=$2"
  timeout 900 $BV/vllm bench serve --backend vllm \
    --base-url http://localhost:8000 --endpoint /v1/completions --model poolside/Laguna-XS-2.1 \
    --dataset-name random --num-prompts $2 --random-input-len $1 --random-output-len 16 \
    --max-concurrency $2 --ignore-eos >> "$LOG" 2>&1; }

vbench(){ # $1 isl $2 osl $3 conc $4 num-prompts
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
  print(f"{srv}\t{isl}\t{osl}\t{conc}\t{tsu:.2f}\t{agg:.1f}\t{e2s}\t{ttft:.0f}")
except Exception as e:
  print(f"{srv}\t{isl}\t{osl}\t{conc}\tERR\tERR\tERR\t{e}")
PY
  log "row: $(tail -1 "$TSV")"; }

if start_server; then
  # ---- WARMUP PASS (tt-inference-server style): touch EVERY measured shape at its measured concurrency
  # BEFORE any measurement, so no first-call compile/alloc pollutes the numbers (the W1 artifact). ----
  log "=== WARMUP PASS (all shapes) ==="
  for ISL in 1024 16384 32768 130048; do warm $ISL 1; done   # C=1 shapes
  for ISL in 1024 16384 32768;        do warm $ISL 8; done   # C=8 shapes (decode-batch-8 + (8,w) prefill)
  # ---- MEASURED PASS ----
  log "=== MEASURED PASS ==="
  # Concurrency 1 — real decode length OSL 1024 (E2EL latency)
  for ISL in 1024 16384 32768 130048; do vbench $ISL 1024 1 1; done
  # Concurrency 8 — OSL 128 (throughput ladder; np = 2*conc). Capped at 32k: the KV pool (~164k tok =
  # ~1.25x of 131072) cannot hold 8 concurrent long contexts, so 128k @ C8 is infeasible/contention-bound.
  for ISL in 1024 16384 32768; do vbench $ISL 128 8 16; done
  stop_server
else stop_server; fi
log "=== SWEEP DONE ==="; echo; echo "final TSV:"; cat "$TSV" | tee -a "$LOG"
