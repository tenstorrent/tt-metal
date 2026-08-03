#!/bin/bash
# Stage D — full latency sweep (Laguna-XS-2.1, P150x4). vllm bench serve ONLY; measured E2EL; never ms/tok.
#
# Two servers IN SEQUENCE (never concurrent -> no two-instance collision):
#   (1) SHIPPED k64 decode (TT_LAGUNA_DECODE_SDPA_PC=1), APC OFF: full single-user grid + concurrency ladder.
#   (2) BASELINE ttnn-default decode (=0), APC OFF: long-context single-user points only = the decode-speed A/B
#       that "shows the improvement" (k64 keeps the max_cores=16 long-ctx speed at =1 accuracy).
#
# Metrics per row: ISL / OSL / E2EL(measured mean_e2el_ms) / TTFT / t/s/u (1000/mean_tpot) / agg tok/s.
# Weight cache DISABLED (unvalidated). APC OFF because random-dataset seed=0 => identical prompts across
# calls; with APC ON the 2nd+ call to an ISL is a cache hit and latency is garbage (prior lesson).
set +e
LOCAL=/home/ttuser/.local/lib/model-bringup/tt-metal
BASE=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1
OUTD=$BASE/doc/vllm_integration/stage_d
mkdir -p "$OUTD"
LOG=$OUTD/sweep.log            # <-- TAIL THIS (control + every bench's full output)
TSV=$OUTD/sweep.tsv
BV=/home/ttuser/.local/lib/tt-inference-server/.workflow_venvs/.venv_benchmarks_vllm/bin
VBIN=/home/ttuser/.tenstorrent-venv/bin
SERVE_PP=/home/ttuser/dev/tt-metal:$LOCAL/vllm:$LOCAL/vllm/plugins/vllm-tt-plugin/src
: > "$LOG"
echo -e "# Stage D latency sweep — MEASURED E2EL (mean_e2el_ms), APC OFF, weight-cache OFF." > "$TSV"
echo -e "# server\tISL\tOSL\tCONC\tt/s/u\tagg_tok/s\tE2EL_s(meas)\tTTFT_ms" >> "$TSV"
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

start_server(){ # $1 = SDPA_PC (1 shipped k64 | 0 ttnn baseline) ; $2 = tag
  log "START server tag=$2 (TT_LAGUNA_DECODE_SDPA_PC=$1; max-num-seqs 16; max-model-len 131072; APC OFF)"
  cd /tmp
  setsid bash -c "
    export TT_METAL_HOME=$LOCAL PYTHONPATH=$SERVE_PP TT_LAGUNA_PIPE_CHUNK=2048 \
           TT_LAGUNA_WEIGHT_CACHE_DISABLE=1 TT_LAGUNA_DECODE_SDPA_PC=$1
    exec $VBIN/python -u -m models.common.readiness_check.run_vllm_server \
      --model-dir $BASE --hf-model poolside/Laguna-XS-2.1 --mesh-device P150x4 --stages serve \
      --max-num-seqs 16 --block-size 64 --max-model-len 131072 \
      --tt-config '{\"trace_region_size\":1500000000,\"fabric_config\":\"FABRIC_1D_RING\"}' \
      --additional-server-args='--trust-remote-code --max-num-batched-tokens 131072 --no-enable-prefix-caching'
  " >> "$LOG" 2>&1 &
  echo $! > /tmp/laguna_srv_pgid_stageD
  local i; for i in $(seq 1 360); do sleep 5
    curl -sf -m3 http://localhost:8000/health >/dev/null 2>&1 && { log "healthy ~$((i*5))s"; return 0; }
    grep -qiE "out of memory|Traceback|FATAL" "$LOG" 2>/dev/null && log "BOOT issue? (still waiting) — check log"
  done; log "FAILED health"; return 1; }

stop_server(){ local g; g=$(cat /tmp/laguna_srv_pgid_stageD 2>/dev/null)
  [ -n "$g" ] && kill -TERM -"$g" 2>/dev/null; sleep 12
  [ -n "$g" ] && kill -KILL -"$g" 2>/dev/null; sleep 5
  tt-smi -r all >/dev/null 2>&1; sleep 10; log "STOPPED+reset (eth cores clean for next boot)"; }

vbench(){ # $1 server-tag $2 isl $3 osl $4 conc  [$5 num-prompts (default = conc*2, min 1)]
  local np=${5:-$(( $4>1 ? $4*2 : 1 ))}
  local out=$OUTD/r_${1}_${2}_${3}_${4}.json
  log ">>> vllm bench: server=$1 ISL=$2 OSL=$3 conc=$4 num-prompts=$np"
  timeout 2400 $BV/vllm bench serve --backend vllm --base-url http://localhost:8000 --endpoint /v1/completions \
    --model poolside/Laguna-XS-2.1 --dataset-name random --num-prompts $np \
    --random-input-len $2 --random-output-len $3 --max-concurrency $4 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-filename "$out" 2>&1 | tee -a "$LOG"
  $VBIN/python - "$out" "$1" "$2" "$3" "$4" >> "$TSV" <<'PY'
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

warm(){ log "warm server=$1 ISL=$2"; timeout 600 $BV/vllm bench serve --backend vllm \
    --base-url http://localhost:8000 --endpoint /v1/completions --model poolside/Laguna-XS-2.1 \
    --dataset-name random --num-prompts 1 --random-input-len $2 --random-output-len 8 \
    --max-concurrency 1 --ignore-eos >> "$LOG" 2>&1; }

# ===== Server 1: shipped k64 (=1) — full grid + ladder =====
if start_server 1 k64; then
  for ISL in 1024 16384 32768 130048; do warm k64 $ISL; vbench k64 $ISL 1024 1; done   # single-user, real OSL 1024
  for ISL in 1024 32768; do for C in 8 16; do vbench k64 $ISL 128 $C; done; done         # concurrency ladder, OSL 128
  stop_server
else stop_server; fi

# ===== Server 2: ttnn-default baseline (=0) — long-context A/B only =====
if start_server 0 ttnndef; then
  for ISL in 32768 130048; do warm ttnndef $ISL; vbench ttnndef $ISL 1024 1; done
  stop_server
else stop_server; fi

log "=== STAGE D SWEEP DONE ==="
echo; echo "final TSV:"; cat "$TSV" | tee -a "$LOG"
