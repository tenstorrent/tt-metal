#!/bin/bash
# Validate the bounded-prefill fix (TT_LAGUNA_PREFILL_FAST). Two checks:
#  (1) ACCURACY: standalone prefill-PCC with FAST=1 must PASS (fix is accuracy-neutral — only chunk COUNT changes).
#  (2) SPEED: FAST=1 + APC-off server, cold prefill @130048 TTFT should be ~126s (vs 299s FAST=0, from Stage D).
set +e
LOCAL=/home/ttuser/.local/lib/model-bringup/tt-metal
BASE=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1
OUTD=$BASE/doc/vllm_integration/prefill_fast_validate
mkdir -p "$OUTD"
LOG=$OUTD/validate.log        # <-- TAIL THIS
BV=/home/ttuser/.local/lib/tt-inference-server/.workflow_venvs/.venv_benchmarks_vllm/bin
VBIN=/home/ttuser/.tenstorrent-venv/bin
SERVE_PP=/home/ttuser/dev/tt-metal:$LOCAL/vllm:$LOCAL/vllm/plugins/vllm-tt-plugin/src
: > "$LOG"
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# --- 0. down the current server + reset the mesh ---
log "down current server + mesh reset"
for p in $(lsof -t /dev/tenstorrent/* 2>/dev/null | sort -u; lsof -t -i:8000 2>/dev/null; pgrep -f run_vllm_server 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 6; tt-smi -r all >/dev/null 2>&1; sleep 10

# --- 1. ACCURACY: prefill-PCC with FAST=1 ---
log "=== (1) ACCURACY: prefill-PCC with TT_LAGUNA_PREFILL_FAST=1 (must PASS; PCC>=0.995) ==="
cd /tmp
TT_METAL_HOME=$LOCAL PYTHONPATH=/home/ttuser/dev/tt-metal TT_LAGUNA_PREFILL_FAST=1 TT_LAGUNA_WEIGHT_CACHE_DISABLE=1 \
  $VBIN/python -m pytest -q \
  "$BASE/tests/test_multichip_decoder.py::test_prefill_pcc" \
  "$BASE/tests/test_multichip_decoder.py::test_prefill_chunked_matches_hf" \
  "$BASE/tests/test_multichip_decoder.py::test_prefill_pipelined_matches_hf" 2>&1 | tee -a "$LOG" | grep -aE "passed|failed|error|PASSED|FAILED|PCC"
log "accuracy result: $(grep -aE 'passed|failed' "$LOG" | tail -1)"
tt-smi -r all >/dev/null 2>&1; sleep 10

# --- 2. SPEED: FAST=1 + APC-off server, cold @130048 TTFT ---
log "=== (2) SPEED: boot FAST=1 APC-off server; cold prefill @130048 (expect TTFT ~126s vs 299s FAST=0) ==="
setsid bash -c "
  export TT_METAL_HOME=$LOCAL PYTHONPATH=$SERVE_PP TT_LAGUNA_PIPE_CHUNK=2048 TT_LAGUNA_PREFILL_FAST=1 TT_LAGUNA_WEIGHT_CACHE_DISABLE=1
  exec $VBIN/python -u -m models.common.readiness_check.run_vllm_server \
    --model-dir $BASE --hf-model poolside/Laguna-XS-2.1 --mesh-device P150x4 --stages serve \
    --max-num-seqs 16 --block-size 64 --max-model-len 131072 \
    --tt-config '{\"trace_region_size\":1500000000,\"fabric_config\":\"FABRIC_1D_RING\"}' \
    --additional-server-args='--trust-remote-code --max-num-batched-tokens 131072 --no-enable-prefix-caching'
" >> "$LOG" 2>&1 &
echo $! > /tmp/laguna_srv_pgid_fastval
for i in $(seq 1 360); do sleep 5; curl -sf -m3 http://localhost:8000/health >/dev/null 2>&1 && { log "healthy ~$((i*5))s"; break; }
  grep -qiE "out of memory|Traceback" "$LOG" 2>/dev/null && log "BOOT issue? check log"; done
log ">>> MEASURE cold prefill @130048 (FAST=1, OSL 8 -> TTFT only)"
timeout 900 $BV/vllm bench serve --backend vllm --base-url http://localhost:8000 --endpoint /v1/completions \
  --model poolside/Laguna-XS-2.1 --dataset-name random --num-prompts 1 \
  --random-input-len 130048 --random-output-len 8 --max-concurrency 1 --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el --save-result --result-filename "$OUTD/fast_130048.json" 2>&1 | tee -a "$LOG"
$VBIN/python -c "import json;d=json.load(open('$OUTD/fast_130048.json'));t=d['mean_ttft_ms']/1000; print(f'RESULT FAST=1 @130048: TTFT={t:.1f}s  (FAST=0 baseline 298.7s -> speedup {298.7/t:.2f}x)')" 2>&1 | tee -a "$LOG"
# check no OOM (confirms 8192 chunk fits at 128k)
grep -qi "out of memory" "$LOG" && log "WARNING: OOM seen — fast path NOT safe at this context" || log "OOM check: none (8192 chunk fits <=128k)"
# teardown
g=$(cat /tmp/laguna_srv_pgid_fastval 2>/dev/null); [ -n "$g" ] && kill -KILL -"$g" 2>/dev/null; sleep 5; tt-smi -r all >/dev/null 2>&1; sleep 8
log "=== PREFILL-FAST VALIDATION DONE ==="
