#!/bin/bash
# Warm TTFT sweep — prefix caching ON, against the already-running server. For each ISL: WARM it (one call
# populates the prefix cache), then MEASURE (same seed=0 prompt -> cache hit) -> the warm TTFT = the
# representative per-turn prefill cost for an agent (~95% cache hit). Complements the COLD Stage-D sweep.
# No server management here (reuses the up APC-on server). vllm bench only.
set +e
BASE=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1
OUTD=$BASE/doc/vllm_integration/stage_d_warm
mkdir -p "$OUTD"
LOG=$OUTD/warm.log        # <-- TAIL THIS
TSV=$OUTD/warm.tsv
BV=/home/ttuser/.local/lib/tt-inference-server/.workflow_venvs/.venv_benchmarks_vllm/bin
VBIN=/home/ttuser/.tenstorrent-venv/bin
: > "$LOG"
echo -e "# Warm TTFT sweep — prefix caching ON (cache-hit = per-turn cost). ISL\tTTFT_ms(warm)\tt/s/u\tE2EL_s" > "$TSV"
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

curl -sf -m4 http://localhost:8000/health >/dev/null 2>&1 || { log "server NOT healthy — abort"; exit 1; }
log "=== warm TTFT sweep (APC on) — warm then measure per ISL ==="

warm(){ log "warm (populate cache) ISL=$1"; timeout 900 $BV/vllm bench serve --backend vllm \
    --base-url http://localhost:8000 --endpoint /v1/completions --model poolside/Laguna-XS-2.1 \
    --dataset-name random --num-prompts 1 --random-input-len $1 --random-output-len 8 \
    --max-concurrency 1 --ignore-eos >> "$LOG" 2>&1; }

vbench(){ # $1 isl : measured cache-hit call
  local out=$OUTD/warm_${1}.json
  log ">>> MEASURE (cache hit) ISL=$1 OSL=128"
  timeout 900 $BV/vllm bench serve --backend vllm --base-url http://localhost:8000 --endpoint /v1/completions \
    --model poolside/Laguna-XS-2.1 --dataset-name random --num-prompts 1 \
    --random-input-len $1 --random-output-len 128 --max-concurrency 1 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --save-result --result-filename "$out" 2>&1 | tee -a "$LOG"
  $VBIN/python - "$out" "$1" >> "$TSV" <<'PY'
import json,sys
f,isl=sys.argv[1:3]
try:
  d=json.load(open(f)); ttft=d["mean_ttft_ms"]; tpot=d["mean_tpot_ms"]
  e2=d.get("mean_e2el_ms"); e2s=f"{e2/1000.0:.1f}" if e2 else "n/a"
  print(f"{isl}\t{ttft:.0f}\t{1000.0/tpot:.2f}\t{e2s}")
except Exception as e:
  print(f"{isl}\tERR\tERR\t{e}")
PY
  log "row: $(tail -1 "$TSV")"; }

for ISL in 1024 16384 32768 130048; do warm $ISL; sleep 2; vbench $ISL; done
log "=== WARM SWEEP DONE ==="; echo; cat "$TSV" | tee -a "$LOG"
