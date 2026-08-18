#!/bin/bash
# Full REAP ISL×batch prefill sweep (chunked MoE, batched).
# Grid: ISL ∈ {128,512,1024,2048,4096} × B ∈ {4,8,16,32} — 20 cells.
# Order: ascending T=ISL×B so small cells finish first.
# RESUME=1 skips cells whose log already contains prefill_s=.
set -euo pipefail

REAP_HOME=/home/tt-admin/sdawle/glm47_reap_218b/tt-metal
OUT=$REAP_HOME/models/experimental/glm4_moe/experiments/prefill_opt_sweep
mkdir -p "$OUT"
SUMMARY=$OUT/reap_full_new_summary.csv
LOG=$OUT/reap_full_sweep.log
RESUME=${RESUME:-0}
# Per-cell wall timeout (seconds). Kill + mark timeout if exceeded.
CELL_TIMEOUT_S=${CELL_TIMEOUT_S:-3600}
PCM=${PCM:-1}
CHUNK=${CHUNK:-4096}

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

wait_idle() {
  local n=0
  while pgrep -f '[p]ython3 .*debug_run_full_tt_greedy\.py' >/dev/null 2>&1 \
     || pgrep -f '[p]ython3 .*run_sweep_isl_batch\.py' >/dev/null 2>&1; do
    n=$((n+1))
    if (( n % 6 == 1 )); then log "waiting for in-flight greedy/sweep to finish..."; fi
    sleep 10
  done
}

parse_and_record() {
  local tag="$1" isl="$2" batch="$3" T="$4" logfile="$5" exit_code="$6" note="${7:-}"
  local status="ok" prefill="" decode="" detail=""
  if [[ "$exit_code" == "124" ]]; then
    status="timeout"
    detail="CELL_TIMEOUT_S=${CELL_TIMEOUT_S}"
  elif [[ "$exit_code" != "0" ]]; then
    status="fail"
    if rg -qi 'Out of Memory|beyond max L1|bank_manager|Not enough space' "$logfile" 2>/dev/null; then
      status="OOM"
    fi
    detail=$(rg -n 'TT_THROW|Traceback|Error|Out of Memory|beyond max L1|timeout' "$logfile" 2>/dev/null | tail -1 | cut -c1-180 | tr ',' ';' || true)
  fi
  prefill=$(rg -o 'prefill_s=[0-9.]+' "$logfile" 2>/dev/null | tail -1 | cut -d= -f2 || true)
  decode=$(rg -o 'subsequent:\s+mean=\s*[0-9.]+' "$logfile" 2>/dev/null | tail -1 | awk '{print $NF}' || true)
  if [[ "$status" == "ok" ]] && [[ -z "$prefill" ]]; then
    status="fail"
    detail="${detail};WARN_no_prefill_s"
  fi
  if ! rg -q 'Batched prefill' "$logfile" 2>/dev/null; then
    detail="${detail};WARN_no_batched_marker"
  fi
  echo "$tag,$isl,$batch,$T,batched,pcm=$PCM,chunk=$CHUNK,$status,${prefill:-},${decode:-},${detail:-},${note},$logfile" >> "$SUMMARY"
  log "$tag isl=$isl b=$batch T=$T status=$status prefill=${prefill:-?} decode_ms=${decode:-?} ec=$exit_code"
}

run_cell() {
  local isl="$1" batch="$2"
  local T=$((isl * batch))
  local tag="reap_full_isl${isl}_b${batch}_batched_pcm${PCM}_chk${CHUNK}"
  local logfile="$OUT/${tag}.log"

  if [[ "$RESUME" == "1" && -f "$logfile" ]] && rg -q 'prefill_s=' "$logfile" 2>/dev/null; then
    log "SKIP $tag (resume, log has prefill_s)"
    # Ensure CSV has a row (idempotent: skip if already recorded)
    if ! rg -q "^${tag}," "$SUMMARY" 2>/dev/null; then
      parse_and_record "$tag" "$isl" "$batch" "$T" "$logfile" 0 "resume_from_log"
    fi
    return 0
  fi

  wait_idle
  log "START $tag T=$T timeout=${CELL_TIMEOUT_S}s"
  cd "$REAP_HOME"
  export TT_METAL_HOME=$REAP_HOME PYTHONPATH=$REAP_HOME
  export GLM4_MOE_REDUCE_IMPL=native
  export GLM4_MOE_EP_REDUCE_DEVICE=1
  export GLM4_MOE_EXPERTS_TT_DTYPE=bf4
  export GLM4_MOE_DISTRIBUTED_QK_NORM=1
  export GLM4_MOE_ROUTER_USE_BIASED_TOPK_VALUES=1
  export GLM4_MOE_CCL_NUM_LINKS=4
  export GLM4_MOE_CCL_TOPOLOGY=ring
  export GLM4_MOE_DRAM_SHARD=1
  export GLM4_MOE_PACKER_L1_ACC=1
  export GLM4_MOE_EP_L1=1
  export GLM4_MOE_SDPA_L1=1
  export GLM4_MOE_PREFILL_CHUNK_SIZE=131072
  export GLM4_MOE_BATCHED_PREFILL=1
  export GLM4_MOE_BATCHED_PREFILL_MAX_TOKENS=131072
  # Explicit chunked path (unambiguous vs adaptive default)
  export GLM4_MOE_MOE_SPARSE_PREFILL_PCM="$PCM"
  export GLM4_MOE_MOE_SPARSE_PREFILL_CHUNK_TOKENS="$CHUNK"
  # Keep attn rs_ag — do not force host AR
  unset GLM4_MOE_ATTN_PREFILL_REDUCE_IMPL || true

  local min_cache=$((isl + 16))
  set +e
  timeout --signal=TERM --kill-after=60 "$CELL_TIMEOUT_S" \
    ./python_env/bin/python3 models/experimental/glm4_moe/scripts/debug_run_full_tt_greedy.py \
      --mesh-rows 8 --mesh-cols 4 \
      --model-id cerebras/GLM-4.7-REAP-218B-A32B \
      --simulate-context-len "$isl" --batch-size "$batch" \
      --max-new-tokens 8 --min-cache-tokens "$min_cache" \
      --enable-trace --trace-mode sampling --warmup-decode-trace \
      > "$logfile" 2>&1
  local ec=$?
  set -e
  parse_and_record "$tag" "$isl" "$batch" "$T" "$logfile" "$ec" ""
}

# Cells sorted by ascending T=ISL×B
CELLS=(
  "128 4"     # 512
  "128 8"     # 1024
  "128 16"    # 2048
  "512 4"     # 2048
  "128 32"    # 4096
  "512 8"     # 4096
  "1024 4"    # 4096
  "512 16"    # 8192
  "1024 8"    # 8192
  "2048 4"    # 8192
  "512 32"    # 16384
  "1024 16"   # 16384
  "2048 8"    # 16384
  "4096 4"    # 16384
  "1024 32"   # 32768
  "2048 16"   # 32768
  "4096 8"    # 32768
  "2048 32"   # 65536
  "4096 16"   # 65536
  "4096 32"   # 131072
)

if [[ "$RESUME" != "1" || ! -f "$SUMMARY" ]]; then
  echo "tag,isl,batch,T,mode,pcm,chunk,status,prefill_s,decode_mean_ms,detail,note,logfile" > "$SUMMARY"
elif [[ "$RESUME" == "1" && ! -f "$SUMMARY" ]]; then
  echo "tag,isl,batch,T,mode,pcm,chunk,status,prefill_s,decode_mean_ms,detail,note,logfile" > "$SUMMARY"
fi

log "=== REAP full ISL×batch sweep start (RESUME=$RESUME PCM=$PCM CHUNK=$CHUNK CELL_TIMEOUT_S=$CELL_TIMEOUT_S) ==="
log "MAX_TOKENS=131072 mesh=8x4 model=cerebras/GLM-4.7-REAP-218B-A32B"

for cell in "${CELLS[@]}"; do
  read -r isl batch <<<"$cell"
  run_cell "$isl" "$batch" || log "WARN run_cell returned non-zero for isl=$isl b=$batch (continuing)"
done

log "=== REAP full ISL×batch sweep done ==="
log "Summary: $SUMMARY"
