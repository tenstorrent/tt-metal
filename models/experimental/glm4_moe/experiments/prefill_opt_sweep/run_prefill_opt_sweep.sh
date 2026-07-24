#!/bin/bash
# Compact REAP prefill optimization sweep: MoE sparse PCM chunking + batched vs serial.
# Serialize with Flash: wait_idle matches only python3 .*debug_run_full_tt_greedy.py
set -euo pipefail

REAP_HOME=/home/tt-admin/sdawle/glm47_reap_218b/tt-metal
OUT=$REAP_HOME/models/experimental/glm4_moe/experiments/prefill_opt_sweep
mkdir -p "$OUT"
SUMMARY=$OUT/sweep_summary.csv
LOG=$OUT/sweep.log
RESUME=${RESUME:-0}
PHASE=${PHASE:-all}  # smoke | pcm | profile | all

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
  local tag="$1" isl="$2" batch="$3" mode="$4" pcm="$5" chunk="$6" logfile="$7" exit_code="$8" extra="${9:-}"
  local status="ok" prefill="" decode="" detail="" moe_ms="" attn_ms=""
  if [[ "$exit_code" != "0" ]]; then
    status="fail"
    if rg -qi 'Out of Memory|beyond max L1|bank_manager|Not enough space' "$logfile"; then
      status="OOM"
    fi
    detail=$(rg -n 'TT_THROW|Traceback|Error|Out of Memory|beyond max L1' "$logfile" | tail -1 | cut -c1-180 | tr ',' ';' || true)
  fi
  prefill=$(rg -o 'prefill_s=[0-9.]+' "$logfile" | tail -1 | cut -d= -f2 || true)
  decode=$(rg -o 'subsequent:\s+mean=\s*[0-9.]+' "$logfile" | tail -1 | awk '{print $NF}' || true)
  # Optional TIME_LAYER means (best-effort parse)
  moe_ms=$(rg -o 'moe[_ ]?ms[=:][ ]*[0-9.]+' "$logfile" -i | tail -1 | grep -oE '[0-9.]+$' || true)
  attn_ms=$(rg -o 'attn[_ ]?ms[=:][ ]*[0-9.]+' "$logfile" -i | tail -1 | grep -oE '[0-9.]+$' || true)
  if [[ "$mode" == "batched" ]] && ! rg -q 'Batched prefill' "$logfile"; then
    detail="${detail};WARN_no_batched_marker"
  fi
  echo "$tag,$isl,$batch,$mode,pcm=$pcm,chunk=$chunk,$status,${prefill:-},${decode:-},${moe_ms:-},${attn_ms:-},${detail:-},${extra}" >> "$SUMMARY"
  log "$tag isl=$isl b=$batch mode=$mode pcm=$pcm chunk=$chunk status=$status prefill=${prefill:-?} decode_ms=${decode:-?}"
}

run_reap() {
  local isl="$1" batch="$2" mode="$3" pcm="$4" chunk="$5" tag_suffix="${6:-}" time_layer="${7:-0}"
  local tag="reap_isl${isl}_b${batch}_${mode}_pcm${pcm}_chk${chunk}${tag_suffix}"
  local logfile="$OUT/${tag}.log"
  if [[ "$RESUME" == "1" && -f "$logfile" ]] && rg -q 'prefill_s=' "$logfile" 2>/dev/null; then
    log "SKIP $tag (resume, log has prefill_s)"
    return 0
  fi
  wait_idle
  log "START $tag"
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
  export GLM4_MOE_BATCHED_PREFILL_MAX_TOKENS=65536
  # Do NOT force host attn AR — keep rs_ag / device reduce defaults.
  unset GLM4_MOE_ATTN_PREFILL_REDUCE_IMPL || true
  export GLM4_MOE_MOE_SPARSE_PREFILL_PCM="$pcm"
  export GLM4_MOE_MOE_SPARSE_PREFILL_CHUNK_TOKENS="$chunk"
  export GLM4_MOE_TIME_LAYER="$time_layer"
  if [[ "$mode" == "batched" ]]; then
    export GLM4_MOE_BATCHED_PREFILL=1
  else
    export GLM4_MOE_BATCHED_PREFILL=0
  fi
  local min_cache=$((isl + 16))
  set +e
  ./python_env/bin/python3 models/experimental/glm4_moe/scripts/debug_run_full_tt_greedy.py \
    --mesh-rows 8 --mesh-cols 4 \
    --model-id cerebras/GLM-4.7-REAP-218B-A32B \
    --simulate-context-len "$isl" --batch-size "$batch" \
    --max-new-tokens 8 --min-cache-tokens "$min_cache" \
    --enable-trace --trace-mode sampling --warmup-decode-trace \
    > "$logfile" 2>&1
  local ec=$?
  set -e
  parse_and_record "$tag" "$isl" "$batch" "$mode" "$pcm" "$chunk" "$logfile" "$ec" "time_layer=$time_layer"
}

if [[ "$RESUME" != "1" && ! -f "$SUMMARY" ]]; then
  echo "tag,isl,batch,mode,pcm,chunk,status,prefill_s,decode_mean_ms,moe_ms,attn_ms,detail,extra" > "$SUMMARY"
elif [[ "$RESUME" != "1" ]]; then
  : # keep existing if re-invoked mid-run with RESUME unset but file present
fi
if [[ "$RESUME" != "1" && "${FORCE_HEADER:-0}" == "1" ]]; then
  echo "tag,isl,batch,mode,pcm,chunk,status,prefill_s,decode_mean_ms,moe_ms,attn_ms,detail,extra" > "$SUMMARY"
fi

log "=== REAP prefill_opt_sweep start (PHASE=$PHASE RESUME=$RESUME) ==="

# ---- SMOKE: ISL=128 B=4 batched vs serial with chunking ON (PCM=1) ----
if [[ "$PHASE" == "all" || "$PHASE" == "smoke" ]]; then
  log "--- smoke: chunking ON (PCM=1) ---"
  run_reap 128 4 batched 1 4096 "" 0
  run_reap 128 4 serial 1 4096 "" 0
  # baseline legacy path (no chunking)
  log "--- smoke: baseline no chunking (PCM=0 CHUNK=0) ---"
  run_reap 128 4 batched 0 0 "_baseline" 0
  run_reap 128 4 serial 0 0 "_baseline" 0
fi

# ---- PCM grid on key ISL×B ----
if [[ "$PHASE" == "all" || "$PHASE" == "pcm" ]]; then
  log "--- PCM grid ---"
  # 128×4: PCM ∈ {0,1,2,4} batched + serial at PCM=1
  for pcm in 0 1 2 4; do
    chk=4096
    [[ "$pcm" == "0" ]] && chk=0
    run_reap 128 4 batched "$pcm" "$chk" "" 0
  done
  run_reap 128 4 serial 1 4096 "" 0
  run_reap 128 1 batched 1 4096 "" 0

  # 512×8
  for pcm in 0 1 2 4; do
    chk=4096
    [[ "$pcm" == "0" ]] && chk=0
    run_reap 512 8 batched "$pcm" "$chk" "" 0
  done
  run_reap 512 8 serial 1 4096 "" 0

  # 1024×8
  for pcm in 1 2 4; do
    run_reap 1024 8 batched "$pcm" 4096 "" 0
  done
  run_reap 1024 8 serial 1 4096 "" 0
  # one baseline at 1024×8 if affordable
  run_reap 1024 8 batched 0 0 "_baseline" 0
fi

# ---- Short TIME_LAYER profile subset ----
if [[ "$PHASE" == "all" || "$PHASE" == "profile" ]]; then
  log "--- TIME_LAYER profile subset ---"
  run_reap 128 4 batched 0 0 "_tl_base" 1
  run_reap 128 4 batched 1 4096 "_tl_pcm1" 1
  run_reap 128 4 serial 1 4096 "_tl_pcm1" 1
fi

log "=== REAP prefill_opt_sweep done ==="
