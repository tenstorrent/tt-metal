#!/bin/bash
# Short paired validation: optimal vs baseline prefill knobs (Flash then REAP).
# Device exclusivity: waits for idle python3 .*debug_run_full_tt_greedy.py
set -euo pipefail

FLASH_HOME=/home/tt-admin/sdawle/glm47_flash_wh_glx/tt-metal
REAP_HOME=/home/tt-admin/sdawle/glm47_reap_218b/tt-metal
FLASH_OUT=$FLASH_HOME/models/experimental/glm4_moe_lite/experiments/prefill_opt_sweep
REAP_OUT=$REAP_HOME/models/experimental/glm4_moe/experiments/prefill_opt_sweep
FLASH_SUMMARY=$FLASH_OUT/validation_summary.csv
REAP_SUMMARY=$REAP_OUT/validation_summary.csv
LOG=$REAP_OUT/validation.log
# Timebox for REAP PCM0 @ 512×8 (seconds). 0 = no limit.
REAP_512_PCM0_TIMEOUT=${REAP_512_PCM0_TIMEOUT:-2400}

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

record() {
  local summary="$1" tag="$2" model="$3" isl="$4" batch="$5" mode="$6" pcm="$7" chunk="$8" \
        role="$9" logfile="${10}" exit_code="${11}" note="${12:-}"
  local status="ok" prefill="" decode="" detail=""
  if [[ "$exit_code" == "124" ]]; then
    status="timeout"
    detail="timed_out"
  elif [[ "$exit_code" != "0" ]]; then
    status="fail"
    if rg -qi 'Out of Memory|beyond max L1|bank_manager|Not enough space' "$logfile" 2>/dev/null; then
      status="OOM"
    fi
    detail=$(rg -n 'TT_THROW|Traceback|Error|Out of Memory|beyond max L1|Killed' "$logfile" 2>/dev/null | tail -1 | cut -c1-180 | tr ',' ';' || true)
  fi
  prefill=$(rg -o 'prefill_s=[0-9.]+' "$logfile" 2>/dev/null | tail -1 | cut -d= -f2 || true)
  decode=$(rg -o 'subsequent:\s+mean=\s*[0-9.]+' "$logfile" 2>/dev/null | tail -1 | awk '{print $NF}' || true)
  if [[ "$mode" == "batched" ]] && [[ -f "$logfile" ]] && ! rg -q 'Batched prefill' "$logfile"; then
    detail="${detail};WARN_no_batched_marker"
  fi
  echo "$tag,$model,$isl,$batch,$mode,pcm=$pcm,chunk=$chunk,$role,$status,${prefill:-},${decode:-},${detail:-}${note:+;$note}" >> "$summary"
  log "RECORD $tag role=$role status=$status prefill=${prefill:-?} decode_ms=${decode:-?}"
}

run_flash() {
  local isl="$1" batch="$2" mode="$3" pcm="$4" chunk="$5" role="$6"
  local tag="val_flash_isl${isl}_b${batch}_${mode}_pcm${pcm}_chk${chunk}"
  local logfile="$FLASH_OUT/${tag}.log"
  wait_idle
  log "START $tag ($role)"
  cd "$FLASH_HOME"
  export TT_METAL_HOME=$FLASH_HOME PYTHONPATH=$FLASH_HOME
  export GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1
  export GLM4_MOE_LITE_FUSE_QKV_A=1
  export GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1
  export GLM4_MOE_LITE_DECODE_L1_ACT=0
  export GLM4_MOE_LITE_EP_L1=0
  export GLM4_MOE_LITE_MAX_PREFILL_CHUNK_SIZE=16384
  export GLM4_MOE_LITE_CCL_NUM_LINKS=4
  export GLM4_MOE_LITE_CCL_TOPOLOGY=ring
  export GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1
  export GLM4_MOE_LITE_SKIP_TYPECAST=1
  export GLM4_MOE_LITE_MOE_SPARSE_PREFILL_PCM="$pcm"
  export GLM4_MOE_LITE_MOE_SPARSE_CHUNK_TOKENS="$chunk"
  if [[ "$mode" == "batched" ]]; then
    export GLM4_MOE_LITE_BATCHED_PREFILL=1
  else
    export GLM4_MOE_LITE_BATCHED_PREFILL=0
  fi
  local min_cache=$((isl + 16))
  set +e
  ./python_env/bin/python3 models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
    --prompt "Summarize the following document. " \
    --simulate-context-len "$isl" --min-cache-tokens "$min_cache" \
    --max-new-tokens 8 --batch-size "$batch" \
    --mesh-rows 4 --mesh-cols 8 --kv-cache-dtype bf8 \
    --phase both --enable-trace --trace-mode sampling \
    > "$logfile" 2>&1
  local ec=$?
  set -e
  record "$FLASH_SUMMARY" "$tag" flash "$isl" "$batch" "$mode" "$pcm" "$chunk" "$role" "$logfile" "$ec"
}

run_reap() {
  local isl="$1" batch="$2" mode="$3" pcm="$4" chunk="$5" role="$6" timeout_s="${7:-0}"
  local tag="val_reap_isl${isl}_b${batch}_${mode}_pcm${pcm}_chk${chunk}"
  local logfile="$REAP_OUT/${tag}.log"
  wait_idle
  log "START $tag ($role) timeout=${timeout_s}s"
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
  unset GLM4_MOE_ATTN_PREFILL_REDUCE_IMPL || true
  export GLM4_MOE_MOE_SPARSE_PREFILL_PCM="$pcm"
  export GLM4_MOE_MOE_SPARSE_PREFILL_CHUNK_TOKENS="$chunk"
  export GLM4_MOE_TIME_LAYER=0
  if [[ "$mode" == "batched" ]]; then
    export GLM4_MOE_BATCHED_PREFILL=1
  else
    export GLM4_MOE_BATCHED_PREFILL=0
  fi
  local min_cache=$((isl + 16))
  set +e
  if [[ "$timeout_s" -gt 0 ]]; then
    timeout --signal=TERM --kill-after=60s "$timeout_s" \
      ./python_env/bin/python3 models/experimental/glm4_moe/scripts/debug_run_full_tt_greedy.py \
        --mesh-rows 8 --mesh-cols 4 \
        --model-id cerebras/GLM-4.7-REAP-218B-A32B \
        --simulate-context-len "$isl" --batch-size "$batch" \
        --max-new-tokens 8 --min-cache-tokens "$min_cache" \
        --enable-trace --trace-mode sampling --warmup-decode-trace \
        > "$logfile" 2>&1
  else
    ./python_env/bin/python3 models/experimental/glm4_moe/scripts/debug_run_full_tt_greedy.py \
      --mesh-rows 8 --mesh-cols 4 \
      --model-id cerebras/GLM-4.7-REAP-218B-A32B \
      --simulate-context-len "$isl" --batch-size "$batch" \
      --max-new-tokens 8 --min-cache-tokens "$min_cache" \
      --enable-trace --trace-mode sampling --warmup-decode-trace \
      > "$logfile" 2>&1
  fi
  local ec=$?
  set -e
  local note=""
  [[ "$timeout_s" -gt 0 ]] && note="timeout_budget=${timeout_s}s"
  record "$REAP_SUMMARY" "$tag" reap "$isl" "$batch" "$mode" "$pcm" "$chunk" "$role" "$logfile" "$ec" "$note"
}

mkdir -p "$FLASH_OUT" "$REAP_OUT"
echo "tag,model,isl,batch,mode,pcm,chunk,role,status,prefill_s,decode_mean_ms,detail" > "$FLASH_SUMMARY"
echo "tag,model,isl,batch,mode,pcm,chunk,role,status,prefill_s,decode_mean_ms,detail" > "$REAP_SUMMARY"
: > "$LOG"

log "=== validation start (Flash then REAP) ==="

# --- Flash: 512×8 batched bad vs opt; optional longer cell ---
run_flash 512 8 batched 1 4096 baseline_bad
run_flash 512 8 batched 2 4096 optimal
run_flash 1024 16 batched 1 4096 baseline_pcm1
run_flash 1024 16 batched 2 4096 optimal

# --- REAP: 128×4 clean paired; 512×8 proof; optional serial anti-pattern ---
run_reap 128 4 batched 0 0 baseline_legacy 0
run_reap 128 4 batched 1 4096 optimal 0
run_reap 512 8 batched 1 4096 optimal 0
run_reap 512 8 batched 0 0 baseline_legacy "$REAP_512_PCM0_TIMEOUT"
run_reap 128 4 serial 0 0 baseline_legacy 0
run_reap 128 4 serial 1 4096 chunked_hurts 0

log "=== validation runs done ==="
