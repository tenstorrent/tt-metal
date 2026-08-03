#!/bin/bash
# Stage A (corrected) — decode config accuracy sweep via TEACHER top1 on the FULL multichip model.
# Single-chip layer PCC is NOT a valid discriminator (uniformly degraded by any SDPA program config).
# Teacher top1 on the 40-layer multichip model IS: k128 -> 0.58 (lossy, recorded); k64 -> expect 0.95.
# Weight cache DISABLED (unvalidated -> must not taint an accuracy measurement). k64 first = headline.
set +e
cd /tmp
export TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal
export PYTHONPATH=/home/ttuser/dev/tt-metal
export TT_LAGUNA_WEIGHT_CACHE_DISABLE=1
BASE=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1
LOG=$BASE/doc/vllm_integration/decode_config_sweep/teacher_sweep.log   # <-- TAIL THIS
PY=/home/ttuser/.tenstorrent-venv/bin/python
: > "$LOG"
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
run(){ # $1 label
  log ">>> TEACHER: $1  (K=${TT_LAGUNA_DECODE_K} EXP=${TT_LAGUNA_DECODE_EXP} SDPA_PC=${TT_LAGUNA_DECODE_SDPA_PC})"
  $PY -u "$BASE/tests/full_model_checks.py" teacher 2>&1 | tee -a "$LOG" | grep -E "TEACHER|AGGREGATE|Error|error|Traceback"
  log "<<< done $1"
  tt-smi -r all >/dev/null 2>&1; sleep 8
}
export TT_LAGUNA_DECODE_SDPA_PC=1 TT_LAGUNA_DECODE_EXP=0
export TT_LAGUNA_DECODE_K=64; run "k64 SHIPPED default"
export TT_LAGUNA_DECODE_K=32; run "k32 sweep-alternative"
log "=== STAGE A TEACHER SWEEP DONE (k64 is headline; compare top1) ==="
