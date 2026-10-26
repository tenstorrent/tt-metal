#!/bin/bash
# Follow-up to g4-5lane-batched.sh (PR #61264 worktree): the 4-lane timing lost when the 1,4,5 capture failed, then 5
# lanes with GEMMA4_ACTIVATIONS_DRAM_ONLY=1 (a 5-lane matmul's CBs clashed with an L1 buffer by ~72 KB). --topo galaxy.
cd "$TT_METAL_HOME"
step() { local n=$1; shift; echo "=== START $n $(date +%T)"; local t=$SECONDS; "$@"; local rc=$?;
         echo "=== END $n rc=$rc $((SECONDS-t))s"; return $rc; }

export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/models/huggingface
export HF_MODEL=google/gemma-4-31B-it
export TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export G4B_REPS=10
F=models/demos/gemma4_d_p/tests/test_batched_prefill.py

# run <timeout> <VAR=value ...> -- <pytest node>; resets the galaxy if the step hangs.
run() {
  local to=$1; shift; local vars=(); while [ "$1" != "--" ]; do vars+=("$1"); shift; done; shift
  env "${vars[@]}" timeout "$to" pytest -sv "$@"; local rc=$?
  if [ $rc -eq 124 ]; then echo "=== TIMEOUT: resetting the galaxy"; tt-smi -glx_reset; fi
  return $rc
}

step perf_1_2_3_4 run 40m G4B_LANE_COUNTS=1,2,3,4 -- "$F::test_batched_step_perf"
step perf_1_4_5_dram run 40m G4B_LANE_COUNTS=1,4,5 GEMMA4_ACTIVATIONS_DRAM_ONLY=1 -- "$F::test_batched_step_perf"
step kv_B5_dram run 30m G4B_KV_LANES=5 GEMMA4_ACTIVATIONS_DRAM_ONLY=1 -- "$F::test_batched_kv_matches_unbatched"
echo "=== ALL DONE $(date +%T)"
