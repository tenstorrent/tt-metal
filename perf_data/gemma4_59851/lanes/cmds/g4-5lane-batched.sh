#!/bin/bash
# Gemma4 D_P batched prefill (PR #61264) at 5 lanes of 2K vs 4 (#59851 follow-up). Needs the worktree's env hooks in
# tests/test_batched_prefill.py: G4B_LANE_COUNTS (perf test lane counts, default 1,2,4) and G4B_KV_LANES (KV-match
# test lane counts, default 2,4); the "mixed" prefix list is extended to 6 lanes. --topo galaxy.
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

# 1. Correctness: batched KV vs the same requests prefilled alone, 4 and 5 lanes.
step kv_B4_B5 run 30m G4B_KV_LANES=4,5 -- "$F::test_batched_kv_matches_unbatched"
# 2. Step time at 1, 4 and 5 lanes (every mix: shallow, u32k, mixed, deep).
step perf_1_4_5 run 70m G4B_LANE_COUNTS=1,4,5 -- "$F::test_batched_step_perf"
# 3. The rest of the curve: 2, 3 and 6 lanes (6 slots is the adapter's KV-slot limit).
step perf_1_2_3_6 run 70m G4B_LANE_COUNTS=1,2,3,6 -- "$F::test_batched_step_perf"
echo "=== ALL DONE $(date +%T)"
