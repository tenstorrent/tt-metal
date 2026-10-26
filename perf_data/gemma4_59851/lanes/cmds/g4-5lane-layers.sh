#!/bin/bash
# Gemma4 D_P per-layer profile at 3/4/5/6 lanes of 2K, one request (#59851 follow-up).
# test_prefill_layer_perf_chunk_n "chunkci" cells under tracy, at a 240K context that 6144/8192/10240/12288 all divide,
# plus 8192 at 256K to cross-check the earlier c04u02 run. Needs the worktree's added sizes (10240, 6144, 12288) and
# context (245760) in demo/text_demo_prefill.py. --topo galaxy.
# Output: /data/mmanzoor/jobs/run/$SLURM_JOB_ID/layer_perf/sz<C>_ctx<k>/ (tracy -o dirs); parse on the login node with
# /data/mmanzoor/matmul_chunk_sweep_c04u02/layer_shares.py.
cd "$TT_METAL_HOME"
step() { local n=$1; shift; echo "=== START $n $(date +%T)"; local t=$SECONDS; "$@"; local rc=$?;
         echo "=== END $n rc=$rc $((SECONDS-t))s"; return $rc; }

export HF_HUB_OFFLINE=1
export HF_HOME=/mnt/models/huggingface
export HF_MODEL=google/gemma-4-31B-it
export TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
OUT=/data/$USER/jobs/run/${SLURM_JOB_ID:-manual}/layer_perf
export PREFILL_SUMMARIES=$OUT/summaries
mkdir -p "$PREFILL_SUMMARIES"
T=models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n

layer() {  # <chunk> <ctx id, e.g. 240k>
  local size=$1 ctx=$2 dir=$OUT/sz${1}_ctx${2}
  mkdir -p "$dir"
  timeout 45m python -m tracy -p -r -v -o "$dir" --check-exit-code --op-support-count 20000 \
    -m "pytest ${T}[blackhole-chunkci-both-sz${size}-ctx_${ctx}-8x4] -vv --tb=short --timeout=2400" > "$dir/run.log" 2>&1
  local rc=$?
  grep -E "passed|failed|skipped|error" "$dir/run.log" | tail -2
  if [ $rc -eq 124 ]; then echo "=== TIMEOUT sz$size: resetting the galaxy"; tt-smi -glx_reset; fi
  return $rc
}

step lanes4_240k layer 8192 240k
step lanes5_240k layer 10240 240k
step lanes3_240k layer 6144 240k
step lanes6_240k layer 12288 240k
step lanes4_256k layer 8192 256k
echo "=== ALL DONE $(date +%T)"
