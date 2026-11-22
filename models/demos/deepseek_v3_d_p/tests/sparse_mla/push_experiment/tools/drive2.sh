#!/usr/bin/env bash
# drive.sh <model> <tag> <layers> "<SPARSE_MLA env exports>"   -> $S/runs/<tag>/{launch.log,rt/,summaries/}
set -uo pipefail
S=${PUSH_EXP_DIR:-/tmp/push_exp}
model=$1 tag=$2 layers=$3 cfg=$4
R=$S/runs/$tag; rm -rf "$R"; mkdir -p "$R/rt"
source /home/pjosipovic/tt-metal/python_env/bin/activate; unset TT_METAL_ENV TT_METAL_CCACHE_KERNEL_SUPPORT
export TT_METAL_HOME=/home/pjosipovic/tt-metal TTRUN_DIR=$S/ttop PREFILL_TCP_INTERFACE=${PREFILL_TCP_INTERFACE:-ens5f0np0}
export LOCAL_NUM_LAYERS=$layers LOCAL_CHECK_PCC=1 LOCAL_WARMUP_CHUNKS=${LOCAL_WARMUP_CHUNKS:-2}
export TABLE_WAIT_SECS=${TABLE_WAIT_SECS:-5400} PREFILL_KEEP_RUN_DIR=1
export SCRATCH_RUNNER_ENV="$cfg export SCRATCH_RT_DUMP=$R/rt; export SCRATCH_RT_KEEP=sparse_kv_push,high_bw_all_gather,topk_large_indices,indexer_score,sparse_sdpa;"
export PREFILL_SUMMARIES=$R/summaries/prefill_summaries; mkdir -p "$PREFILL_SUMMARIES"
cd "$TT_METAL_HOME"
exec 9>/tmp/tt-device.lock; flock 9
rm -f /dev/shm/*prefill* 2>/dev/null
echo "=== $tag start $(date -u +%T) cfg: $cfg"
bash "$S/run_local.sh" "$model" sc1 > "$R/launch.log" 2>&1
echo "=== $tag rc=$? end $(date -u +%T)"
