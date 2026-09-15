#!/bin/bash
# T2.5 model-level tracy ops reports (Llama 3.1 8B attention module with real weights, one layer), production configs.
set -e
# Path resolution for both layouts (workspace or $TTM/analysis/campaigns): see campaign_paths.sh.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$SD/campaign_paths.sh"
ML=${ML:-$HANDOFF/data/model_level}   # in a checkout HANDOFF is DD, so ML lands under the invocation directory
HF_MODEL=${HF_MODEL:-/proj_sw/user_dev/llama31-8b-data/Llama-3.1-8B-Instruct}   # shared read-only dataset, not part of the workspace
cd $TTM; export TT_METAL_HOME=$TTM ARCH_NAME=blackhole HF_MODEL=$HF_MODEL MESH_DEVICE=N150 TT_CACHE_PATH=${TT_CACHE_PATH:-$ML/tt_cache}
source $PYENV
cd $DD && $SD/set_zone_config.sh 0 0 0 0 0 >/dev/null && cd $TTM
run(){ # tag, test target, env already exported
  T0=$(date -u +%FT%TZ); rm -rf generated/profiler/reports/*; rm -f generated/profiler/.logs/profile_log_device.csv
  echo "### $1 start $T0 sha=$(git rev-parse --short HEAD) env: $(env | grep -E '^(ATTN_|HF_MODEL)' | tr '\n' ' ')"
  python -m tracy -r -m pytest $2 -s > $ML/$1.log 2>&1 || true
  R=$(ls -d generated/profiler/reports/*/ 2>/dev/null | sort | tail -1)
  if [ -z "$R" ] || ! ls $R/ops_perf_results*.csv >/dev/null 2>&1; then echo "RUN FAILED $1 (no ops report)"; grep -E "Error|FAILED|Traceback" $ML/$1.log | head -5; return 0; fi
  cp $R/ops_perf_results*.csv $ML/$1_ops_perf_results.csv; cp generated/profiler/.logs/profile_log_device.csv $ML/$1_profile_log_device.csv 2>/dev/null || true
  FW=$(grep -m1 -o "firmware bundle version: [0-9a-z.-]*" $ML/$1.log | sed "s/firmware bundle version: //")
  echo "PROVENANCE: p100a fw_bundle=$FW; tt-metal $(git rev-parse HEAD) branch $(git branch --show-current) (kernels 72620d5 + zone patch compiled out, SDPA_ZONES=0); $T0; cmd: HF_MODEL=$HF_MODEL MESH_DEVICE=N150 $(env | grep -E '^ATTN_' | tr '\n' ' ') python -m tracy -r -m pytest $2 -s; ops report copied from $R" > $ML/$1_PROVENANCE.txt
  echo "### $1 done $(date -u +%H:%M:%SZ)"
}
export ATTN_ITERS=3 ATTN_SKIP_REF=1 ATTN_PAGED=1
for S in 1024 4096 8192; do ATTN_SEQ=$S run llama8b_attn_prefill_S$S analysis/attn_prefill_perf.py::test_attn_prefill_perf; done
ATTN_BATCH=32 ATTN_POS=128,1024,4096 ATTN_MAXSEQ=8192 run llama8b_attn_decode_b32_pos128_1024_4096 analysis/attn_decode_perf.py::test_attn_decode_perf
echo "### T2.5 done $(date -u +%H:%M:%SZ)"
