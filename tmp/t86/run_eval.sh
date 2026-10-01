#!/bin/bash
# One eval-pack config on blx03: 4x8 ring, 1080p/145f, gen#0 (capture) + gen#1 + 2 fresh-prompt replays
# + 5 seed replays on the default prompt (same prompt/seeds as ref_dv145).
# Usage (via submit.sh): bash /home/smarton/fasth3/t86/tmp/t86/run_eval.sh <config> [VAR=val ...]
# DRY_RUN=1 prints the environment and the pytest command instead of running it (no device).
set -o pipefail
W=${W:-/home/smarton/fasth3/t86}
BASE=${BASE:-/home/smarton/fasth3/tt-metal}
OUT=${OUT:-/var/tmp/fasth3/eval86}
label=$1; shift
cd $W
[ -n "$DRY_RUN" ] || source $BASE/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools
export LTX_VERSION=2.5 LTX25_DIFFVAE=0
export LTX_CHECKPOINT=/home/smarton/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export LTX25_VIDEO_VAE=$LTX_CHECKPOINT
export LTX25_ROOT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
export FASTH3_DATA=${FASTH3_DATA:-/var/tmp/fasth3}
export GEMMA_PATH=$FASTH3_DATA/models/gemma-3-12b-it-qat-q4_0-unquantized
export NO_PROMPT=1 HF_HUB_OFFLINE=1 SEED=0 RUN_WARMUP=1 LTX_TRACED=1 RUN_VBENCH=0 RUN_CLIP=0
export LTX_E2E_EXTRA_REPLAYS=2 LTX_FRESH_PROMPTS=1 LTX_TIME_STAGES=1 DIFFVAE_MEM_LOG=1
export LTX_SEEDS=0,1,2,3,4
export TT_METAL_CACHE=$FASTH3_DATA/cache/tt-metal-cache TT_DIT_CACHE_DIR=$FASTH3_DATA/cache/dit-ltx25
PYTEST_TIMEOUT=${PYTEST_TIMEOUT:-2400}
for kv in "$@"; do export "$kv"; done
mkdir -p $OUT/$label
export LTX_OUT_DIR=$OUT/$label
cmd=(python -u -m pytest -sv --timeout=$PYTEST_TIMEOUT
  "models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" -k bh_4x8sp1tp0_ring)
echo "[run_eval] host=$(hostname) tree=$W commit=$(git -C $W rev-parse --short HEAD) label=$label flags=$*"
if [ -n "$DRY_RUN" ]; then
  env | grep -E '^(LTX|DIFFVAE|TT_DIT|NO_PROMPT|SEED|RUN_)' | sort
  echo "${cmd[*]}"
  echo "DRY_EXIT[$label]=0"
  exit 0
fi
(while true; do echo "[hb] $(date +%T)"; sleep 45; done) & HB=$!
"${cmd[@]}" 2>&1 | tee $OUT/$label/run.log
rc=$?
kill $HB
mv -f ltx_av_fast_*.mp4 $OUT/$label/ 2>/dev/null
echo "RUN_EXIT[$label]=$rc" | tee -a $OUT/$label/run.log
exit $rc
