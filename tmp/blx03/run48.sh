#!/bin/bash
# t48 integrated LTX-2.5 e2e on blx03: 4x8 ring (full mesh), 1080p/145f, seed 0, gen #0 (capture) + 3 timed gens.
# Usage: bash ~/fasth3/t48/tmp/blx03/run48.sh <label> [VAR=val ...]
# Runs the t48 worktree with its own build (neighbor_pad_async logical_w is C++, so the t36 build cannot run it)
# and the shared tree's venv.
set -o pipefail
W=${W:-/home/smarton/fasth3/t48}
BASE=${BASE:-/home/smarton/fasth3/tt-metal}
OUT=${OUT:-/home/smarton/fasth3/out/t48}
label=$1; shift
cd $W
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools
export LTX_VERSION=2.5 LTX25_DIFFVAE=0
export LTX_CHECKPOINT=/home/smarton/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export LTX25_VIDEO_VAE=$LTX_CHECKPOINT
export LTX25_ROOT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
export FASTH3_DATA=${FASTH3_DATA:-/var/tmp/fasth3}
export GEMMA_PATH=$FASTH3_DATA/models/gemma-3-12b-it-qat-q4_0-unquantized
export NO_PROMPT=1 HF_HUB_OFFLINE=1 SEED=0 RUN_WARMUP=1 LTX_TRACED=1 RUN_VBENCH=0 RUN_CLIP=0
export LTX_E2E_EXTRA_REPLAYS=2 LTX_FRESH_PROMPTS=1 LTX_TIME_STAGES=1
export TT_METAL_CACHE=$FASTH3_DATA/cache/tt-metal-cache TT_DIT_CACHE_DIR=$FASTH3_DATA/cache/dit-ltx25
PYTEST_TIMEOUT=${PYTEST_TIMEOUT:-1500}
for kv in "$@"; do export "$kv"; done
mkdir -p $OUT/$label
export LTX_OUT_DIR=$OUT/$label
echo "[run48] host=$(hostname) tree=$W commit=$(git -C $W rev-parse --short HEAD) base=$(git -C $BASE rev-parse --short HEAD) label=$label fold=${LTX_VAE_FOLD_TIME_PAD:-1} wmask=${LTX_VAE_FOLD_W_MASK:-1}"
(while true; do echo "[hb] $(date +%T)"; sleep 45; done) & HB=$!
python -u -m pytest -sv --timeout=$PYTEST_TIMEOUT \
  "models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" \
  -k bh_4x8sp1tp0_ring 2>&1 | tee $OUT/$label/run.log
rc=$?
kill $HB
mv -f ltx_av_fast_*.mp4 $OUT/$label/ 2>/dev/null
echo "RUN_EXIT[$label]=$rc"
exit $rc
