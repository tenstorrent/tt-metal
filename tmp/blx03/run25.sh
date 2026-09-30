#!/bin/bash
# LTX-2.5 1080p run on blx03 (g14blx03). Usage: tmp/blx03/run25.sh <label> [VAR=val ...]
# Labels: dv145 (DiffVAE 145f@24), conv145 (LTX25_DIFFVAE=0), dv145_c211 (DIFFVAE_NA_CHUNK_BRICKS=2,1,1 DIFFVAE_NA_UNSAFE_CHUNK=1).
# W=<worktree> runs another tree with its own build; outputs land in $OUT/<label>.
set -o pipefail
W=${W:-/home/smarton/fasth3/tt-metal}
OUT=${OUT:-/home/smarton/fasth3/out/ltx25_1080p_6s}
label=$1; shift
cd $W
source /home/smarton/fasth3/tt-metal/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools
export LTX_VERSION=2.5 LTX25_DIFFVAE=1
export LTX_CHECKPOINT=/home/smarton/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export LTX25_VIDEO_VAE=$LTX_CHECKPOINT
export LTX25_ROOT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
export GEMMA_PATH=/home/smarton/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/68f7ee4fbd59087436ada77ed2d62f373fdd4482
export NO_PROMPT=1 HF_HUB_OFFLINE=1 SEED=0 RUN_WARMUP=1 LTX_TRACED=1 RUN_VBENCH=0 RUN_CLIP=0
export LTX_E2E_EXTRA_REPLAYS=0 LTX_FRESH_PROMPTS=1 LTX_TIME_STAGES=1 DIFFVAE_MEM_LOG=1
# All caches under ~/fasth3 (house rule on blx03).
export TT_METAL_CACHE=/home/smarton/fasth3/cache/tt-metal-cache TT_DIT_CACHE_DIR=/home/smarton/fasth3/cache/dit-ltx25
PYTEST_TIMEOUT=${PYTEST_TIMEOUT:-580}
for kv in "$@"; do export "$kv"; done
mkdir -p $OUT/$label
export LTX_OUT_DIR=$OUT/$label
echo "[run25] host=$(hostname) tree=$W commit=$(git -C $W rev-parse --short HEAD) label=$label"
(while true; do echo "[hb] $(date +%T)"; sleep 45; done) & HB=$!
python -u -m pytest -sv --timeout=$PYTEST_TIMEOUT \
  "models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" \
  -k bh_4x8sp1tp0_ring 2>&1 | tee $OUT/$label/run.log
rc=$?
kill $HB
mv -f ltx_av_fast_*.mp4 $OUT/$label/ 2>/dev/null
echo "RUN_EXIT[$label]=$rc"
exit $rc
