#!/bin/bash
# One eval-pack config as one blx03 broker job: 4x8 ring, 1088x1920/145f, traced, SEED=0, fresh prompts.
# One pytest process: pipeline warmup, gen#0 (trace capture), gen#1 (warm replay = headline, same protocol
# as t138 job 099), gen#2 (one more warm replay for run-to-run spread).
# Python comes from the staged overlay $S (this branch); C++ build and kernels from $B, which has the same C++.
# Usage: bash run_cfg.sh <label> [VAR=val ...]. LTX_E2E_SEEDS=0,1,2,3,4 in the args gives one warm gen per seed.
set -o pipefail
BASE=/home/smarton/fasth3/tt-metal; B=${B:-/home/smarton/fasth3/t48}; V=/var/tmp/fasth3/t140; S=$V/src
label=$1; shift
OUT=$V/$label
mkdir -p $OUT
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1
export LTX_VERSION=2.5 LTX25_DIFFVAE=0
export LTX_CHECKPOINT=/home/smarton/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export LTX25_VIDEO_VAE=$LTX_CHECKPOINT
export LTX25_ROOT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
export GEMMA_PATH=/var/tmp/fasth3/models/gemma-3-12b-it-qat-q4_0-unquantized
export NO_PROMPT=1 SEED=0 RUN_WARMUP=1 LTX_TRACED=1 RUN_VBENCH=0 RUN_CLIP=0
export LTX_E2E_EXTRA_REPLAYS=1 LTX_FRESH_PROMPTS=1 LTX_TIME_STAGES=1 LTX_CONV3D_BLOCKING_MESH=4,8 DIFFVAE_MEM_LOG=1
export TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache TT_DIT_CACHE_DIR=/var/tmp/fasth3/cache/dit-ltx25
for kv in "$@"; do export "$kv"; done
export LTX_OUT_DIR=$OUT
cd $S
{
  echo "[t140] label=$label flags=$* host=$(hostname) boot=$(uptime -s) build=$(git -C $B rev-parse --short=11 HEAD) src=$(cat $S/REV) $(date -u '+%F %T')"
  env | grep -E '^(LTX|TT_DIT|TT_METAL_CACHE|NO_PROMPT|SEED|RUN_|GEMMA|DIFFVAE)' | sort
} | tee $OUT/run.log
test -f $B/ttnn/ttnn/_ttnn.so || { echo "[t140] no build at $B" | tee -a $OUT/run.log; echo "T140_EXIT[$label]=4" | tee -a $OUT/run.log; exit 4; }
for f in $LTX_CHECKPOINT $LTX25_ROOT $GEMMA_PATH; do
  [ -e $f ] || { echo "missing $f" | tee -a $OUT/run.log; echo "T140_EXIT[$label]=3" | tee -a $OUT/run.log; exit 3; }
done
T0=$(date +%s)
timeout 1500 python -u -m pytest -c $S/pytest.ini --rootdir=$S -sv --timeout=1450 \
  "models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" \
  -k bh_4x8sp1tp0_ring 2>&1 | tee -a $OUT/run.log
rc=${PIPESTATUS[0]}
echo "[t140] process wall $(( $(date +%s) - T0 )) s" | tee -a $OUT/run.log
mv -f $S/ltx_av_fast_*.mp4 $OUT/ 2>/dev/null
for mp4 in $OUT/*.mp4; do
  [ -e "$mp4" ] && ffmpeg -loglevel error -y -ss 3 -i "$mp4" -frames:v 1 "${mp4%.mp4}_t3s.png"
done
echo "T140_EXIT[$label]=$rc" | tee -a $OUT/run.log
exit $rc
