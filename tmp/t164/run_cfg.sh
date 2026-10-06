#!/bin/bash
# One eval-pack config as one g15blx02 broker job: 4x8 ring, 1088x1920/145f, traced, fresh prompts.
# One pytest process: pipeline warmup, gen#0 (trace capture, default prompt), gen#1 (first warm replay,
# the headline, same protocol as job 399), plus LTX_E2E_EXTRA_REPLAYS more warm replays for spread.
# Python from the t164 worktree (= t48 HEAD models/), C++ build and kernels from the t158 tree (bf7db12a14),
# caches from data/g15 (job 399's): nothing new is built or converted.
# Usage: bash run_cfg.sh <label> [VAR=val ...]   (PYTEST_S sizes the in-process timeout)
set -o pipefail
BASE=/home/smarton/fasth3/tt-metal
W=$BASE/tt-project/worktrees/t158; S=$BASE/tt-project/worktrees/t164; DATA=$BASE/tt-project/data/g15
label=$1; shift
OUT=$DATA/t164/$label; mkdir -p $OUT
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$S:$W/ttnn:$W/tools HF_HUB_OFFLINE=1
export LTX_VERSION=2.5 LTX25_DIFFVAE=0
export LTX_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/7caa482d5cd10a2eae6b34cb48f093ebc45a263e/ltx-2.3-22b-dev.safetensors
export LTX25_VIDEO_VAE=$LTX_CHECKPOINT
export LTX25_ROOT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
export NO_PROMPT=1 SEED=0 RUN_WARMUP=1 LTX_TRACED=1 RUN_VBENCH=0 RUN_CLIP=0
export LTX_E2E_EXTRA_REPLAYS=1 LTX_FRESH_PROMPTS=1 LTX_TIME_STAGES=1 LTX_CONV3D_BLOCKING_MESH=4,8
export TT_METAL_CACHE=$DATA/tt-metal-cache TT_DIT_CACHE_DIR=$DATA/dit-ltx25
for kv in "$@"; do export "$kv"; done
export LTX_OUT_DIR=$OUT
cd $S
{
  echo "[t164] label=$label flags=$* host=$(hostname) boot=$(uptime -s) build=$(git -C $W rev-parse --short=11 HEAD) py=$(git -C $S rev-parse --short=11 HEAD) $(date -u '+%F %T')"
  env | grep -E '^(LTX|TT_DIT|TT_METAL_CACHE|NO_PROMPT|SEED|RUN_)' | sort
} | tee $OUT/run.log
test -f $W/ttnn/ttnn/_ttnn.so || { echo "[t164] no build at $W" | tee -a $OUT/run.log; echo "T164_EXIT[$label]=4" | tee -a $OUT/run.log; exit 4; }
for f in $LTX_CHECKPOINT $LTX25_ROOT $TT_DIT_CACHE_DIR; do
  [ -e $f ] || { echo "missing $f" | tee -a $OUT/run.log; echo "T164_EXIT[$label]=3" | tee -a $OUT/run.log; exit 3; }
done
T0=$(date +%s)
python -u -m pytest -c $S/pytest.ini --rootdir=$S -sv --timeout=${PYTEST_S:-330} \
  "models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" \
  -k bh_4x8sp1tp0_ring 2>&1 | tee -a $OUT/run.log
rc=${PIPESTATUS[0]}
echo "[t164] process wall $(( $(date +%s) - T0 )) s" | tee -a $OUT/run.log
mv -f $S/ltx_av_fast_*.mp4 $OUT/ 2>/dev/null
for mp4 in $OUT/*.mp4; do
  [ -e "$mp4" ] && ffmpeg -loglevel error -y -ss 3 -i "$mp4" -frames:v 1 -q:v 3 "${mp4%.mp4}_t3s.jpg"
done
grep -E 'E2E_WALL_S|Video export' $OUT/run.log | tail -12
echo "T164_EXIT[$label]=$rc" | tee -a $OUT/run.log
exit $rc
