#!/bin/bash
# t142 on g15blx02: 5-seed confirm with the t166 warmup cuts, LTX_FRESH_PROMPTS=0 (every gen uses DEFAULT_LTX_PROMPT, pairs with ref_dv145/seed<N>.mp4).
# Same protocol as t158 job 399: ONE pytest process, warmup, gen#0 (seed 0, captures), then one warm replay gen
# per LTX_E2E_SEEDS entry (gen#1 = seed 0, comparable to job 399's _1.mp4).
# Python overlay $T/tree: hardlinked t158 (bf7db12a14) models/ with the t166 python files swapped in;
# TT_METAL_HOME stays t158 so its warm JIT cache (data/g15/tt-metal-cache) is reused. No new caches.
set -o pipefail
P=/home/smarton/fasth3/tt-metal/tt-project; BASE=/home/smarton/fasth3/tt-metal
W=$P/worktrees/t158; DATA=$P/data/g15; T=$DATA/t166; O=$T/tree
TAG=${TAG:-a}; OUT=$DATA/t142/out/$TAG; PYTEST_S=${PYTEST_S:-540}
mkdir -p $OUT; cd $O || { echo T142_EXIT=3; exit 3; }
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$O:$W:$W/ttnn:$W/tools
export LTX_VERSION=2.5 LTX25_DIFFVAE=0
export LTX_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/7caa482d5cd10a2eae6b34cb48f093ebc45a263e/ltx-2.3-22b-dev.safetensors
export LTX25_VIDEO_VAE=$LTX_CHECKPOINT
export LTX25_ROOT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
export NO_PROMPT=1 HF_HUB_OFFLINE=1 SEED=0 RUN_WARMUP=1 LTX_TRACED=1 RUN_VBENCH=0 RUN_CLIP=0
export LTX_E2E_EXTRA_REPLAYS=0 LTX_FRESH_PROMPTS=0 LTX_TIME_STAGES=1 LTX_CONV3D_BLOCKING_MESH=4,8
export LTX_E2E_SEEDS=${LTX_E2E_SEEDS:-0,1,2,3,4}
export LTX_WARMUP_T2V_ONLY=${LTX_WARMUP_T2V_ONLY:-1} LTX_WARMUP_ENCODERS=${LTX_WARMUP_ENCODERS:-0}
export TT_METAL_CACHE=$DATA/tt-metal-cache TT_DIT_CACHE_DIR=$DATA/dit-ltx25
export LTX_OUT_DIR=$OUT
cmd=(python -u -m pytest -sv -p no:cacheprovider --timeout=$PYTEST_S
  "models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" -k bh_4x8sp1tp0_ring)
{
  echo "[t142] tag=$TAG host=$(hostname) boot=$(uptime -s) tree=$W commit=$(git -C $W rev-parse HEAD) $(date -u '+%F %T') UTC"
  echo "[t142] overlay=$(cat $O/OVERLAY_COMMIT) md5=$(md5sum $O/models/tt_dit/pipelines/ltx/pipeline_ltx_distilled.py | cut -c1-12)"
  env | grep -E '^(LTX|TT_DIT|TT_METAL_CACHE|NO_PROMPT|SEED|RUN_|PYTHONPATH)' | sort
  echo "${cmd[*]}"
} | tee $OUT/run.log
for f in $LTX_CHECKPOINT $LTX25_ROOT $TT_DIT_CACHE_DIR $O/conftest.py; do [ -e $f ] || { echo "missing $f" | tee -a $OUT/run.log; echo T142_EXIT=3; exit 3; }; done
T0=$(date +%s)
"${cmd[@]}" 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[t142] process wall $(( $(date +%s) - T0 )) s" | tee -a $OUT/run.log
mv -f ltx_av_fast_*.mp4 $OUT/ 2>/dev/null
for mp4 in $OUT/*.mp4; do
  [ -e "$mp4" ] && ffmpeg -loglevel error -y -ss 3 -i "$mp4" -frames:v 1 "${mp4%.mp4}_t3s.png"
done
grep -E 'E2E_WALL_S|done in|JIT cache stats|LTX_WARMUP_T2V_ONLY' $OUT/run.log | tail -30
echo "T142_EXIT=$rc" | tee -a $OUT/run.log
exit $rc
