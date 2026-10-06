#!/bin/bash
# g15blx02 copy of tmp/t138/run_e2e.sh (blx03 job 099): warm 4x8 1080p 6s e2e, seed 0, default t48 config.
# Differences: tree and data live under ~/fasth3/tt-metal (charter); the 2.3 VAE comes from the NFS
# ltx-2.3-22b-dev file, whose 170 vae.* tensors and VAE config are byte-identical to distilled-1.1's.
W=${W:-/home/smarton/fasth3/tt-metal/tt-project/worktrees/t158}; BASE=${BASE:-/home/smarton/fasth3/tt-metal}
DATA=${DATA:-/home/smarton/fasth3/tt-metal/tt-project/data/g15}
OUT=${OUT:-$DATA/out}; PYTEST_S=${PYTEST_S:-3000}
cd $W; source $BASE/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools
export LTX_VERSION=2.5 LTX25_DIFFVAE=0
export LTX_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/7caa482d5cd10a2eae6b34cb48f093ebc45a263e/ltx-2.3-22b-dev.safetensors
export LTX25_VIDEO_VAE=$LTX_CHECKPOINT
export LTX25_ROOT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
export NO_PROMPT=1 HF_HUB_OFFLINE=1 SEED=0 RUN_WARMUP=1 LTX_TRACED=1 RUN_VBENCH=0 RUN_CLIP=0
export LTX_E2E_EXTRA_REPLAYS=0 LTX_FRESH_PROMPTS=1 LTX_TIME_STAGES=1 LTX_CONV3D_BLOCKING_MESH=4,8
export TT_METAL_CACHE=$DATA/tt-metal-cache TT_DIT_CACHE_DIR=$DATA/dit-ltx25
export LTX_OUT_DIR=$OUT
mkdir -p $OUT
cmd=(python -u -m pytest -sv --timeout=$PYTEST_S
  "models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" -k bh_4x8sp1tp0_ring)
{
  echo "[t158] host=$(hostname) boot=$(uptime -s) tree=$W commit=$(git -C $W rev-parse HEAD) $(date -u '+%F %T')"
  env | grep -E '^(LTX|TT_DIT|TT_METAL_CACHE|NO_PROMPT|SEED|RUN_)' | sort
  echo "${cmd[*]}"
} | tee $OUT/run.log
if [ -n "$DRY_RUN" ]; then echo "T158_EXIT=0"; exit 0; fi
for f in $LTX_CHECKPOINT $LTX25_ROOT $TT_DIT_CACHE_DIR; do [ -e $f ] || { echo "missing $f" | tee -a $OUT/run.log; echo T158_EXIT=3; exit 3; }; done
T0=$(date +%s)
"${cmd[@]}" 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[t158] process wall $(( $(date +%s) - T0 )) s" | tee -a $OUT/run.log
mv -f ltx_av_fast_*.mp4 $OUT/ 2>/dev/null
for mp4 in $OUT/*.mp4; do
  [ -e "$mp4" ] && ffmpeg -loglevel error -y -ss 3 -i "$mp4" -frames:v 1 "${mp4%.mp4}_t3s.png"
done
grep -E 'E2E_WALL_S|│ (Encoder|Stage|Latent|VAE|Audio|Total)|Video export|Total \(compute\)' $OUT/run.log | tail -30
echo "T158_EXIT=$rc" | tee -a $OUT/run.log
exit $rc
