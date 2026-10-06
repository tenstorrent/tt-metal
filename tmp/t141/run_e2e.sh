#!/bin/bash
# t141: 5-seed confirmation of #138's warm 4x8 LTX-2.5 1080p/145f e2e on blx03, t48 DEFAULT config, in ONE
# pytest process: warmup, gen#0 (seed 0, DEFAULT_LTX_PROMPT, captures traces), then one warm pure-replay gen
# per seed 0..4 (gen#1..#5, LTX_E2E_SEEDS, test knob at 9f2b28b7663 = c4409b1fa24 + test-only change).
# LTX_FRESH_PROMPTS=1: gen#k uses FRESH_LTX_PROMPTS[(k-1)%3]; gen#1 (seed 0, "red paper boat") matches
# #138 gen#1 and the t20 baseline gen#1. Usage: W=<t48 tree on blx03> bash <copy of this script>. DRY_RUN=1: print only.
set -o pipefail
W=${W:-/home/smarton/fasth3/t48}
BASE=${BASE:-/home/smarton/fasth3/tt-metal}
OUT=${OUT:-$([ -n "$DRY_RUN" ] && echo /tmp/t141_dry || echo /var/tmp/fasth3/t141/out)}
PYTEST_S=${PYTEST_S:-1500}
cd $W
[ -z "$DRY_RUN" ] && source $BASE/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools
export LTX_VERSION=2.5 LTX25_DIFFVAE=0
export LTX_CHECKPOINT=/home/smarton/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export LTX25_VIDEO_VAE=$LTX_CHECKPOINT
export LTX25_ROOT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
export FASTH3_DATA=${FASTH3_DATA:-/var/tmp/fasth3}
export GEMMA_PATH=$FASTH3_DATA/models/gemma-3-12b-it-qat-q4_0-unquantized
export NO_PROMPT=1 HF_HUB_OFFLINE=1 SEED=0 RUN_WARMUP=1 LTX_TRACED=1 RUN_VBENCH=0 RUN_CLIP=0
export LTX_E2E_EXTRA_REPLAYS=0 LTX_E2E_SEEDS=0,1,2,3,4 LTX_FRESH_PROMPTS=1 LTX_TIME_STAGES=1 LTX_CONV3D_BLOCKING_MESH=4,8
export TT_METAL_CACHE=$FASTH3_DATA/cache/tt-metal-cache TT_DIT_CACHE_DIR=$FASTH3_DATA/cache/dit-ltx25
export LTX_OUT_DIR=$OUT
mkdir -p $OUT
cmd=(python -u -m pytest -sv --timeout=$PYTEST_S
  "models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" -k bh_4x8sp1tp0_ring)
{
  echo "[t141] host=$(hostname) boot=$(uptime -s) tree=$W commit=$(git -C $W rev-parse HEAD) $(date -u '+%F %T')"
  env | grep -E '^(LTX|TT_DIT|TT_METAL_CACHE|NO_PROMPT|SEED|RUN_|GEMMA)' | sort
  echo "${cmd[*]}"
} | tee $OUT/run.log
if [ -n "$DRY_RUN" ]; then echo "T141_EXIT=0"; exit 0; fi
for f in $LTX_CHECKPOINT $LTX25_ROOT $GEMMA_PATH; do [ -e $f ] || { echo "missing $f" | tee -a $OUT/run.log; echo T141_EXIT=3; exit 3; }; done
T0=$(date +%s)
"${cmd[@]}" 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[t141] process wall $(( $(date +%s) - T0 )) s" | tee -a $OUT/run.log
mv -f ltx_av_fast_*.mp4 $OUT/ 2>/dev/null
for mp4 in $OUT/*.mp4; do
  [ -e "$mp4" ] && ffmpeg -loglevel error -y -ss 3 -i "$mp4" -frames:v 1 "${mp4%.mp4}_t3s.png"
done
grep -E 'E2E_WALL_S|│ (Encoder|Stage|Latent|VAE|Audio|Total)|Video export|Total \(compute\)' $OUT/run.log | tail -80
echo "T141_EXIT=$rc" | tee -a $OUT/run.log
exit $rc
