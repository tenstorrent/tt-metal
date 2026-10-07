#!/bin/bash
# t136: #134 LTX_SDPA_MM_LOFI e2e A/B on blx03, full 4x8 mesh, production layout (bh_4x8sp1tp0_ring),
# 1920x1088/145f. ONE broker job, two pytest processes on the ~/fasth3/t134 build: arm OFF (knob unset) then
# arm LOFI (LTX_SDPA_MM_LOFI=1). Each is #138's protocol: warmup, gen#0 (cold, trace capture), gen#1 (warm
# replay, fresh prompt = headline). Baseline = #138 job 099 (t48 c4409b1fa24), /var/tmp/fasth3/t138/out.
# Usage (via tmp/blx03/submit.sh): bash /var/tmp/fasth3/t136/run_ab.sh. DRY_RUN=1: print only.
set -o pipefail
W=${W:-/home/smarton/fasth3/t134}
BASE=${BASE:-/home/smarton/fasth3/tt-metal}
V=${V:-$([ -n "$DRY_RUN" ] && echo /tmp/t136_dry || echo /var/tmp/fasth3/t136)}
PYTEST_S=${PYTEST_S:-1500}
ARMS=${ARMS:-OFF LOFI}
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
export LTX_E2E_EXTRA_REPLAYS=0 LTX_FRESH_PROMPTS=1 LTX_TIME_STAGES=1 LTX_CONV3D_BLOCKING_MESH=4,8
export TT_METAL_CACHE=$FASTH3_DATA/cache/tt-metal-cache TT_DIT_CACHE_DIR=$FASTH3_DATA/cache/dit-ltx25
cmd=(python -u -m pytest -sv --timeout=$PYTEST_S
  "models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" -k bh_4x8sp1tp0_ring)
for f in $LTX_CHECKPOINT $LTX25_ROOT $GEMMA_PATH; do
  [ -n "$DRY_RUN" ] || [ -e $f ] || { echo "missing $f"; echo T136_EXIT=3; exit 3; }
done
rc=0
for arm in $ARMS; do
  OUT=$V/out/$(echo $arm | tr A-Z a-z); mkdir -p $OUT
  export LTX_OUT_DIR=$OUT
  if [ $arm = LOFI ]; then export LTX_SDPA_MM_LOFI=1; else unset LTX_SDPA_MM_LOFI; fi
  {
    echo "[t136] arm=$arm host=$(hostname) boot=$(uptime -s) tree=$W commit=$(git -C $W rev-parse HEAD) $(date -u '+%F %T')"
    env | grep -E '^(LTX|TT_DIT|TT_METAL_CACHE|NO_PROMPT|SEED|RUN_|GEMMA)' | sort
    echo "${cmd[*]}"
  } | tee $OUT/run.log
  [ -n "$DRY_RUN" ] && continue
  T0=$(date +%s)
  "${cmd[@]}" 2>&1 | tee -a $OUT/run.log; arc=${PIPESTATUS[0]}
  echo "[t136] arm=$arm process wall $(( $(date +%s) - T0 )) s" | tee -a $OUT/run.log
  mv -f ltx_av_fast_*.mp4 $OUT/ 2>/dev/null
  for mp4 in $OUT/*.mp4; do
    [ -e "$mp4" ] && ffmpeg -loglevel error -y -ss 3 -i "$mp4" -frames:v 1 "${mp4%.mp4}_t3s.png"
  done
  grep -E 'E2E_WALL_S|│ (Encoder|Stage|Latent|VAE|Audio|Total)|Video export|Total \(compute\)' $OUT/run.log | tail -30
  echo "T136_ARM_EXIT $arm $arc" | tee -a $OUT/run.log
  [ $arc = 0 ] || { rc=$arc; break; }   # OFF failing means LOFI would too: keep the job short
done
echo "T136_EXIT=$rc"
exit $rc
