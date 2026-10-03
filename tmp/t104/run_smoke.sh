#!/bin/bash
# 10-minute 4x8 e2e smoke of t48 on blx03: baseline then 'all', 1 seed, 1080p/145f, gen#0 (capture) + gen#1 (warm),
# LTX_TIME_STAGES=1, mp4 + still frame per config. Derived from t95's tmp/t86/run_eval.sh.
# Both configs run in ONE broker job (submit.sh timeout 600); each pytest gets its own process so the
# config's env flags apply at import. Budget per config: PER_CFG_S (default 280).
# Usage (via submit.sh): bash /home/smarton/fasth3/t104/tmp/t104/run_smoke.sh
# DRY_RUN=1 prints each config's env and pytest command (no device). DRY_RUN=import also imports the test module.
set -o pipefail
W=${W:-/home/smarton/fasth3/t104}
BASE=${BASE:-/home/smarton/fasth3/tt-metal}
OUT=${OUT:-$([ -n "$DRY_RUN" ] && echo /tmp/smoke104_dry || echo /var/tmp/fasth3/smoke104)}
PER_CFG_S=${PER_CFG_S:-280}
cd $W
[ -z "$DRY_RUN" -o "$DRY_RUN" = import ] && source $BASE/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools
export LTX_VERSION=2.5 LTX25_DIFFVAE=0
export LTX_CHECKPOINT=/home/smarton/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export LTX25_VIDEO_VAE=$LTX_CHECKPOINT
export LTX25_ROOT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
export FASTH3_DATA=${FASTH3_DATA:-/var/tmp/fasth3}
export GEMMA_PATH=$FASTH3_DATA/models/gemma-3-12b-it-qat-q4_0-unquantized
export NO_PROMPT=1 HF_HUB_OFFLINE=1 SEED=0 RUN_WARMUP=1 LTX_TRACED=1 RUN_VBENCH=0 RUN_CLIP=0
export LTX_E2E_EXTRA_REPLAYS=0 LTX_FRESH_PROMPTS=0 LTX_TIME_STAGES=1 LTX_CONV3D_BLOCKING_MESH=4,8
export TT_METAL_CACHE=$FASTH3_DATA/cache/tt-metal-cache TT_DIT_CACHE_DIR=$FASTH3_DATA/cache/dit-ltx25
cmd=(python -u -m pytest -sv --timeout=$PER_CFG_S
  "models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" -k bh_4x8sp1tp0_ring)
echo "[smoke] host=$(hostname) tree=$W commit=$(git -C $W rev-parse --short HEAD)"
rc_all=0
while read -r label flags; do
  mkdir -p $OUT/$label
  (
    for kv in $flags; do export "$kv"; done
    export LTX_OUT_DIR=$OUT/$label
    echo "[smoke] $(date +%T) $label flags=$flags"
    if [ -n "$DRY_RUN" ]; then
      env | grep -E '^(LTX|TT_DIT|NO_PROMPT|SEED|RUN_)' | sort
      echo "${cmd[*]}"
      if [ "$DRY_RUN" = import ]; then
        python -c "import models.tt_dit.tests.models.ltx.test_pipeline_ltx_distilled as m; print('import ok', m.test_pipeline_distilled.__name__)" || exit 1
      fi
      exit 0
    fi
    "${cmd[@]}" 2>&1 | tee $OUT/$label/run.log; rc=$?
    mv -f ltx_av_fast_*.mp4 $OUT/$label/ 2>/dev/null
    for mp4 in $OUT/$label/*.mp4; do
      [ -e "$mp4" ] && ffmpeg -loglevel error -y -ss 2 -i "$mp4" -frames:v 1 "${mp4%.mp4}_still.png"
    done
    exit $rc
  ) 2>&1 | tee -a $OUT/smoke.log
  rc=${PIPESTATUS[0]}
  echo "RUN_EXIT[$label]=$rc" | tee -a $OUT/smoke.log
  [ $rc -ne 0 ] && { rc_all=$rc; break; }  # a failed config may be a drop: never start the next one
done <<'CFG'
baseline
all LTX_FUSE_GATE_ON_DEVICE=1 LTX_FUSE_NORM_ADALN=1 LTX_VAE_CONV_FIDELITY=LoFi
CFG
grep -hE 'TIME_STAGES|stage|wall' $OUT/*/run.log 2>/dev/null | tail -20
echo "SMOKE_DONE rc=$rc_all" | tee -a $OUT/smoke.log
exit $rc_all
