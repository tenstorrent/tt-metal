#!/bin/bash
# job.sbatch: one self-ending exabox batch job on one dit BH Galaxy (full 4x8 mesh):
# build tt-metal if missing, then warm LTX-2.5 1080p 6s e2e (gen#0 cold/trace capture, gen#1 warm), seed 0,
# default config. Submit only through submit.sh (it enforces the idle >= 2 h rule).
#SBATCH -J smarton-fasth3-ltx25
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -o /data/smarton/fasth3/t160/slurm-%j.out
set -eo pipefail
F=/data/smarton/fasth3; W=$F/t48
OUT=${OUT:-$F/t160/out-$SLURM_JOB_ID}; mkdir -p $OUT
RC=$F/t160/job-$SLURM_JOB_ID.rc
trap 'rc=$?; pkill -P $$ 2>/dev/null; echo "JOB_RC=$rc $(date -u +%FT%TZ)" > $RC' EXIT
log() { echo "[$(date -u +%FT%TZ)] $*"; }
log "start node=$(hostname) job=$SLURM_JOB_ID"
log "devices: $(ls /dev/tenstorrent 2>/dev/null | wc -l)"
[ "$(ls /dev/tenstorrent | wc -l)" -ge 32 ]
avail=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
log "/data free ${avail}G"; [ "$avail" -ge 300 ]
cd $W; [ "$(git rev-parse HEAD)" = bf7db12a149bc1e8bf0559350ec2c874e86dc431 ]
if [ ! -f ttnn/ttnn/_ttnn.so ]; then
  log "build start"
  bash build_metal.sh --release --cpm-source-cache $F/cpm-cache
  log "build done"
fi
[ -f ttnn/ttnn/_ttnn.so ]
export LTX_VERSION=2.5 LTX25_DIFFVAE=0
export LTX_CHECKPOINT=/mnt/models/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors
export LTX25_VIDEO_VAE=$LTX_CHECKPOINT LTX25_ROOT=$F/models/ltx-2.5
export GEMMA_PATH=/mnt/models/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/68f7ee4fbd59087436ada77ed2d62f373fdd4482
export NO_PROMPT=1 HF_HUB_OFFLINE=1 SEED=${SEED:-0} RUN_WARMUP=1 LTX_TRACED=1 RUN_VBENCH=0 RUN_CLIP=0
export LTX_E2E_EXTRA_REPLAYS=0 LTX_FRESH_PROMPTS=1 LTX_TIME_STAGES=1 LTX_CONV3D_BLOCKING_MESH=4,8
export TT_METAL_CACHE=$F/cache/tt-metal-cache TT_DIT_CACHE_DIR=$F/cache/dit-ltx25
export LTX_OUT_DIR=$OUT TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools
export PATH=$F/python_env/bin:$PATH
log "e2e start"
python -u -m pytest -sv --timeout=${PYTEST_S:-6000} \
  "models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" -k bh_4x8sp1tp0_ring \
  2>&1 | tee $OUT/e2e.log
log "e2e done"
mv -f $W/ltx_av_fast_*.mp4 $OUT/ 2>/dev/null || true
grep -E 'E2E_WALL_S|stage|gen#|PASSED|FAILED' $OUT/e2e.log | tail -40 > $OUT/timings.txt || true
du -sh $F/cache/* > $OUT/cache_du.txt 2>/dev/null || true
log "end"
