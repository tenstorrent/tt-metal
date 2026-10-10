#!/bin/bash
# t315: the repo's standard LTX-2.3 e2e test, unmodified, on ltx-rt f6547442b30 (own release build), blx03 4x8, full clock.
# 8+3 (the tree's default), LTX-2.3 22B distilled 1.1, 1088x1920, 145 frames, seed 10 (test default), the test's own prompt,
# bh_4x8sp1tp0_ring, traced. TT_DIT_CACHE_DIR unset (loads the local checkpoint, writes no DiT cache), RUN_VBENCH=0 RUN_CLIP=0.
# Own JIT cache under /var/tmp/fasth3/t315 (cold on the first run). Broker: -e env315.yaml -t 570.
if [ -z "$INNER" ]; then
  INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
set -o pipefail
F=/var/tmp/fasth3; D=$F/t315; W=/home/smarton/fasth3/t315; WANT=f6547442b30; OUT=$D/out
# blx03 caps: / at most 85% used.
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 85 ] || { echo "[t315] df / $use% > 85%"; exit 5; }
# This task's footprint (JIT cache + outputs) at most 20G; /home (HOME) keeps 150G free.
gb=$(timeout 120 du -sxBG $D | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 20 ] || { echo "[t315] $D ${gb}G > 20G"; exit 5; }
free=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9); [ "$free" -ge 150 ] || { echo "[t315] /home ${free}G free < 150G"; exit 5; }
rm -rf $OUT; mkdir -p $OUT $D/tmp
export HOME=/home/smarton TMPDIR=$D/tmp HF_HOME=/home/sulphur/hf HF_HUB_OFFLINE=1
source /home/smarton/fasth3/tt-metal/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=$D/jit
export LTX_CHECKPOINT=/home/smarton/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export GEMMA_PATH=$F/models/gemma-3-12b-it-qat-q4_0-unquantized
export RUN_VBENCH=0 RUN_CLIP=0 LTX_OUT_DIR=$OUT
unset TT_DIT_CACHE_DIR LTX_FUSE_GATE_ON_DEVICE LTX_FUSE_NORM_ADALN LTX_QUANT LTX_QUANT_ACTIVATIONS LTX_QUALITY LTX_FAST \
  LTX_S1_SIGMAS LTX_S2_SIGMAS LTX_TRACED LTX_VERSION LTX_ITER_ENV NO_PROMPT SEED RUN_WARMUP PROMPT OUTPUT_PATH \
  LTX_E2E_SEEDS LTX_E2E_EXTRA_REPLAYS LTX_FRESH_PROMPTS LTX_E2E_AB_ENV NUM_FRAMES HEIGHT WIDTH FPS
cd $OUT || exit 3
T=$W/models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py
HEAD=$(git -C $W rev-parse --short=11 HEAD 2>/dev/null)
{ echo "[t315] host=$(hostname) commit=$HEAD dirty=$(git -C $W status --porcelain -uno | wc -l) test_md5=$(md5sum < $T | cut -c1-32) $(date -u '+%F %T') UTC"
  env | grep -E '^(LTX_|TT_|HF_|GEMMA|PYTHONPATH|RUN_)' | sort; } | tee run.log
[ "$HEAD" = $WANT ] || { echo "[t315] wrong commit $HEAD != $WANT" | tee -a run.log; exit 3; }
[ -e $W/tmp/ltx_env_prewarm.yaml ] && { echo "[t315] $W/tmp/ltx_env_prewarm.yaml would change env" | tee -a run.log; exit 3; }
for f in $W/ttnn/ttnn/_ttnn.so $LTX_CHECKPOINT $GEMMA_PATH $T; do
  [ -r $f ] || { echo "[t315] missing $f" | tee -a run.log; exit 3; }
done
T0=$(date +%s)
python -u -m pytest -c $W/pytest.ini --rootdir=$W -sv -p no:cacheprovider --timeout=540 \
  "$T::test_pipeline_distilled[blackhole-bh_4x8sp1tp0_ring-True]" 2>&1 | tee -a run.log
rc=${PIPESTATUS[0]}
echo "[t315] process wall $(( $(date +%s) - T0 )) s" | tee -a run.log
echo "[t315] AICLK clamp warnings: $(grep -c "AICLK failed to settle" run.log)" | tee -a run.log
md5sum ltx_av_fast_*.mp4 2>/dev/null | tee -a run.log
echo "T315_EXIT=$rc" | tee -a run.log
exit $rc
