#!/bin/bash
# t301: the repo's standard LTX e2e test, unmodified, 8+3 (each tree's default), one arm per broker job.
#   main: origin/main 80b1cd689d0, t293's lean build.   pr: ttp/ltx23-main-pr df9e5ecaac6, own lean build.
# Both: LTX-2.3 22B distilled 1.1, bf16 default, 1088x1920, 145 frames, seed 10, 4x8 ring nl2 fsdp0
#   (main id 4x8sp1tp0nl2_ring_is_fsdp0 = ltx-rt id bh_4x8sp1tp0_ring),
# TT_DIT_CACHE_DIR unset (loads from the local checkpoint, writes no DiT cache), RUN_VBENCH=0 RUN_CLIP=0.
# Warm table = gen #2 (pure replay). Usage (broker -t 570): run301.sh main|pr
set -o pipefail
A=${1:?main|pr}; F=/var/tmp/fasth3; D=$F/t301; OUT=$D/out_$A
case $A in
  main) W=$F/t293/main; WANT=80b1cd689d0;;
  pr)   W=$D/pr; WANT=df9e5ecaac6;;
  *) echo "bad arm $A"; exit 2;;
esac
rm -rf $OUT; mkdir -p $OUT $D/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$D/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=/home/sulphur/hf
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$D/jit-$A
export LTX_CHECKPOINT=$F/models/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export GEMMA_PATH=$F/models/gemma-3-12b-it-qat-q4_0-unquantized
export RUN_VBENCH=0 RUN_CLIP=0 LTX_OUT_DIR=$OUT
unset TT_DIT_CACHE_DIR LTX_FUSE_GATE_ON_DEVICE LTX_QUANT LTX_QUANT_ACTIVATIONS LTX_QUALITY LTX_FAST LTX_S1_SIGMAS LTX_S2_SIGMAS \
  LTX_TRACED LTX_VERSION LTX_ITER_ENV NO_PROMPT SEED RUN_WARMUP PROMPT OUTPUT_PATH LTX_E2E_SEEDS LTX_E2E_EXTRA_REPLAYS LTX_FRESH_PROMPTS
cd $OUT || exit 3
T=$W/models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py
HEAD=$(git -C $W rev-parse --short=11 HEAD 2>/dev/null)
{ echo "[t301] arm=$A host=$(hostname) commit=$HEAD dirty=$(git -C $W status --porcelain -uno | wc -l) test_md5=$(md5sum < $T | cut -c1-32) $(date -u '+%F %T') UTC"
  env | grep -E '^(LTX_|TT_|HF_|GEMMA|NUM_FRAMES|HEIGHT|WIDTH|FPS|SEED|RUN_|NO_PROMPT|PYTHONPATH)' | sort; } | tee run.log
[ "$HEAD" = $WANT ] || { echo "wrong commit $HEAD != $WANT" | tee -a run.log; exit 3; }
for f in $W/ttnn/ttnn/_ttnn.so $LTX_CHECKPOINT $GEMMA_PATH $T; do
  [ -r $f ] || { echo "missing $f" | tee -a run.log; exit 3; }
done
T0=$(date +%s)
# pytest runs in its own process group; any exit or broker kill takes the whole group (no orphan holds the device).
export T W
setsid bash -c 'python -u -m pytest -c $W/pytest.ini --rootdir=$W -sv -p no:cacheprovider --timeout=540 "$T::test_pipeline_distilled" -k 4x8sp1tp0nl2_ring_is_fsdp0 2>&1 | tee -a run.log; exit ${PIPESTATUS[0]}' &
PG=$!
trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
trap 'exit 143' TERM INT
wait $PG; rc=$?
echo "[t301] process wall $(( $(date +%s) - T0 )) s" | tee -a run.log
md5sum ltx_av_fast_*.mp4 2>/dev/null | tee -a run.log
echo "T301_EXIT=$rc" | tee -a run.log
exit $rc
