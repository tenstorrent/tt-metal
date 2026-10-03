#!/bin/bash
# blx03 broker job for #96: conv VAE decode rebaseline on the 2x4 submesh of the full mesh (both tests open
# (4,8) and call create_submesh(2,4)), 544x960/145f real 2.5 latent, fused YUV output, production 4x8 conv3d
# blockings (LTX_CONV3D_BLOCKING_MESH=4,8).
#   1) traced vs eager: 1 warmup + 3 eager + capture + 3 traced replays (LTX_VIDEO_VAE_TRACE=1, LTX_TIME_STAGES=1)
#   2) one eager decode under the device profiler (op table)
# Python from the staged overlay $S (stage96.sh); C++ build and kernel sources from $B (B=... to override).
BASE=/home/smarton/fasth3/tt-metal; B=${B:-/home/smarton/fasth3/t48}; V=/var/tmp/fasth3/t96; S=$V/src
PARTS=${T96_PARTS:-12}; LOG=$V/run96${T96_TAG}.log
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1
export LTX_FUSE_YUV_OUTPUT=1 LTX_CONV3D_BLOCKING_MESH=4,8 AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt
export LTX_TRACE_REGION=${LTX_TRACE_REGION:-500000000}
cd $S
echo "[t96] build=$(git -C $B rev-parse --short HEAD) src=$(cat $S/REV) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
test -f "$AB_LATENT" || { echo "[t96] $AB_LATENT missing" | tee -a $LOG; exit 3; }
test -f $B/ttnn/ttnn/_ttnn.so || { echo "[t96] no build at $B" | tee -a $LOG; exit 4; }

rc1=0
if [[ $PARTS == *1* ]]; then
echo "[t96] part 1: traced vs eager" | tee -a $LOG
TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache LTX_TIME_STAGES=1 AB_OUT_DIR=$V \
  timeout 900 python -m pytest -c $S/pytest.ini --rootdir=$S -sv --timeout=860 \
  models/tt_dit/tests/models/ltx/test_vae_ltx_trace_ab.py 2>&1 | tee -a $LOG
rc1=${PIPESTATUS[0]}
echo "T96_PART1_EXIT=$rc1" | tee -a $LOG
fi

echo "[t96] part 2: profiled eager decode" | tee -a $LOG
# Profiler kernel builds go to a throwaway JIT dir so the shared cache does not grow.
# LTX_PIN_CORES=0: the pinning re-exec rebuilds argv as "python pytest ...", which fails under tracy -m.
mkdir -p $V/jit $V/prof
LTX_PIN_CORES=0 TT_METAL_CACHE=$V/jit TT_METAL_PROFILER_DIR=$V/prof TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
  timeout 600 python -m tracy -p -r -v -o $V/prof -m pytest -c $S/pytest.ini --rootdir=$S -sv --timeout=560 \
  models/tt_dit/tests/models/ltx/test_vae_ltx_prof_2x4.py 2>&1 | tee -a $LOG
rc2=${PIPESTATUS[0]}
echo "T96_PART2_EXIT=$rc2" | tee -a $LOG

python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
# tracy starts a WASM GUI server and copies the .tracy into the build tree; neither is needed here.
pkill -u smarton -f serve_wasm; rm -f $B/build/profiler/build_wasm/traces/test_vae_ltx_prof_2x4_*.tracy
echo "T96_EXIT=$((rc1 | rc2))" | tee -a $LOG
exit $((rc1 | rc2))
