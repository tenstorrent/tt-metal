#!/bin/bash
# blx03 broker job for #99 on the 2x4 submesh of the full mesh (every test opens (4,8), then create_submesh(2,4)):
#   1) fused RMSNorm+add unit shapes on 2x4
#   2) traced conv decode A/B, 544x960/145f, LTX_CONV3D_BLOCKING_MESH=4,8: LTX_FUSE_NORM_ADD=0, then =1
# Python from the staged overlay $S (stage99.sh); C++ build and kernels from $B.
BASE=/home/smarton/fasth3/tt-metal; B=/home/smarton/fasth3/t99; V=/var/tmp/fasth3/t99; S=$V/src; LOG=$V/run99.log
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1 T99_SRC=$S
export LTX_FUSE_YUV_OUTPUT=1 LTX_CONV3D_BLOCKING_MESH=4,8 AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt
export LTX_TRACE_REGION=500000000 TT_METAL_CACHE=$V/jit
cd $S
echo "[t99] build=$(git -C $B rev-parse --short HEAD) src=$(cat $S/REV) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
test -f "$AB_LATENT" || { echo "[t99] $AB_LATENT missing" | tee -a $LOG; exit 3; }
test -f $B/ttnn/ttnn/_ttnn.so || { echo "[t99] no build at $B" | tee -a $LOG; exit 4; }
mkdir -p $V/jit
echo "[t99] part 1: unit shapes" | tee -a $LOG
timeout 600 python -m pytest -c $S/pytest.ini --rootdir=$S -sv --timeout=560 tmp/blx03/t99/test_rs_2x4.py 2>&1 | tee -a $LOG
rc1=${PIPESTATUS[0]}; echo "T99_PART1_EXIT=$rc1" | tee -a $LOG
rc=$rc1
if [ $rc1 = 0 ]; then
  for f in 0 1; do
    echo "[t99] part 2: decode LTX_FUSE_NORM_ADD=$f" | tee -a $LOG
    LTX_FUSE_NORM_ADD=$f LTX_TIME_STAGES=1 AB_OUT_DIR=$V/fuse$f timeout 900 python -m pytest -c $S/pytest.ini --rootdir=$S -sv \
      --timeout=860 models/tt_dit/tests/models/ltx/test_vae_ltx_trace_ab.py 2>&1 | tee -a $LOG
    r=${PIPESTATUS[0]}; echo "T99_DECODE${f}_EXIT=$r" | tee -a $LOG; rc=$((rc | r))
    [ $r = 0 ] || break
  done
  for k in eager traced; do
    [ -f $V/fuse1/yuv_$k.pt ] && python $S/tmp/blx03/t99/cmp99.py $V/fuse0/yuv_$k.pt $V/fuse1/yuv_$k.pt 2>&1 | sed "s/^/[$k] /" | tee -a $LOG
  done
fi
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
echo "T99_EXIT=$rc" | tee -a $LOG
exit $rc
