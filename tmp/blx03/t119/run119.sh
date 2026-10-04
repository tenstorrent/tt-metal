#!/bin/bash
# One blx03 broker job for #119: traced conv decode on the 2x4 submesh of the full mesh (the test opens
# (4,8), then create_submesh(2,4)), checked against the stored reference decode.
# Usage: run119.sh <step>. Steps: ref544 | s0ups:<blk> (544x960, LTX_CONV3D_BLOCKING_MESH=4,8)
#        ref1080 | s4res:<blk> (1088x1920, real 2x4 keys, no mesh override). <blk> = Cin,Cout,T,H,W.
# ref* steps decode with LTX_VAE_HALO_ONLY=0 and record the reference if it is missing.
STEP=$1
BASE=/home/smarton/fasth3/tt-metal; B=${B:-/home/smarton/fasth3/t48}; V=/var/tmp/fasth3/t119; S=$V/src
LOG=$V/run_${STEP//[:,]/_}_$(date -u +%H%M%S).log
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1
export LTX_FUSE_YUV_OUTPUT=1 AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt
export LTX_TRACE_REGION=500000000 TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache LTX_TIME_STAGES=1
case $STEP in
  ref544) export LTX_CONV3D_BLOCKING_MESH=4,8 LTX_VAE_HALO_ONLY=0 LTX_VAE_REF_RECORD=1 T119_KEY=none ;;
  s0ups:*) export LTX_CONV3D_BLOCKING_MESH=4,8 T119_KEY=s0ups T119_BLK=${STEP#*:} ;;
  ref1080) export LTX_VAE_AB_HW=1088,1920 LTX_VAE_HALO_ONLY=0 LTX_VAE_REF_RECORD=1 T119_KEY=none ;;
  s4res:*) export LTX_VAE_AB_HW=1088,1920 T119_KEY=s4res2x4 T119_BLK=${STEP#*:} ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
cd $S
# The clock ceiling holds every arm at 1150 MHz (as in #100's 506.2 ms); restore it however the job ends.
echo "[t119] step=$STEP build=$(git -C $B rev-parse --short HEAD) src=$(cat $S/REV)" | tee $LOG
test -f "$AB_LATENT" || { echo "[t119] $AB_LATENT missing" | tee -a $LOG; exit 3; }
test -f $B/ttnn/ttnn/_ttnn.so || { echo "[t119] no build at $B" | tee -a $LOG; exit 4; }
timeout 1500 python $S/tmp/blx03/t119/ab119.py -c $S/pytest.ini --rootdir=$S -sv --timeout=1440 \
  models/tt_dit/tests/models/ltx/test_vae_ltx_trace_ab.py 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
echo "T119_EXIT step=$STEP rc=$rc" | tee -a $LOG
exit $rc
