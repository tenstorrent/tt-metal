#!/bin/bash
# One blx03 broker job for #119 on the full 4x8 mesh (production layout, 1080p/145f shards).
# Usage: run119.sh <step>. Steps: vae (halo-off reference + default decoder, eager and traced, VAE_REF gate)
#        ups:<blkA>/<blkB> (latent upsampler ups_initial A/B vs the fp32 reference). <blk> = Cin,Cout,T,H,W.
STEP=$1
BASE=/home/smarton/fasth3/tt-metal; B=${B:-/home/smarton/fasth3/t48}; V=/var/tmp/fasth3/t119; S=$V/src
LOG=$V/run_${STEP//[:,\/]/_}_$(date -u +%H%M%S).log
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1
export LTX_FUSE_YUV_OUTPUT=1 AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt
export LTX_TRACE_REGION=500000000 TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache LTX_TIME_STAGES=1
export LTX_CONV3D_BLOCKING_MESH=4,8 T119_KEY=none
case $STEP in
  vae) export LTX_VAE_AB_HW=1088,1920 LTX_VAE_REF_RECORD=1 TEST=test_vae ;;
  ups:*) export T119_UPS_ARMS=${STEP#ups:} TEST=test_ups_ab ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
cd $S
echo "[t119] step=$STEP build=$(git -C $B rev-parse --short HEAD) src=$(cat $S/REV) boot=$(uptime -s)" | tee $LOG
test -f "$AB_LATENT" || { echo "[t119] $AB_LATENT missing" | tee -a $LOG; exit 3; }
test -f $B/ttnn/ttnn/_ttnn.so || { echo "[t119] no build at $B" | tee -a $LOG; exit 4; }
timeout 1500 python $S/tmp/blx03/t119/ab119.py -c $S/pytest.ini --rootdir=$S -sv --timeout=1440 \
  tmp/blx03/t119/test_t119_4x8.py::$TEST 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
echo "T119_EXIT step=$STEP rc=$rc" | tee -a $LOG
exit $rc
