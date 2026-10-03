#!/bin/bash
# blx03 broker job for #100: traced conv decode A/B, 544x960/145f on the 2x4 submesh of the full mesh
# (the test opens (4,8), then create_submesh(2,4)), LTX_CONV3D_BLOCKING_MESH=4,8. Arms (ab100.py):
# table, exact, best, table2 (re-run of table to bracket drift). Python from the staged overlay $S
# (stage100.sh ttp/t100-stage), C++ build and kernels from $B.
BASE=/home/smarton/fasth3/tt-metal; B=${B:-/home/smarton/fasth3/t48}; V=/var/tmp/fasth3/t100; S=$V/src; LOG=$V/runab100.log
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1
export LTX_FUSE_YUV_OUTPUT=1 LTX_CONV3D_BLOCKING_MESH=4,8 AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt
export LTX_TRACE_REGION=500000000 TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache
cd $S
echo "[t100ab] build=$(git -C $B rev-parse --short HEAD) src=$(cat $S/REV) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
test -f "$AB_LATENT" || { echo "[t100ab] $AB_LATENT missing" | tee -a $LOG; exit 3; }
test -f $B/ttnn/ttnn/_ttnn.so || { echo "[t100ab] no build at $B" | tee -a $LOG; exit 4; }
rc=0
for a in table exact best table2; do
  echo "[t100ab] arm $a" | tee -a $LOG
  T100_ARM=$a LTX_TIME_STAGES=1 AB_OUT_DIR=$V/ab/$a timeout 600 python $S/tmp/blx03/t100/ab100.py -c $S/pytest.ini \
    --rootdir=$S -sv --timeout=560 models/tt_dit/tests/models/ltx/test_vae_ltx_trace_ab.py 2>&1 | tee -a $LOG
  r=${PIPESTATUS[0]}; echo "T100AB_${a}_EXIT=$r" | tee -a $LOG; rc=$((rc | r))
  [ $r = 0 ] || break
done
for a in exact best table2; do
  [ -f $V/ab/$a/yuv_traced.pt ] && python $S/tmp/blx03/t100/cmp100.py $V/ab/table/yuv_traced.pt $V/ab/$a/yuv_traced.pt 2>&1 | sed "s/^/[$a] /" | tee -a $LOG
done
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
echo "T100AB_EXIT=$rc" | tee -a $LOG
exit $rc
