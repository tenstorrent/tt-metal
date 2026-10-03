#!/bin/bash
# blx03 broker job for #97: padded vs exact-shard (LTX_VAE_EXACT_SHARD) conv VAE decode on the 2x4 submesh of
# the full mesh (the test opens (4,8) and calls create_submesh(2,4)), 544x960/145f real 2.5 latent, fused YUV
# output, production 4x8 conv3d blockings (LTX_CONV3D_BLOCKING_MESH=4,8). Eager: 1 warmup per arm, then 3
# interleaved timed decodes per arm; md5 of each arm's yuv.
# Python from the staged overlay $S (stage97.sh); C++ build and kernel sources from $B (ce356b8815a C++).
BASE=/home/smarton/fasth3/tt-metal; B=${B:-/home/smarton/fasth3/t48}; V=/var/tmp/fasth3/t97; S=$V/src
LOG=$V/run97.log
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1
export LTX_FUSE_YUV_OUTPUT=1 LTX_CONV3D_BLOCKING_MESH=4,8 AB_LATENT=/home/smarton/fasth3/out/t37/s2reuse0/lat.gen0.pt
unset LTX_VAE_EXACT_SHARD
cd $S
echo "[t97] build=$(git -C $B rev-parse --short HEAD) src=$(cat $S/REV) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
test -f "$AB_LATENT" || { echo "[t97] $AB_LATENT missing" | tee -a $LOG; exit 3; }
test -f $B/ttnn/ttnn/_ttnn.so || { echo "[t97] no build at $B" | tee -a $LOG; exit 4; }
TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache AB_OUT_DIR=$V \
  timeout 900 python -m pytest -c $S/pytest.ini --rootdir=$S -sv --timeout=860 \
  models/tt_dit/tests/models/ltx/test_vae_ltx_exact_shard_ab.py 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
echo "T97_EXIT=$rc" | tee -a $LOG
exit $rc
