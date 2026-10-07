#!/bin/bash
# t212: one DiffVAE decode-only broker job on blx01 (1080p, 145 frames, 4x8 ring, 2 links, slab 78).
# T212_TAG=A: unported t48 5e4e0cd643a (python overlay treeA, C++ build /var/tmp/fasth3/t48, NA code identical).
# T212_TAG=B: the t212 port (tree b, own build). Both dump uint8 pixels for the PCC/PSNR compare.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t212; tag=${T212_TAG:?}
OUT=$T/out_$tag; mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp HF_HUB_OFFLINE=1
source $F/t48/python_env/bin/activate
if [ "$tag" = A ]; then W=$F/t48; P=$T/treeA; else W=$T/b; P=$W; fi
export TT_METAL_HOME=$W PYTHONPATH=$P:$W:$W/ttnn:$W/tools
export TT_METAL_CACHE=$F/cache/tt-metal-cache TT_DIT_CACHE_DIR=$F/cache/dit-ltx25
export DIFFVAE_CHECKPOINT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941/vae/ltx-2.5-video-vae-bf16.safetensors
export TT_DIT_STAGE_TIMING=1 TT_DIT_STAGE_LOG=1 DIFFVAE_DUMP_PIXELS=$OUT/px.pt
cd $OUT
echo "[t212] tag=$tag host=$(hostname) build=$(git -C $W rev-parse --short=11 HEAD) py=$P $(date -u '+%F %T') UTC" | tee $OUT/run.log
T0=$(date +%s)
python -u -m pytest -c $P/pytest.ini --rootdir=$P -sv -p no:cacheprovider --timeout=${T212_PYTEST_TIMEOUT:-560} \
  "$P/models/tt_dit/tests/models/vae/test_diffvae_ltx.py::test_decode_wsp_timing" -k s34x60 -x \
  --diffvae-slab-frames 78 --diffvae-topology ring --diffvae-num-links 2 2>&1 | tee -a $OUT/run.log
rc=${PIPESTATUS[0]}
echo "[t212] process wall $(( $(date +%s) - T0 )) s" | tee -a $OUT/run.log
echo "T212_EXIT=$rc" | tee -a $OUT/run.log
exit $rc
