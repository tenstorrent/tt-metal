#!/bin/bash
# t40: Gemma-4 text-encode timing breakdown on blx03 (one short device job).
set -o pipefail
W=${W:-/home/smarton/fasth3/t40}
OUT=${OUT:-/home/smarton/fasth3/out/t40}
cd $W
source /home/smarton/fasth3/tt-metal/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1
export LTX25_ROOT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
export FASTH3_DATA=/var/tmp/fasth3
export TT_METAL_CACHE=$FASTH3_DATA/cache/tt-metal-cache TT_DIT_CACHE_DIR=$FASTH3_DATA/cache/dit-ltx25
mkdir -p $OUT
python -u -m pytest -sv --timeout=900 models/tt_dit/tests/encoders/gemma4/test_gemma4_encode_timing.py ${PYTEST_K:+-k $PYTEST_K} 2>&1 | tee $OUT/${LOG:-prof.log}
