#!/bin/bash
# One blx01 broker job for #127: one latent-upsampler ups_initial blocking A/B on the full 4x8 mesh
# (LTX_CONV3D_BLOCKING_MESH=4,8, S1 latent 19x17x30), arms in one process: base, candidate, base again (drift check).
# Usage: bash run127.sh <job tag> <arms "Cin,Cout,T,H,W/...">
F=/var/tmp/fasth3; A=$F/t48; V=$F/t127; J=$1; ARMS=$2
LOG=$V/run127_$J.log
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp
source $A/python_env/bin/activate
export TT_METAL_HOME=$A TT_METAL_RUNTIME_ROOT=$A PYTHONPATH=$A:$A/ttnn:$A/tools HF_HUB_OFFLINE=1
export TT_METAL_CACHE=$F/cache/tt-metal-cache LTX_CONV3D_BLOCKING_MESH=4,8 LTX_TRACE_REGION=500000000
export T119_UPS_ARMS=$ARMS
cd $A
echo "[t127] job=$J arms=$ARMS tree=$(git rev-parse --short=11 HEAD) dirty=$(git status --short -uno | wc -l) boot=$(uptime -s) start epoch=$(date +%s)" | tee $LOG
timeout 540 python -m pytest -c $A/pytest.ini --rootdir=$A -sv --timeout=520 tmp/t127/test_t119_4x8.py::test_ups_ab 2>&1 |
  tee -a $LOG | grep -E '^\[t127\]|^T119|passed|failed|Error'
rc=${PIPESTATUS[0]}
echo "T127_EXIT=$rc end_epoch=$(date +%s)" | tee -a $LOG
exit $rc
