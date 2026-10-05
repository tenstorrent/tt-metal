#!/bin/bash
# blx03 broker job for #134: LTX_SDPA_MM_LOFI A/B on a 2x4 submesh carved from the full (4,8) mesh.
# Usage: run134.sh ref|new
#   ref: ~/fasth3/t48 build (this branch's base, without the SDPA cherry-picks and the knob), arm REF;
#        saves its outputs to $V for the new run.
#   new: ~/fasth3/t134 build (this branch), arms OFF and LOFI, compared against REF's saved outputs.
mode=$1
V=/var/tmp/fasth3/t134; D=$V/run; H=/home/smarton/fasth3/t134
case "$mode" in
  ref) M=/home/smarton/fasth3/t48; ARMS=REF; T=1000 ;;
  new) M=$H; ARMS=OFF,LOFI; T=1600 ;;
  *) echo "usage: $0 ref|new"; exit 2 ;;
esac
LOG=$V/run134_$mode.log
mkdir -p $D && cp $H/tmp/t134/test_sdpa_mm_lofi_ab.py $D/ || exit 3
source /home/smarton/fasth3/tt-metal/python_env/bin/activate
export TT_METAL_HOME=$M PYTHONPATH=$M:$M/ttnn:$M/tools HF_HUB_OFFLINE=1
export TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache T134_ARMS=$ARMS T134_OUT=$V
unset LTX_SDPA_MM_LOFI
# cwd is $D, outside both trees: pytest loads only $M's root conftest (-p conftest), not also the t134 one
# that a test file inside ~/fasth3/t134 would pick up.
cd $D
echo "[t134] mode=$mode tree=$(git -C $M rev-parse --short HEAD) harness=$(git -C $H rev-parse --short HEAD)" | tee $LOG
timeout $T python -m pytest -p conftest -c $M/pytest.ini --rootdir=$M -sv --timeout=$((T - 50)) \
  $D/test_sdpa_mm_lofi_ab.py 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
echo "T134_EXIT_$mode=$rc" | tee -a $LOG
exit $rc
