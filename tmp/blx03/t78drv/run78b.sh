#!/bin/bash
# blx03 broker job for #78 step 3: conv3d halo unit tests (-k halo) on a 2x4 submesh of the full mesh
# (conftest.py next to test_conv3d.py overrides `device`). Needs the t78 build.
BASE=/home/smarton/fasth3/tt-metal; W=/home/smarton/fasth3/t78; V=/var/tmp/fasth3/t78
LOG=$V/run78b.log
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1 TT_METAL_CACHE=$V/jit
cd $W
echo "[t78b] tree=$(git rev-parse --short HEAD) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
test -f tests/ttnn/unit_tests/operations/conv/conftest.py || { echo "[t78b] no conftest override" | tee -a $LOG; exit 4; }
timeout 500 python -m pytest -c $W/pytest.ini --rootdir=$W -sv --timeout=450 \
  tests/ttnn/unit_tests/operations/conv/test_conv3d.py -k halo 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
echo "T78B_EXIT=$rc" | tee -a $LOG
exit $rc
