#!/bin/bash
# blx03 broker job for #110: gate fold + norm-AdaLN block A/B on a 2x4 submesh carved from the full (4,8) mesh.
# Tree, build and models/tt_dit come from ~/fasth3/t48 @83c11ee2b3 (t48 tip a9898c4c850 differs only in a CPU test).
M=/home/smarton/fasth3/t48; D=/var/tmp/fasth3/t110; LOG=$D/run110.log
source /home/smarton/fasth3/tt-metal/python_env/bin/activate
export TT_METAL_HOME=$M PYTHONPATH=$M:$M/ttnn:$M/tools HF_HUB_OFFLINE=1 T110_TREE=$M/models/tt_dit
export TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache
cd $D
echo "[t110] tree=$(git -C $M rev-parse --short HEAD) dirty=$(git -C $M status --short -- models/tt_dit | wc -l)" | tee $LOG
timeout 1440 python -m pytest -p conftest -c $M/pytest.ini --rootdir=$M -sv --timeout=1400 $D/test_gate_adaln_ab.py 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
echo "T110_EXIT=$rc" | tee -a $LOG
exit $rc
