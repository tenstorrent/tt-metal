#!/bin/bash
# blx03 broker job for #85: denoise-trim block A/B on a 2x4 submesh carved from the full (4,8) mesh.
# Build, kernels and models/common come from ~/fasth3/t48 @a613d669ee (same C++ as the t85 base for the DiT
# block); models/tt_dit comes from the t85 branch (src/, ahead on PYTHONPATH; models is a namespace package).
M=/home/smarton/fasth3/t48; D=/home/smarton/fasth3/t85; LOG=$D/run85.log
source /home/smarton/fasth3/tt-metal/python_env/bin/activate
export TT_METAL_HOME=$M PYTHONPATH=$D/src:$M:$M/ttnn:$M/tools HF_HUB_OFFLINE=1
export TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache
# cwd is $D, not $M: python -m puts cwd first on sys.path, and $M ahead of src/ would shadow models/tt_dit.
cd $D
echo "[t85] tree=$(git -C $M rev-parse --short HEAD) src=$(cat $D/src/REV)" | tee $LOG
timeout 1440 python -m pytest -p conftest -c $M/pytest.ini --rootdir=$M -sv --timeout=1400 $D/test_denoise_trims_ab.py 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
echo "T85_EXIT=$rc" | tee -a $LOG
exit $rc
