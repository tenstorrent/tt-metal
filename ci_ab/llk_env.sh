#!/usr/bin/env bash
# Scratch CI, not for merge. source ci_ab/llk_env.sh: the tt-llk test environment in a test-dispatch job (sfpi from tests/sfpi-version,
# a venv with tests/requirements.txt). Sets LLKPY (python of that venv).
W=${CIAB_W:-/work}; L=$W/tt_metal/tt-llk/tests; t=$(date +%s)
(cd $L && ./setup_testing_env.sh > /tmp/llk_setup.log 2>&1); echo "CIAB LLKENV sfpi rc=$? $(ls $L/sfpi 2>/dev/null | head -3 | tr '\n' ' ')"; tail -n 3 /tmp/llk_setup.log
if python3 -m venv /tmp/llkv > /tmp/llk_venv.log 2>&1; then LLKPY=/tmp/llkv/bin/python
  /tmp/llkv/bin/pip install -q -r $L/requirements.txt >> /tmp/llk_venv.log 2>&1; echo "CIAB LLKENV venv pip rc=$?"
else LLKPY=python3; python3 -m pip install -q -r $L/requirements.txt >> /tmp/llk_venv.log 2>&1; echo "CIAB LLKENV system pip rc=$?"; fi
tail -n 3 /tmp/llk_venv.log; $LLKPY -c "import ttexalens, pytest, pandas; print('CIAB LLKENV imports ok', ttexalens.__file__)"; export LLKPY
echo "CIAB LLKENV done $(( $(date +%s) - t )) s"
