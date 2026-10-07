#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary: test_eb_swap.py with the HiFi2 rule and with both orders at HiFi4.
cd /work
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
python3 -m pytest -p no:cacheprovider -q -s tests/eb_r3_ci/test_eb_swap.py 2>&1 | grep -E "^SWAP|passed|failed|Error" 
EB_R3_NO_HIFI2=1 TT_METAL_CACHE=/tmp/ebswap_cache python3 -m pytest -p no:cacheprovider -q -s tests/eb_r3_ci/test_eb_swap.py 2>&1 | grep -E "^SWAP|passed|failed|Error"
