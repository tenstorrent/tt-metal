#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary: the build error of the fused addcmul minimal matmul on this tree.
cd /work
export TT_METAL_CACHE=/tmp/mmc PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
python3 -m pytest -p no:cacheprovider -q -x tests/ttnn/nightly/unit_tests/operations/experimental/test_dit_minimal_matmul_addcmul_fused.py -k "basic and no_bias" 2>&1 | tail -5
for f in $(find /tmp/mmc -name "*.log" -size +0 2>/dev/null); do echo "=== $f"; grep -E "error|Error|note:" "$f" | head -30; done
git log --oneline -1
