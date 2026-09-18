#!/usr/bin/env bash
# Agent S's bitwise/oracle verification of the two-stream backward: the overlap module on 1x2, the whole SP suite under
# the overlap (1x2), the overlap module on 1x4 ring and line. One devrun each; summary in logs/S_unit_chain.txt.
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
PY_TEST="../python_env/bin/python -m pytest -q -p no:cacheprovider"
OUT="$SPFUSE/logs/S_unit_chain.txt"; : > "$OUT"
r() { local name=$1; shift; "$SPFUSE/devrun.sh" "$name" 900 3600 -- "$@" > /dev/null 2>&1; echo "$name: $(grep -oE '[0-9]+ (passed|failed)[^=]*' $SPFUSE/logs/$name.log | tail -1)" | tee -a "$OUT"; }
r S_ovl_1x2      "cd tt-train && $PY_TEST tests/python/test_sp_overlap.py 2>&1 | grep -v '^2026-.*| info'"
r S_suite_overlap "cd tt-train && TTML_SP_OVERLAP=backward $PY_TEST tests/python/test_sequence_parallel.py 2>&1 | grep -v '^2026-.*| info'"
r S_ovl_1x4_ring "cd tt-train && TTML_SP_TEST_MESH=1x4_ring $PY_TEST tests/python/test_sp_overlap.py 2>&1 | grep -v '^2026-.*| info'"
r S_ovl_1x4_line "cd tt-train && TTML_SP_TEST_MESH=1x4_line $PY_TEST tests/python/test_sp_overlap.py 2>&1 | grep -v '^2026-.*| info'"
echo "UNIT_CHAIN_DONE $(date +%T)" | tee -a "$OUT"
