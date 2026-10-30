#!/bin/bash
# Iteration 12: build _ttml with the single-phase rmsnorm_bw_rows op, standalone check, rmsnorm harness tests.
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
[ "${SKIP_BUILD:-0}" = 1 ] && echo "=== build skipped" || { echo "=== build _ttml $(date -u +%FT%TZ)"
cd /home/ttuser/agobeljic/tt-metal/build_Release && cmake . > $O/iter17_cmake.log 2>&1 && timeout 3600 ninja _ttml > $O/iter17_ninja.log 2>&1; rc=$?; echo "build rc=$rc"
grep -E "error:|FAILED" $O/iter17_ninja.log | head -20
[ $rc -ne 0 ] && { echo "=== build failed $(date -u +%FT%TZ)"; exit 1; }; }
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== rmsnorm_bw_rows check $(date -u +%FT%TZ)"
timeout 1200 ../python_env/bin/python $O/iter17_rmsnorm_check.py > $O/iter17_rmsnorm_check.log 2>&1; echo "check rc=$?"
grep -E "pcc|Error|error:|FATAL|Traceback" $O/iter17_rmsnorm_check.log | grep -v "riscv" | head -30
echo "=== lora harness tests $(date -u +%FT%TZ)"
timeout 1800 ../python_env/bin/python -m pytest tests/python/test_mistral.py -q -p no:cacheprovider -k "rmsnorm or add_rmsnorm or fused_residual" > $O/iter17_rmsnorm_tests.log 2>&1; echo "tests rc=$?"
grep -E "passed|failed|Error" $O/iter17_rmsnorm_tests.log | tail -5
echo "=== done $(date -u +%FT%TZ)"
