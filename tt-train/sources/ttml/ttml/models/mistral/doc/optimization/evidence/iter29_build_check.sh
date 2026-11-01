#!/bin/bash
# Iteration 29: rmsnorm_bw_rows pass 2 on the FPU (iteration 20 variant A, admissible under the relaxed precision rule).
# Build _ttml, standalone backward check (vs fp64: pcc, rel, wall), norm harness tests.
# norm harness tests.
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
[ "${SKIP_BUILD:-0}" = 1 ] && echo "=== build skipped" || { echo "=== build _ttml $(date -u +%FT%TZ)"
cd /home/ttuser/agobeljic/tt-metal/build_Release && timeout 3600 ninja _ttml > $O/iter29_ninja.log 2>&1; rc=$?; echo "build rc=$rc"
grep -E "error:|FAILED" $O/iter29_ninja.log | head -20
[ $rc -ne 0 ] && { echo "=== build failed $(date -u +%FT%TZ)"; exit 1; }; }
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== rmsnorm_bw_rows check $(date -u +%FT%TZ)"
timeout 1200 ../python_env/bin/python $O/iter17_rmsnorm_check.py > $O/iter29_norm_bw_check.log 2>&1; echo "bw check rc=$?"
grep -E "pcc|Error|error:|FATAL|Traceback" $O/iter29_norm_bw_check.log | grep -v "riscv" | head -20
echo "=== norm harness tests $(date -u +%FT%TZ)"
timeout 1800 ../python_env/bin/python -m pytest tests/python/test_mistral.py -q -p no:cacheprovider -k "norm or fused_residual" > $O/iter29_norm_tests.log 2>&1; echo "tests rc=$?"
grep -E "passed|failed|Error" $O/iter29_norm_tests.log | tail -5
echo "=== done $(date -u +%FT%TZ)"
