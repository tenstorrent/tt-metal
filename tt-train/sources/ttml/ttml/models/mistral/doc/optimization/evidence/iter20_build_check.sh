#!/bin/bash
# Iteration 20: norm kernels (add_rmsnorm_fw_rows pass 2, rmsnorm_bw_rows passes 1 and 2) on the FPU, 3-4 tiles per DEST
# acquire. Build _ttml, standalone forward check (vs the two-phase op and fp64), standalone backward check (vs fp64),
# norm harness tests.
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
[ "${SKIP_BUILD:-0}" = 1 ] && echo "=== build skipped" || { echo "=== build _ttml $(date -u +%FT%TZ)"
cd /home/ttuser/agobeljic/tt-metal/build_Release && timeout 3600 ninja _ttml > $O/iter20_ninja.log 2>&1; rc=$?; echo "build rc=$rc"
grep -E "error:|FAILED" $O/iter20_ninja.log | head -20
[ $rc -ne 0 ] && { echo "=== build failed $(date -u +%FT%TZ)"; exit 1; }; }
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== add_rmsnorm_fw_rows check $(date -u +%FT%TZ)"
timeout 1200 ../python_env/bin/python $O/iter19_norm_check.py > $O/iter20_norm_fw_check.log 2>&1; echo "fw check rc=$?"
grep -E "^B=|^   |Error|error:|FATAL|Traceback" $O/iter20_norm_fw_check.log | grep -v "riscv" | head -40
echo "=== rmsnorm_bw_rows check $(date -u +%FT%TZ)"
timeout 1200 ../python_env/bin/python $O/iter17_rmsnorm_check.py > $O/iter20_norm_bw_check.log 2>&1; echo "bw check rc=$?"
grep -E "pcc|Error|error:|FATAL|Traceback" $O/iter20_norm_bw_check.log | grep -v "riscv" | head -20
echo "=== norm harness tests $(date -u +%FT%TZ)"
timeout 1800 ../python_env/bin/python -m pytest tests/python/test_mistral.py -q -p no:cacheprovider -k "norm or fused_residual" > $O/iter20_norm_tests.log 2>&1; echo "tests rc=$?"
grep -E "passed|failed|Error" $O/iter20_norm_tests.log | tail -5
echo "=== done $(date -u +%FT%TZ)"
