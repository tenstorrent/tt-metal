#!/bin/bash
# Iteration 19: build _ttml with the single-phase add_rmsnorm_fw_rows op (dispatched from add_rmsnorm_fw_allcore) and
# the 2D w2-forward matmul config; standalone norm check (vs the two-phase op and fp64), norm + lora harness tests.
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
[ "${SKIP_BUILD:-0}" = 1 ] && echo "=== build skipped" || { echo "=== build _ttml $(date -u +%FT%TZ)"
cd /home/ttuser/agobeljic/tt-metal/build_Release && cmake . > /tmp/claude-1000/-home-ttuser-agobeljic-tt-metal/0cbc71c0-30d2-4e45-a0c0-f0a2063e85a7/scratchpad/iter19_cmake.log 2>&1 && timeout 3600 ninja _ttml > $O/iter19_ninja.log 2>&1; rc=$?; echo "build rc=$rc"
grep -E "error:|FAILED" $O/iter19_ninja.log | head -20
[ $rc -ne 0 ] && { echo "=== build failed $(date -u +%FT%TZ)"; exit 1; }; }
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== add_rmsnorm_fw_rows check $(date -u +%FT%TZ)"
timeout 1200 ../python_env/bin/python $O/iter19_norm_check.py > $O/iter19_norm_check.log 2>&1; echo "check rc=$?"
grep -E "^B=|^   |Error|error:|FATAL|Traceback" $O/iter19_norm_check.log | grep -v "riscv" | head -40
echo "=== norm + lora harness tests $(date -u +%FT%TZ)"
timeout 1800 ../python_env/bin/python -m pytest tests/python/test_mistral.py -q -p no:cacheprovider -k "norm or lora or skinny_contract or block" > $O/iter19_norm_tests.log 2>&1; echo "tests rc=$?"
grep -E "passed|failed|Error" $O/iter19_norm_tests.log | tail -5
echo "=== done $(date -u +%FT%TZ)"
