#!/bin/bash
# Iteration 11: rebuild _ttml with the SDPA work-list kernels, compare against the baseline outputs, run the
# attention harness tests.
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
SCR=/tmp/claude-1000/-home-ttuser-agobeljic-tt-metal/0cbc71c0-30d2-4e45-a0c0-f0a2063e85a7/scratchpad
[ "${SKIP_BUILD:-0}" = 1 ] && echo "=== build skipped" || { echo "=== build _ttml $(date -u +%FT%TZ)"
cd /home/ttuser/agobeljic/tt-metal/build_Release && cmake . > $O/iter11_cmake.log 2>&1 && timeout 3600 ninja _ttml > $O/iter11_ninja.log 2>&1; rc=$?; echo "build rc=$rc"
grep -E "error:|FAILED" $O/iter11_ninja.log | head -20
[ $rc -ne 0 ] && { echo "=== build failed $(date -u +%FT%TZ)"; exit 1; }; }
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== sdpa check vs baseline $(date -u +%FT%TZ)"
timeout 1200 ../python_env/bin/python $O/iter11_sdpa_check.py --ref $SCR/iter11_sdpa_baseline.npz > $O/iter11_sdpa_check.log 2>&1; echo "check rc=$?"
grep -E "case|pcc|Error|error|FATAL|Traceback" $O/iter11_sdpa_check.log | grep -v "riscv" | head -60
echo "=== attention harness tests $(date -u +%FT%TZ)"
timeout 1800 ../python_env/bin/python -m pytest tests/python/test_mistral.py -q -p no:cacheprovider -k "attention or block" > $O/iter11_attention_tests.log 2>&1; echo "tests rc=$?"
grep -E "passed|failed|Error" $O/iter11_attention_tests.log | tail -5
echo "=== done $(date -u +%FT%TZ)"
