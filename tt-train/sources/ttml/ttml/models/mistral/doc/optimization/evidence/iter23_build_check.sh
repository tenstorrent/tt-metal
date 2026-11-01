#!/bin/bash
# Iteration 23: the Q backward chain forwarding moved to the writer RISC (receive and forward overlap); build,
# standalone comparison against the iteration-20/21 kernel outputs and walls, attention harness tests.
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
SP=/tmp/claude-1000/-home-ttuser-agobeljic-tt-metal/0cbc71c0-30d2-4e45-a0c0-f0a2063e85a7/scratchpad
TAG=${TAG:-iter23}
[ "${SKIP_BUILD:-0}" = 1 ] && echo "=== build skipped" || { echo "=== build _ttml $(date -u +%FT%TZ)"
cd /home/ttuser/agobeljic/tt-metal/build_Release && cmake . > $O/${TAG}_cmake.log 2>&1 && timeout 3600 ninja _ttml > $O/${TAG}_ninja.log 2>&1; rc=$?; echo "build rc=$rc"
grep -E "error:|FAILED" $O/${TAG}_ninja.log | head -20
[ $rc -ne 0 ] && { echo "=== build failed $(date -u +%FT%TZ)"; exit 1; }; }
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== sdpa check vs the iteration-21 baseline outputs $(date -u +%FT%TZ)"
timeout 900 ../python_env/bin/python $O/iter11_sdpa_check.py --ref $SP/iter21_sdpa_baseline.npz > $O/${TAG}_sdpa_check.log 2>&1; echo "check rc=$?"
grep -E "^case|pcc|Error|error:|FATAL|Traceback" $O/${TAG}_sdpa_check.log | grep -v "riscv" | head -40
echo "=== attention harness tests $(date -u +%FT%TZ)"
timeout 1800 ../python_env/bin/python -m pytest tests/python/test_mistral.py -q -p no:cacheprovider -k "attention or sdpa" > $O/${TAG}_attention_tests.log 2>&1; echo "tests rc=$?"
grep -E "passed|failed|Error" $O/${TAG}_attention_tests.log | tail -5
echo "=== done $(date -u +%FT%TZ)"
