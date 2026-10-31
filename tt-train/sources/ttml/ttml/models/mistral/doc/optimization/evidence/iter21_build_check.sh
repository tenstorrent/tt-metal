#!/bin/bash
# Iteration 21: SDPA forward from the sdpa_scheduler stack (PRs 61884 + 62646: multi-row Q chunks, K/V steps,
# head stacking, longest-first placement with cut heavy items merged in-kernel) replacing the chain-mode
# forward. Build, standalone comparison against the iteration-20 kernels, the benchmark at the shard shape,
# attention harness tests.
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
SP=/tmp/claude-1000/-home-ttuser-agobeljic-tt-metal/0cbc71c0-30d2-4e45-a0c0-f0a2063e85a7/scratchpad
[ "${SKIP_BUILD:-0}" = 1 ] && echo "=== build skipped" || { echo "=== build _ttml sdpa_fw_benchmark $(date -u +%FT%TZ)"
cd /home/ttuser/agobeljic/tt-metal/build_Release && cmake . > $O/iter21_cmake.log 2>&1 && timeout 3600 ninja _ttml sdpa_fw_benchmark > $O/iter21_ninja.log 2>&1; rc=$?; echo "build rc=$rc"
grep -E "error:|FAILED" $O/iter21_ninja.log | head -20
[ $rc -ne 0 ] && { echo "=== build failed $(date -u +%FT%TZ)"; exit 1; }; }
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== sdpa check vs the iteration-20 kernels $(date -u +%FT%TZ)"
timeout 900 ../python_env/bin/python $O/iter11_sdpa_check.py --ref $SP/iter21_sdpa_baseline.npz > $O/iter21_sdpa_check.log 2>&1; echo "check rc=$?"
grep -E "^case|pcc|Error|error:|FATAL|Traceback" $O/iter21_sdpa_check.log | grep -v "riscv" | head -40
echo "=== benchmark at the shard shape $(date -u +%FT%TZ)"
timeout 900 /home/ttuser/agobeljic/tt-metal/build_Release/tt-train/tests/sdpa_fw_benchmark mistral24b 0,0,0,1 128,128,1 128,256,1 256,256,1 128,128,2 128,256,2 128,256,1,2 128,256,1,4 256,256,1,4 > $O/iter21_sdpa_benchmark.log 2>&1; echo "bench rc=$?"
grep -v "riscv\|^ *|\|^/home\|^In file\|^Building\|warning\|info" $O/iter21_sdpa_benchmark.log | tail -25
echo "=== attention harness tests $(date -u +%FT%TZ)"
timeout 1800 ../python_env/bin/python -m pytest tests/python/test_mistral.py -q -p no:cacheprovider -k "attention or sdpa" > $O/iter21_attention_tests.log 2>&1; echo "tests rc=$?"
grep -E "passed|failed|Error" $O/iter21_attention_tests.log | tail -5
echo "=== done $(date -u +%FT%TZ)"
