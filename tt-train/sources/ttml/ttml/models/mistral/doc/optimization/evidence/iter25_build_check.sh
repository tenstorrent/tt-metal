#!/bin/bash
# Iteration 25: the low-rank update fused into transpose_split (dx = dcat^T[:in] + s_dh @ A in one pass); build, the
# standalone comparison against the two-op path, the fused-LoRA harness subset.

set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
TAG=${TAG:-iter25}
[ "${SKIP_BUILD:-0}" = 1 ] && echo "=== build skipped" || { echo "=== build _ttml $(date -u +%FT%TZ)"
cd /home/ttuser/agobeljic/tt-metal/build_Release && cmake . > $O/${TAG}_cmake.log 2>&1 && timeout 3600 ninja _ttml > $O/${TAG}_ninja.log 2>&1; rc=$?; echo "build rc=$rc"
grep -E "error:|FAILED" $O/${TAG}_ninja.log | head -20
[ $rc -ne 0 ] && { echo "=== build failed $(date -u +%FT%TZ)"; exit 1; }; }
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== fused lowrank check $(date -u +%FT%TZ)"
timeout 900 ../python_env/bin/python $O/iter25_fused_check.py > $O/${TAG}_fused_check.log 2>&1; echo "check rc=$?"
grep -E "fused|tail|wide|us:|FUSED|Error|error:|FATAL|Traceback" $O/${TAG}_fused_check.log | grep -v "riscv" | head -40
echo "=== fused-LoRA harness subset $(date -u +%FT%TZ)"
timeout 2400 ../python_env/bin/python -m pytest tests/python/test_mistral.py -q -p no:cacheprovider -k "fused or lora" > $O/${TAG}_lora_tests.log 2>&1; echo "tests rc=$?"
grep -E "passed|failed|Error" $O/${TAG}_lora_tests.log | tail -5
echo "=== done $(date -u +%FT%TZ)"
