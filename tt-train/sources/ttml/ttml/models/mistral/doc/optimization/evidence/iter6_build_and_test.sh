#!/bin/bash
# Iteration 6 (all-core RMSNorm backward): waits for the iteration-5 chain to finish its TP measurement, stops its
# trailing TP-shape sweeps (lower priority, rerun later), rebuilds the Python extension with the new op, wires the
# op into the training scripts and runs the harness with the new RMSNorm test.
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
SCR=/tmp/claude-1000/-home-ttuser-agobeljic-tt-metal/0cbc71c0-30d2-4e45-a0c0-f0a2063e85a7/scratchpad
until grep -q "narrow TP-shape sweeps" $O/iter5.log 2>/dev/null; do sleep 20; done
echo "=== iter5 TP measurement done, stopping its sweeps $(date -u +%FT%TZ)"
pkill -f "[i]ter5.sh"; pkill -f "[m]atmul_sweep1d.py"; sleep 5
echo "SWEEPS STOPPED by iter6_build_and_test.sh (device needed for iteration 6; rerun later)" >> $O/iter5.log
echo "=== build _ttml $(date -u +%FT%TZ)"
cd /home/ttuser/agobeljic/tt-metal/build_Release && cmake . > $O/iter6_cmake.log 2>&1 && ninja _ttml > $O/iter6_ninja.log 2>&1; echo "build rc=$?"; grep -E "error|FAILED" $O/iter6_ninja.log | head -5
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== wire iteration 6 into the scripts"
python3 $SCR/iter6_wire.py
echo "=== iter6 rmsnorm test $(date -u +%FT%TZ)"
timeout 1800 ../python_env/bin/python -m pytest tests/python/test_mistral.py -q -p no:cacheprovider -k rmsnorm > $O/iter6_rmsnorm_test.log 2>&1; echo "rmsnorm tests rc=$?"; grep -E "passed|failed|Error" $O/iter6_rmsnorm_test.log | tail -3
echo "=== done $(date -u +%FT%TZ)"
