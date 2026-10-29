#!/bin/bash
# Composite-phase device evidence, run serially on chip 0.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
E=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/composite/evidence
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== memory probe $(date -u +%FT%TZ)"
for L in 2 4 8; do
  timeout 1800 $PY $TRAIN --layers $L --steps 2 --warmup-steps 1 --random-init --random-data --mem-probe > $E/mem_probe_${L}layers.log 2>&1
  echo "layers $L rc=$?"; grep "DRAM allocated" $E/mem_probe_${L}layers.log
done
echo "=== tracy $(date -u +%FT%TZ)"
for L in 1 2 4 8; do
  rm -rf $E/tracy_${L}layers
  timeout 3600 $PY -m tracy -r -p --op-support-count 40000 -o $E/tracy_${L}layers $TRAIN --layers $L --steps 4 --warmup-steps 2 --random-init --random-data > $E/tracy_${L}layers.log 2>&1
  echo "tracy layers $L rc=$?"; ls $E/tracy_${L}layers/reports/*/ops_perf_results*.csv 2>/dev/null
  tt-smi -s 2>/dev/null | grep -i -m2 "aiclk\|asic_temperature" | tr -d ' ' | tr '\n' ' '; echo
done
echo "=== watcher $(date -u +%FT%TZ)"
rm -rf /home/ttuser/agobeljic/tt-metal/generated/watcher
TT_METAL_WATCHER=1 timeout 3600 $PY $TRAIN --layers 1 --steps 2 --warmup-steps 1 --random-init --random-data > $E/watcher_1layer.log 2>&1
echo "watcher rc=$?"; cp /home/ttuser/agobeljic/tt-metal/generated/watcher/watcher.log $E/watcher_1layer_watcher.log 2>/dev/null; grep -c "" $E/watcher_1layer_watcher.log; grep -i -m3 "error\|hang\|stall" $E/watcher_1layer_watcher.log
echo "=== determinism + MFU rerun, 8 layers real weights $(date -u +%FT%TZ)"
timeout 3600 $PY $TRAIN --layers 8 --steps 10 --warmup-steps 2 --loss-log $E/loss_8layers_run2.json > $E/train_8layers_run2.log 2>&1
echo "rerun rc=$?"; grep -E "^step|layers:" $E/train_8layers_run2.log
echo "=== done $(date -u +%FT%TZ)"
