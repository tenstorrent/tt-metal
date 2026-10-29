#!/bin/bash
# Composite-phase review round 1 follow-ups, serial on chip 0.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
E=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/composite/evidence
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== harness at HEAD $(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $E/pcc_harness_head.log 2>&1; echo "test_mistral rc=$?"; grep -E "passed|failed" $E/pcc_harness_head.log | tail -1
timeout 1800 $PY -m pytest tests/python/test_llama.py tests/python/test_llama_safetensors_loader.py -q -p no:cacheprovider > $E/test_llama_head.log 2>&1; echo "test_llama rc=$?"; grep -E "passed|failed" $E/test_llama_head.log | tail -1
echo "=== 8-layer real weights with host fp32 loss $(date -u +%FT%TZ)"
timeout 3600 $PY $TRAIN --layers 8 --steps 10 --warmup-steps 2 --host-loss --loss-log $E/loss_8layers_hostloss.json > $E/train_8layers_hostloss.log 2>&1; echo "rc=$?"; grep -E "^step|layers:" $E/train_8layers_hostloss.log
echo "=== clock trace: 8 layers, 60 steps, tt-smi every 1 s $(date -u +%FT%TZ)"
( timeout 1200 $PY $TRAIN --layers 8 --steps 60 --warmup-steps 2 --random-init --random-data --loss-log $E/clock_trace_8layers.json > $E/clock_trace_8layers.log 2>&1; echo "trace rc=$?" >> $E/clock_trace_8layers.log ) &
for i in $(seq 1 75); do
  echo "$(date -u +%T) $(timeout 5 tt-smi -s 2>/dev/null | $PY -c "
import sys, json
try:
    d = json.load(sys.stdin)['device_info'][0]['telemetry']; print('aiclk', d['aiclk'].strip(), 'temp', d['asic_temperature'].strip(), 'power', d['power'].strip())
except Exception as e: print('n/a')")"
  sleep 1
done > $E/clock_trace_8layers_tt-smi.txt
wait
grep -E "layers:|trace rc" $E/clock_trace_8layers.log
echo "=== done $(date -u +%FT%TZ)"
