#!/usr/bin/env bash
# One arm per process, each with a bounded timeout, each followed by a reset if it hung.
set -u
ROOT=/home/ttuser/dev/muse-glimmer/tt-metal/models/autoports/meta_models_muse_glimmer_30b
LOGS=$ROOT/doc/vllm_integration/logs
TTSMI=/home/ttuser/.tenstorrent-venv/bin/tt-smi
mkdir -p "$LOGS"
for arm in prefill3_decode1 prefill1_decode3 prefill3_drain prefill3_decode3; do
  echo "=== $arm $(date -u +%H:%M:%S) ==="
  timeout 420 python "$ROOT/doc/vllm_integration/bench/multi_slot_bisect.py" \
      --arm "$arm" --out "$ROOT/doc/vllm_integration/bisect_$arm.json" \
      > "$LOGS/bisect_$arm.log" 2>&1
  rc=$?
  echo "ARM $arm rc=$rc  $(grep -c ARM_OK "$LOGS/bisect_$arm.log") ok-marker(s)"
  pkill -9 -f multi_slot_bisect.py 2>/dev/null
  sleep 5
  if [ $rc -ne 0 ]; then
    echo "resetting after $arm"
    timeout 240 $TTSMI -r >/dev/null 2>&1
    sleep 5
  fi
done
echo "BISECT_DONE"
