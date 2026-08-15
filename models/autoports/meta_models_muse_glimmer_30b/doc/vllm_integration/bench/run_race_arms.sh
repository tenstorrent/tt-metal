#!/usr/bin/env bash
set -u
ROOT=/home/ttuser/dev/muse-glimmer/tt-metal/models/autoports/meta_models_muse_glimmer_30b
LOGS=$ROOT/doc/vllm_integration/logs
TTSMI=/home/ttuser/.tenstorrent-venv/bin/tt-smi
P="$ROOT/doc/vllm_integration/bench/adapter_probe.py"
run () {
  name=$1; to=$2; shift 2
  echo "=== $name $(date -u +%H:%M:%S) ==="
  timeout "$to" python "$P" --out "$ROOT/doc/vllm_integration/probe_$name.json" "$@" \
      > "$LOGS/probe_$name.log" 2>&1
  rc=$?
  echo "ARM $name rc=$rc"
  pkill -9 -f adapter_probe.py 2>/dev/null; sleep 5
  if [ $rc -ne 0 ]; then timeout 240 $TTSMI -r >/dev/null 2>&1; sleep 5; fi
}
R="--layers 0,3 --kv-token-budget 262144"
run stale_repeat1 420 $R
run stale_drain   420 $R --drain-per-step
run stale_sync    420 $R --read-mode sync
echo "RACE_ARMS_DONE"
