#!/usr/bin/env bash
set -u
ROOT=/home/ttuser/dev/muse-glimmer/tt-metal/models/autoports/meta_models_muse_glimmer_30b
LOGS=$ROOT/doc/vllm_integration/logs
TTSMI=/home/ttuser/.tenstorrent-venv/bin/tt-smi
P="$ROOT/doc/vllm_integration/bench/adapter_probe.py"
run () {
  name=$1; shift
  echo "=== $name $(date -u +%H:%M:%S) ==="
  timeout 420 python "$P" --layers 0,3 --kv-token-budget 262144 \
      --out "$ROOT/doc/vllm_integration/probe_$name.json" "$@" \
      > "$LOGS/probe_$name.log" 2>&1
  rc=$?
  echo "ARM $name rc=$rc"
  pkill -9 -f adapter_probe.py 2>/dev/null
  sleep 5
  if [ $rc -ne 0 ]; then timeout 240 $TTSMI -r >/dev/null 2>&1; sleep 5; fi
}
run nostale --no-stale-inputs
run short2 --decode-steps 2
run onlyfirst --prompt-lens 128
echo "ARMS_DONE"
