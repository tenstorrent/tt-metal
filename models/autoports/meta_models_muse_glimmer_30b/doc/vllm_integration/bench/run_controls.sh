#!/usr/bin/env bash
# Controls for the reduced-target multi-slot hang, one arm per process.
#   sync   : read_from_device=True (no async split)  -> implicates the deferred read
#   drain  : ttnn.synchronize_device after each step -> implicates submission rate
set -u
ROOT=/home/ttuser/dev/muse-glimmer/tt-metal/models/autoports/meta_models_muse_glimmer_30b
LOGS=$ROOT/doc/vllm_integration/logs
TTSMI=/home/ttuser/.tenstorrent-venv/bin/tt-smi
P="$ROOT/doc/vllm_integration/bench/adapter_probe.py"
run () {
  name=$1; shift
  echo "=== $name $(date -u +%H:%M:%S) ==="
  timeout 420 python "$P" --layers 0,3 --kv-token-budget 262144 \
      --out "$ROOT/doc/vllm_integration/probe_$name.json" "$@" > "$LOGS/probe_$name.log" 2>&1
  echo "ARM $name rc=$?"
  pgrep -f adapter_probe.py | xargs -r kill -9; sleep 5
  timeout 240 $TTSMI -r >/dev/null 2>&1; sleep 5
}
run ctl_sync  --read-mode sync
run ctl_drain --drain-per-step
echo "CONTROLS_DONE"
