#!/usr/bin/env bash
# One emulator leg of the Quasar pool perf A/B:  run_leg.sh <leg> <tree> [qpool_emu_perf.py args]
#   e.g. run_leg.sh after  /localdev/wransom/tt-metal-emu
#        run_leg.sh before /localdev/wransom/tt-metal-emu-before
# Env: ZeBu emu-quasar-2x3 via TT_METAL_SIMULATOR (the umd run.sh ssh's to soc-l-04 and the emulator
# calls back on NNG_SOCKET_ADDR = this IRD container's Debuda port). Device profiler on, fresh JIT cache
# per leg, one process = one ZeBu job. Logs/results in /localdev/wransom/qpool_emu_perf/results/<leg>/.
set -uo pipefail
LEG=$1; TREE=$2; shift 2
ROOT=/localdev/wransom/qpool_emu_perf
OUT=$ROOT/results/$LEG; mkdir -p "$OUT"

export TT_METAL_HOME=$TREE TT_METAL_RUNTIME_ROOT=$TREE PYTHONPATH=$TREE/ttnn:$TREE
export TT_METAL_SIMULATOR=${TT_METAL_SIMULATOR:-/localdev/wransom/tt-umd-simulators/build/emu-quasar-2x3/}
export NNG_SOCKET_ADDR=tcp://yyzc-swc08:${P_USER_DBD_PORT:-51176}
export NNG_SOCKET_LOCAL_PORT=5555
export TT_METAL_ENV=dev TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DEVICE_PROFILER=1
# Without MID_RUN_DUMP the profiler only writes profile_log_device.csv at device close, so the
# per-call ReadDeviceProfiler drains would attribute nothing.
export TT_METAL_PROFILER_MID_RUN_DUMP=1
export TT_METAL_CACHE=$ROOT/cache_$LEG
unset TT_METAL_LLK_ASSERTS TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS TT_METAL_DPRINT_CORES TT_METAL_WATCHER TT_METAL_FORCE_JIT_COMPILE 2>/dev/null
source /localdev/wransom/tt-metal-emu/python_env/bin/activate

{
  echo "##### $(date -u) LEG $LEG tree=$TREE ($(git -C "$TREE" log -1 --format=%h)) sim=$TT_METAL_SIMULATOR #####"
  echo "TT_METAL_CACHE=$TT_METAL_CACHE  NNG_SOCKET_ADDR=$NNG_SOCKET_ADDR"
  "$HOME/zebu_preflight.sh"
  cd "$TREE" || exit 1
  timeout 14400 python "$ROOT/qpool_emu_perf.py" --leg "$LEG" --out "$OUT" "$@"
  rc=$?
  echo "##### $(date -u) LEG $LEG EXIT $rc #####"
  # the umd emulator launcher writes emu_<timestamp>_.log into the cwd; keep the newest with the leg
  newest=$(ls -t "$TREE"/emu_*.log 2>/dev/null | head -1); [ -n "$newest" ] && cp "$newest" "$OUT/"
  exit $rc
} 2>&1 | tee "$OUT/run.log"
