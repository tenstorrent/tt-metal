#!/bin/bash
# Runs unary_grid_probe.py in the Custom test dispatch container (repo at /work), on whatever commit the job built.
# Usage: run_probe.sh timing|kernel [unary_grid_probe.py args]
#   timing: trace replay and eager time per op (use a non-tracy build)
#   kernel: DEVICE KERNEL DURATION and CORE COUNT per config from the ops report (needs a tracy build)
set -u
MODE=$1
shift
cd /work
echo "=== PROBE commit $(git rev-parse HEAD 2>/dev/null) host $(hostname) mode $MODE"
for f in /sys/class/tenstorrent/*/tt_card_type /sys/class/tenstorrent/*/tt_serial /sys/class/tenstorrent/*/tt_fw_bundle_ver \
         /sys/class/tenstorrent/*/tt_aiclk /sys/module/tenstorrent/version; do
    [ -r "$f" ] && echo "MACHINE $f = $(cat "$f" 2>/dev/null)"
done
BASE=https://raw.githubusercontent.com/tenstorrent/tt-metal/virdhatchani/unary-small-grid-probe/tests/scripts/unary_small_grid
curl -fsSL "$BASE/unary_grid_probe.py" -o /work/unary_grid_probe.py || exit 1
if [ "$MODE" = kernel ]; then
    python3 -m tracy -r -p -v /work/unary_grid_probe.py --mode kernel "$@" > /work/probe_kernel.log 2>&1
    echo "=== tracy exit $?"
    grep -E "^(PROBE|KCFG|FAIL)" /work/probe_kernel.log
    grep -E "^(PROBE|KCFG|FAIL)" -q /work/probe_kernel.log || tail -60 /work/probe_kernel.log
    python3 /work/unary_grid_probe.py --parse '/work/generated/profiler/reports/**/ops_perf_results*.csv'
else
    python3 /work/unary_grid_probe.py --mode timing "$@"
fi
echo "=== PROBE done"
