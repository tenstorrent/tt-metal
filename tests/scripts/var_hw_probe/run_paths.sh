#!/bin/bash
# One program vs composite for var_hw / std_hw on many-plane shapes, on CI hardware: DEVICE KERNEL DURATION per call
# (RESULT lines) and per-iteration wall time under trace replay (TRACE lines). Runs from /work in Custom test dispatch.
set -u
cd /work
echo "=== PROBE commit $(git rev-parse HEAD 2>/dev/null) host $(hostname)"
for f in /sys/class/tenstorrent/*/tt_card_type /sys/class/tenstorrent/*/tt_serial /sys/class/tenstorrent/*/tt_fw_bundle_ver; do
    [ -r "$f" ] && echo "MACHINE $f = $(cat "$f" 2>/dev/null)"
done
B=https://raw.githubusercontent.com/tenstorrent/tt-metal/virdhatchani/var_hw-probe/tests/scripts/var_hw_probe
curl -fsSL "$B/paths_sweep.py" -o /work/paths_sweep.py && curl -fsSL "$B/sum_signposts.py" -o /work/sum_signposts.py || exit 1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/work/tools"
MODE=kernel python -m tracy -r -p -v -o /work/paths_report /work/paths_sweep.py > /work/paths_kernel.log 2>&1
echo "=== tracy exit $?"
grep -E "^SKIP|Traceback|Error" /work/paths_kernel.log | head -20
csv=$(find /work/paths_report -name "ops_perf_results*.csv" | head -1)
[ -n "$csv" ] && python /work/sum_signposts.py "$csv" | sed 's/^/RESULT /'
MODE=trace python /work/paths_sweep.py > /work/paths_trace.log 2>&1
echo "=== trace exit $?"
grep -E "^TRACE|^SKIP" /work/paths_trace.log
grep -E "Traceback|Error" /work/paths_trace.log | head -10
