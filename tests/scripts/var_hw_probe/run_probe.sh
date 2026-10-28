#!/bin/bash
# var_hw / std_hw kernel-time sweep on CI hardware. Sums DEVICE KERNEL DURATION over every program of a call,
# the eltwise dashboard's metric, and prints one RESULT line per (op, dtype, shape).
# Runs from /work inside the Custom test dispatch container.
set -u
cd /work
echo "=== PROBE commit $(git rev-parse HEAD 2>/dev/null) host $(hostname)"
for f in /sys/class/tenstorrent/*/tt_card_type /sys/class/tenstorrent/*/tt_serial /sys/class/tenstorrent/*/tt_fw_bundle_ver; do
    [ -r "$f" ] && echo "MACHINE $f = $(cat "$f" 2>/dev/null)"
done
B=https://raw.githubusercontent.com/tenstorrent/tt-metal/virdhatchani/var_hw-probe/tests/scripts/var_hw_probe
curl -fsSL "$B/perf_final.py" -o /work/perf_final.py && curl -fsSL "$B/sum_signposts.py" -o /work/sum_signposts.py || exit 1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/work/tools"
python -m tracy -r -p -v -o /work/var_hw_report /work/perf_final.py > /work/var_hw_probe.log 2>&1
echo "=== tracy exit $?"
grep -E "^SKIP|Traceback|Error" /work/var_hw_probe.log | head -20
csv=$(find /work/var_hw_report -name "ops_perf_results*.csv" | head -1)
[ -z "$csv" ] && { tail -40 /work/var_hw_probe.log; exit 1; }
python /work/sum_signposts.py "$csv" | sed 's/^/RESULT /'
