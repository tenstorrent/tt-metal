#!/usr/bin/env bash
# Chain the remaining legs after the running 'after' leg: before -> after_t1 -> aux (profiler semantics).
R=/localdev/wransom/qpool_emu_perf
while pgrep -f 'qpool_emu_perf.py --leg after ' >/dev/null; do sleep 15; done
echo "##### $(date -u) after leg done; starting before #####"
$R/run_leg.sh before   /localdev/wransom/tt-metal-emu-before   > /dev/null 2>&1; echo "before rc=$?"
$R/run_leg.sh after_t1 /localdev/wransom/tt-metal-emu-after-t1 > /dev/null 2>&1; echo "after_t1 rc=$?"
$R/run_leg.sh aux      /localdev/wransom/tt-metal-emu --no-cases --aux > /dev/null 2>&1; echo "aux rc=$?"
echo "##### $(date -u) CHAIN DONE #####"
