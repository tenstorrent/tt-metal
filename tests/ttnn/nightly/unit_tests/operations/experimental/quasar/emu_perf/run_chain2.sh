#!/usr/bin/env bash
# Diagnostic leg after the main chain: after tree with the whole-ring identity fill removed.
R=/localdev/wransom/qpool_emu_perf
until grep -q 'CHAIN DONE' $R/logs/chain.log; do sleep 20; done
echo "##### $(date -u) main chain done; starting after_nofill (diagnostic) #####"
$R/run_leg.sh after_nofill /localdev/wransom/tt-metal-emu-after-nofill > /dev/null 2>&1; echo "after_nofill rc=$?"
echo "##### $(date -u) CHAIN2 DONE #####"
