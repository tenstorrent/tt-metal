#!/bin/bash
# Job 095 was cut at the broker's 600 s default before its controls' suites ran. This job: v4gir (v4gi + only the READ
# bit in the slot records, no proof for groups no command read; v4gi's timing) with the VC1 split at -O3, then every
# VC0 suite for v4gir and the controls v4gi (label v4gic) and v4vc (v4vcd), as v4x_runs.sh. v4gi2 and v4gi3 have their
# suites from job 095.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
VC1_RUN=run_vc1_cap vc1_variant v4gir fabric_exp/variants/v4gir v4gir O3
bash $CON/v4x_runs.sh v4gir v4gi:v4gic v4vc:v4vcd || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
