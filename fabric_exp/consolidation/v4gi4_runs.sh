#!/bin/bash
# v4gi4: v4gi with the due unit relative to the oldest group (the rest of it after a partial credit, so credits realign
# at once), computed as T - ((next_index - waiting) & (T - 1)); no READ bit (v4gi2/v4gi3/v4gir: the bit costs E1). Rings
# and tori compile to v4gi's code. Against v4gi (label v4gie) in the same job:
#   1. the VC1 split suite at -O3, hang capture armed;
#   2. VC0-only builds: data-checked validation, then every suite and the latency rows, as v4x_runs.sh;
#   3. the long N/S burst rows (pcs3/burstns8, burstns32) under the PC sampler, as job 094 (labels pcsv_v4gi4b_*).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
VC1_RUN=run_vc1_cap vc1_variant v4gi4 fabric_exp/variants/v4gi4 v4gi4 O3
VC1_RUN=run_vc1_cap vc1_variant v4gie fabric_exp/variants/v4gi v4gie O3
bash $CON/v4x_runs.sh v4gi4 v4gi:v4gie || { echo STOPPED; exit 1; }
bash $CON/pcsv_runs.sh fabric_exp/variants/v4gi4 v4gi4b 2D burstns8 burstns32 || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
