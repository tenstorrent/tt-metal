#!/bin/bash
# VC1 inter-mesh rows: window-bound? (2026-11-01; V4_BOUNDS.md §1b). vc1_inter.yaml on v4rxv2i3 at the study's VC0+VC1
# row (19,9,16,5,4) and with deeper VC1 windows from the worker's spare (TT_FABRIC_EXP_SLOTS_VC1=13,8,16,7,7; same 81
# slots at 4352 B), bracketed: d, w, d. Labels bndw_<arm>[_2]_default_p1_o3 (per-test CSVs ..._vc1_tests.csv). Slots
# logged. A failure stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
C=/home/aagarwal/.tt-metal-cache-bndw
python3 $CON/swap.py apply fabric_exp/variants/v4rxv2i3 $BK || stop
VC1CAP_EXTRA_ENV="TT_FABRIC_EXP_LOG_SLOTS=1" VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 \
    run_vc1 bndw_d_default_p1_o3 $C O3 || stop
VC1CAP_EXTRA_ENV="TT_FABRIC_EXP_LOG_SLOTS=1 TT_FABRIC_EXP_SLOTS_VC1=13,8,16,7,7" VC1_YAML=fabric_exp/vc1/vc1_inter.yaml \
    VC1_TIMEOUT=180 run_vc1 bndw_w_default_p1_o3 $C O3 || stop
VC1CAP_EXTRA_ENV="TT_FABRIC_EXP_LOG_SLOTS=1" VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 \
    run_vc1 bndw_d_2_default_p1_o3 $C O3 || stop
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
