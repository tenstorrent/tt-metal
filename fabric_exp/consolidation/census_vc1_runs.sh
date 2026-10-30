#!/bin/bash
# Perf census, the VC1 split (2026-10-29; the VC0 suites are census_runs.sh). The P150x8 as two 2x2 meshes, one MPI
# rank each (vc1_lib.sh): vc1_alloc.yaml (hang capture armed) and vc1_inter.yaml, per router, same job:
#   main_fw   main's router as shipped (kernel v1, flags 0, firmware keepalive), the builder's opt level
#   v3qm      the v3 lead with the post-merge header path, -O3, study settings
#   v4trm     the v4 lead, -O3, study settings (its 22 VC1 sender images differ from v4rs's by three stack slots)
# census_vc1_runs.sh PASS; labels cen<pass>vc1[i]_<router>_default_p1_<opt>. A hang stops the job.
# ROUTERS (default "main_fw v3qm v4trm"): main_fw or variant directory names.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
P=$1
[ -n "$P" ] || { echo "usage: census_vc1_runs.sh PASS"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
MAIN="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"

for r in ${ROUTERS:-main_fw v3qm v4trm}; do
    echo "##### $r (job at ${SECONDS} s)"
    c=/home/aagarwal/.tt-metal-cache-cenvc1-$r
    rm -rf "${c:?}"; mkdir -p $c
    if [ $r = main_fw ]; then
        KA_ENV= VC1CAP_EXTRA_ENV="$MAIN" VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 \
            run_vc1_cap cen${P}vc1_main_fw_default_p1_auto $c auto || { echo STOPPED; exit 1; }
        KA_ENV= VC1CAP_EXTRA_ENV="$MAIN" VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 \
            run_vc1 cen${P}vc1i_main_fw_default_p1_auto $c auto || echo "!!! inter rows failed"
        continue
    fi
    python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
    VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 \
        run_vc1_cap cen${P}vc1_${r}_default_p1_o3 $c O3 || stop
    VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 run_vc1 cen${P}vc1i_${r}_default_p1_o3 $c O3 \
        || echo "!!! inter rows failed"
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
