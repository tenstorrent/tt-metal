#!/bin/bash
# Step 2 of the sender cleanup (V4_SENDER_SHAPE.md; 2026-10-29): variants/v4sh2 = v4sh1b with one free-slots register
# read for every VC per pass and one in-flight load for both VCs' forwarding groups. Only VC1 routers change (every
# other image equals v4sh1b's but for one commuted compare), so this measures the VC1 rows, same job, two routers back
# to back per pass: the VC1 split (vc1_alloc.yaml, data-checked, hang capture armed), the inter-mesh rows
# (vc1_inter.yaml) and mixed VC0 + VC1 load on one link (vc1_mix.yaml: where a VC1 packet may now wait behind VC0's
# visits). sh2_vc1_runs.sh PASS; labels sh2<pass>vc1{,i,m}_<router>_default_p1_o3.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
P=$1
[ -n "$P" ] || { echo "usage: sh2_vc1_runs.sh PASS"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
for r in v4sh1b v4sh2; do
    echo "##### $r (job at ${SECONDS} s)"
    c=/home/aagarwal/.tt-metal-cache-sh2vc1-$r
    rm -rf "${c:?}"; mkdir -p $c
    python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
    VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 \
        run_vc1_cap sh2${P}vc1_${r}_default_p1_o3 $c O3 || stop
    VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 run_vc1 sh2${P}vc1i_${r}_default_p1_o3 $c O3 \
        || echo "!!! inter rows failed"
    VC1_YAML=fabric_exp/vc1/vc1_mix.yaml VC1_TIMEOUT=180 run_vc1 sh2${P}vc1m_${r}_default_p1_o3 $c O3 \
        || echo "!!! mix rows failed"
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
