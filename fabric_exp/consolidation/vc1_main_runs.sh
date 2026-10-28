#!/bin/bash
# Main's router (stock: kernel v1, flags 0, the firmware keepalive) on the VC1 split (two 2x2 meshes, fabric_exp/vc1):
# the production reference for v4's VC1 work on this host. The split suite (multi-mesh sanity, data checked, then the
# intra-mesh rows on VC1 routers), hang capture armed, at the builder's level (auto = -Os) and at -O2 (main does not fit
# at -O3), each in a fresh kernel cache.
#   vc1_main_runs.sh [fw|1000]   fw (default): the firmware keepalive, labels mainvc1_{auto,o2};
#                                1000: the study's keepalive (as the v3/v4 runs), labels mainvc1ka_{auto,o2}
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
ka=${1:-fw}
for o in auto O2; do
    if [ $ka = fw ]; then
        l=mainvc1_$(echo $o | tr A-Z a-z)
        KA_ENV= VC1CAP_EXTRA_ENV="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0" \
            run_vc1_cap $l /home/aagarwal/.tt-metal-cache-mainvc1-$o $o || { echo STOPPED; exit 1; }
    else
        l=mainvc1ka_$(echo $o | tr A-Z a-z)
        VC1CAP_EXTRA_ENV="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0" \
            run_vc1_cap $l /home/aagarwal/.tt-metal-cache-mainvc1ka-$o $o || { echo STOPPED; exit 1; }
    fi
done
rm -f $M
