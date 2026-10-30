#!/bin/bash
# Where each router ERISC spends its cycles (2026-10-29; V4_PROFILE.md): PC sampling (pc_sampler.py over the debug bus:
# no router code, rate unchanged within 0.01 cycles/packet) of v4trm and v3qm on one representative row per gap, through
# pcsv_runs.sh (fresh cache per router; compile commands logged for the -g twins).
#   prof_pcs_runs.sh two   two ERISCs: ew4352, hop3, mcast (2D), lin1 (1D), ring1, halfring (ring), torusns (Y-torus)
#   prof_pcs_runs.sh one   one ERISC: ew4352, lin1, nbr (NeighborExchange), ringbidir; mux mode muxfullring
# Labels pcsv_p<3|4><mode>_<row>_13129088_trial, samples results/pcsv_p<3|4><mode>_<row>_samples.*.
M=$1
case $M in
    two) ROWS="ew4352 hop3 mcast lin1 ring1 halfring torusns"; ENVX="" ;;
    one) ROWS="ew4352 lin1 nbr ringbidir muxfullring"; ENVX="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1" ;;
    *) echo "usage: prof_pcs_runs.sh two|one"; exit 1 ;;
esac
P=/home/aagarwal/tt-metal/fabric_exp/consolidation/pcsv_runs.sh
for v in v4trm:p4 v3qm:p3; do
    echo "##### ${v%%:*}, $M ($(date +%T))"
    EXTRA_ENV="$ENVX" bash $P fabric_exp/variants/${v%%:*} ${v##*:}$M 2D $ROWS || { echo STOPPED; exit 1; }
done
echo ALL_DONE
