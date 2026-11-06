#!/bin/bash
# Offline gate of the single-ERISC A/B (single_ab_runs.sh; local, never committed), on a09u32, mock clusters only:
#   offline.sh mock A|B [CASE...]   port_tune/cov/cov.py mock runs (each run's JIT builds every kernel, compile commands
#                                   and the JIT's program-size telemetry logged) of arm A (/home/aagarwal/tt-metal-single-ab,
#                                   ba486eb70ba + the telemetry switch, its own host build) or arm B (the farm
#                                   /home/aagarwal/tt-metal-single-in, mk_arm_b.py), COV dirs of equal path length
#   offline.sh cmp [CASE...]        port_tune/cov/covcmp_sig.py A vs B (builds matched by signature across the roots,
#                                   compared by loadable bytes)
#   offline.sh fit [CASE...]        per case and arm: the mock's result, its least ACTIVE_ETH config-buffer headroom
#                                   (JIT telemetry) and any "too large for kernel config buffer"
# Cases default to ONE + TWO below.
set -uo pipefail
H=/home/aagarwal/tt-metal/fabric_exp/port_tune/cov
declare -A TREE=([A]=/home/aagarwal/tt-metal-single-ab [B]=/home/aagarwal/tt-metal-single-in)
declare -A COVD=([A]=/home/aagarwal/.tt-metal-sab-A-cov [B]=/home/aagarwal/.tt-metal-sab-B-cov)
ONE="bh_1d_1erisc bh_2d_1erisc bh_ring_1erisc bh_torus_1erisc bh_neighbor_1erisc bh_1d_mux bh_2d_mux bh_ring_mux
     bh_2d_vc2_1erisc bh_vc1_1erisc bh_vc1_vc2_1erisc_Os_log bh_2d_1erisc_streaming glx_express_1erisc glx2_mesh_novc_1erisc"
TWO="bh_1d bh_2d bh_ring bh_torus bh_neighbor bh_2d_vc2 bh_torus_vc2 bh_vc1 bh_vc1_vc2 bh_2d_trim glx_2d glx_torus"
what=$1; shift
case $what in
    mock)
        arm=$1; shift
        cases=${*:-$ONE $TWO}
        TT_COV_TREE=${TREE[$arm]} TT_COV_DIR=${COVD[$arm]} TT_COV_EXTRA_ENV='{"TT_METAL_LOG_JIT_TELEMETRY": "1"}' \
            python3 $H/cov.py mock --jobs ${SAB_JOBS:-3} $cases ;;
    cmp)
        python3 $H/covcmp_sig.py ${TREE[A]} ${COVD[A]} ${TREE[B]} ${COVD[B]} ${*:-$ONE $TWO} ;;
    fit)
        for c in ${*:-$ONE $TWO}; do
            for arm in A B; do
                l=${COVD[$arm]}/cases/$c/mock.log
                [ -f $l ] || { printf '%-26s %s  no mock\n' $c $arm; continue; }
                s=$(sed 's/\x1b\[[0-9;]*m//g' $l)
                hr=$(grep -aoE "ACTIVE_ETH config buffer, least headroom +[0-9]+ B left \([0-9]+ of [0-9]+ B used\)" <<< "$s" | sed 's/.*headroom *//' | head -1)
                big=$(grep -ac "too large for kernel config buffer" <<< "$s")
                routers=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sum('/fabric_erisc_router' in k for k in d))" ${COVD[$arm]}/cases/$c/cmds.json 2>/dev/null)
                printf '%-26s %s  router builds %4s  ACTIVE_ETH least headroom: %-34s too-large errors: %s\n' $c $arm "${routers:-?}" "${hr:-not logged}" $big
            done
        done ;;
    *) echo "usage: offline.sh mock A|B [CASE...] | cmp [CASE...] | fit [CASE...]"; exit 1 ;;
esac
