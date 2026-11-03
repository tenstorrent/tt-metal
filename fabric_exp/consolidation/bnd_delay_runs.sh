#!/bin/bash
# Bound map on the lead (2026-11-01; V4_BOUNDS.md): prof_delay_runs.sh's one-side spins on variants/v4bdd (v4rxv2i3 +
# v4trmd's hooks: TT_FABRIC_EXP_TX_DELAY after each send, TT_FABRIC_EXP_RX_DELAY after each packet received,
# TT_FABRIC_EXP_DRISC_DELAY per pass, low 16 bits sender / high 16 bits receiver; at 0 every image equals v4rxv2i3's,
# checked offline at -O3 and -Os). Per group, one job, one cache:
#   bnd_delay_runs.sh two    two ERISCs: the prof_delay_runs.sh suites (4352, 1d, mix, ring, torusy), then VC2 on
#                            (pcs3 rows ew4352, ewscatter, torusns, torusns_scatter: vc2_<row>) and the VC2 traffic rows
#                            (vc2ew4352, vc02ew4352, vc02bew4352: v2t_<row>; per-sender rates in the logs, vc2rows_report.py)
#   bnd_delay_runs.sh one    one ERISC: 4352, 1d, ring (exp_ring_se); mux mode exp_mux
#   bnd_delay_runs.sh vc1    the VC1 split: vc1_alloc.yaml (hang capture armed) and vc1_inter.yaml, two MPI ranks
#   bnd_delay_runs.sh trim   channel-trimmed rows (2026-11-03, C0 trimmed-replay A/B): txc/trim_fwd_4352.yaml, Mesh3HopEW
#                            and MeshMulticast, each a capture pass then a trimmed replay (bnd_report.py: replay = '#1')
# ARMS (default "base tx32 rx32", the cheap pass): base, tx<N>, rx<N>, sp<N>, rp<N> (per-pass spins). SUITES limits the
# group to some suites (names as in the labels: 4352 1d mix ring torusy vc2 v2t | 4352_1e 1d_1e rse mux | vc1 vc1i).
# Labels bnd_<suite>_<arm> (VC2 rows: bnd_vc2_<row>_<arm>, bnd_v2t_<row>_<arm>; VC1: bndvc1[i]_<arm>_default_p1_o3).
# Slot rows logged on VC2 runs (17/14/16 + 2/4 at 4352 B). A hang stops the job.
# BND_VARIANT (default v4bdd) picks the hook variant; BND_TAG (default empty) goes into every label after "bnd"
# (bnd<TAG>_<suite>_<arm>, bnd<TAG>vc1[i]_<arm>_...), so a new router's runs (and their passes) keep their own files;
# bnd_report.py --tag TAG reads them. DRY=1 prints the steps only (no swap, no cache change, no device).
if [ "${DRY:-0}" = 1 ]; then
    # nothing sourced: jobs_lib.sh restores any active swap when sourced and on exit (a broker job's included)
    CON=/home/aagarwal/tt-metal/fabric_exp/consolidation; out=/home/aagarwal/tt-metal/fabric_exp/results
    BK=/home/aagarwal/.tt-metal-p1-swap-backup; M=
else
    source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
    source $CON/cap_lib.sh
fi
G=$1
case $G in two|one|vc1|trim) ;; *) echo "usage: [DRY=1] [ARMS=...] [SUITES=...] bnd_delay_runs.sh two|one|vc1|trim"; exit 1 ;; esac
[ $G = vc1 ] && [ "${DRY:-0}" != 1 ] && { source $CON/vc1_lib.sh; source $CON/vc1_capture_lib.sh; }
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS \
    TT_FABRIC_EXP_SLOTS_VC2
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
VC2="TT_METAL_ENABLE_FABRIC_VC2=1 TT_FABRIC_EXP_LOG_SLOTS=1"
ARMS=${ARMS:-base tx32 rx32}
T=${BND_TAG:-}
C=/home/aagarwal/.tt-metal-cache-bnd-$G
if [ "${DRY:-0}" = 1 ]; then
    run_cap() { echo "   run_cap $*"; return 0; }
    swp() { echo "   swap.py $*"; }
    run_vc1_cap() { echo "   run_vc1_cap $* (env: $VC1CAP_EXTRA_ENV)"; return 0; }
    run_vc1() { echo "   run_vc1 $* (env: $VC1CAP_EXTRA_ENV)"; return 0; }
else
    swp() { python3 $CON/swap.py "$@"; }
    rm -rf "${C:?}"; mkdir -p $C
fi
stop() { swp restore $BK; echo STOPPED; exit 1; }
want() { [ -z "$SUITES" ] || [[ " $SUITES " == *" $1 "* ]]; }
arm_env() {
    case $1 in
        base) echo "" ;; tx*) echo "TT_FABRIC_EXP_TX_DELAY=${1#tx}" ;; rx*) echo "TT_FABRIC_EXP_RX_DELAY=${1#rx}" ;;
        sp*) echo "TT_FABRIC_EXP_DRISC_DELAY=${1#sp}" ;; rp*) echo "TT_FABRIC_EXP_DRISC_DELAY=$(( ${1#rp} << 16 ))" ;;
    esac
}
sweep() {  # label-stem yaml timeout [ENV...]
    local s=$1 y=$2 t=$3 a rc; shift 3
    for a in $ARMS; do
        [ "${DRY:-0}" = 1 ] || rm -f generated/fabric/bandwidth_results_blackhole.csv
        run_cap bnd${T}_${s}_$a $y $C $t $(arm_env $a) "$@"; rc=$?
        [ "${DRY:-0}" != 1 ] && [ -f generated/fabric/bandwidth_results_blackhole.csv ] && \
            cp generated/fabric/bandwidth_results_blackhole.csv $out/bnd${T}_${s}_${a}_detail.csv
        [ $rc = 2 ] && stop
    done
}
swp apply fabric_exp/variants/${BND_VARIANT:-v4bdd} $BK || stop
case $G in
    two)
        want 4352 && sweep 4352 $(suite_yaml 4352) 300
        want 1d && sweep 1d $(suite_yaml 1d) 300
        want mix && sweep mix $(suite_yaml mix) 300
        want ring && sweep ring fabric_exp/v4/exp_ring.yaml 200
        want torusy && sweep torusy fabric_exp/v4/exp_torus_y.yaml 200
        if want vc2; then
            for row in ${VC2_ROWS:-ew4352 ewscatter torusns torusns_scatter}; do
                sweep vc2_$row fabric_exp/pcs3/$row.yaml 200 $VC2
            done
        fi
        if want v2t; then
            for row in ${V2T_ROWS:-vc2ew4352 vc02ew4352 vc02bew4352}; do
                sweep v2t_$row fabric_exp/pcs3/$row.yaml 300 $VC2
            done
        fi ;;
    one)
        want 4352_1e && sweep 4352_1e $(suite_yaml 4352) 300 $E
        want 1d_1e && sweep 1d_1e $(suite_yaml 1d) 300 $E
        want rse && sweep rse fabric_exp/v4/exp_ring_se.yaml 300 $E
        want mux && sweep mux fabric_exp/v4/exp_mux.yaml 400 ;;
    vc1)
        for a in $ARMS; do
            env_a=$(arm_env $a)
            if want vc1; then
                VC1CAP_EXTRA_ENV="$env_a" VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 \
                    run_vc1_cap bnd${T}vc1_${a}_default_p1_o3 $C O3 || stop
            fi
            if want vc1i; then
                VC1CAP_EXTRA_ENV="$env_a" VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 \
                    run_vc1 bnd${T}vc1i_${a}_default_p1_o3 $C O3 || stop
            fi
        done ;;
    trim)
        want trim && sweep trim fabric_exp/consolidation/txc/trim_fwd_4352.yaml 300 ;;
esac
swp restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
