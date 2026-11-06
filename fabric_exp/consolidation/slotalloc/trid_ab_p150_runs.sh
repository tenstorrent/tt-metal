#!/bin/bash
# Transaction-ID allocation A/B on the P150 (TRID_ALLOCATION.md §6.2; broker job on a09u32 after
# host_rebuild_tridalloc.sh, broker 649): one pass per job, routers swapped in turn, -O3, study settings, BW telemetry
# off, hang capture armed (a hang or a failure stops the job).
#   trid_ab_p150_runs.sh PASS       PASS a: routers a11 idoff id; b: id idoff a11, rows reversed (ROUTERS overrides)
# Routers (the host's TT_FABRIC_EXP_TRID_ALLOC decides the IDs; TT_FABRIC_EXP_LOG_SLOTS=1 logs them for E1):
#   a11    variants/v4txa11 (today's uniform IDs, compiled in; the host's switch off)
#   idoff  variants/v4txa11id with the switch off: the uniform IDs through the new per-channel args, so a11 -> idoff
#          isolates the code change and idoff -> id the ID counts (CLAUDE.md: a cause needs the variant changing only it)
#   id     variants/v4txa11id with TT_FABRIC_EXP_TRID_ALLOC=1
# Rows, at the default slots unless named:
#   ring   frgap/p150_it_ring.yaml (Ring: 1-hop unicast, full- and half-ring multicast; 3072 / 4352 B): lone ring
#          receiver 32a, IDs 8 -> 16
#   t24    torus/torusxy_2x4_sizes.yaml (XY torus 2x4; wrapped receivers 16a, 8 -> 16)
#   t42    torus/torusxy_4x2_sizes.yaml on the 4x2 XY torus (study descriptor)
#   vc1    the VC1 split at the dynamic 3072 B slots, floor off (yamls/sls_vc1_all_3072.yaml, two ranks, 82 finishes):
#          30,14,16,7,6, VC1 receiver 6: IDs 4 -> 6 (VC0 16: 4 -> 4)
#   vc1f   the same with the VC0 receiver floor (default policy): 26,10,32,7,6, VC0 32 | VC1 6: IDs 4 | 4 -> 8 | 6
#   e2d    census subset, 2D: slotstudy/sls_2d_p150.yaml (14 tests), 17,16,16: IDs 16 -> 4 (expected neutral)
#   e1d    census subset, 1D: slotstudy/sls_1d_p150.yaml (6 tests), 26,25,32: IDs 16 -> 8 (expected neutral)
# After each run the logged IDs must be the router's (a11 / idoff: uniform-rule lines only; id: assigned only), and the
# vc1 / vc1f rows' logged slots their row. Labels <TAG><PASS>_<router>_<row> (results/<label>.log, .csv, _tests.csv;
# vc1 rows <label>_13129088_vc1.*), TAG default tridab. Fresh caches per router (and per VC1 row) each pass.
# Runs are not started past TRID_BUDGET s (default 1300) of job time.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
P=${1:-}
case $P in a) R=${ROUTERS:-a11 idoff id}; ROWS="ring t24 t42 vc1 vc1f e2d e1d" ;;
           b) R=${ROUTERS:-id idoff a11}; ROWS="e1d e2d vc1f vc1 t42 t24 ring" ;;
           *) echo "usage: [ROUTERS=...] trid_ab_p150_runs.sh a|b"; exit 1 ;; esac
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS_VC2 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D \
    TT_FABRIC_EXP_TIER_SLOTS TT_FABRIC_EXP_DYN_SLOTS TT_FABRIC_EXP_DYN_SLOTS_POLICY TT_FABRIC_EXP_TRID_ALLOC \
    TT_FABRIC_EXP_DRISC_DELAY TT_MESH_GRAPH_DESC_PATH TT_FABRIC_PHASE_PROFILE TT_FABRIC_PHASE_PROFILE_CSV
for k in TT_FABRIC_EXP_TRID_ALLOC vc0_receiver_floor; do
    grep -aq $k build_Release/tt_metal/libtt_metal.so || { echo "!!! the host library has no $k (host_rebuild_tridalloc.sh)"; exit 1; }
done
TAG=${TAG:-tridab}
SA=$CON/slotalloc
MGD=TT_MESH_GRAPH_DESC_PATH=$PWD/fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto
for f in $CON/frgap/p150_it_ring.yaml fabric_exp/torus/torusxy_2x4_sizes.yaml fabric_exp/torus/torusxy_4x2_sizes.yaml \
         ${MGD#*=} $SA/yamls/sls_vc1_all_3072.yaml fabric_exp/slotstudy/sls_2d_p150.yaml fabric_exp/slotstudy/sls_1d_p150.yaml; do
    [ -f $f ] || { echo "no $f"; exit 1; }
done
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
clean() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
ids() {  # log router: the IDs the host logged are the router's
    local na nu
    na=$(clean $1 | grep -ac "receiver transaction IDs (TT_FABRIC_EXP_TRID_ALLOC)")
    nu=$(clean $1 | grep -ac "receiver transaction IDs (uniform rule)")
    echo "   transaction IDs: $na assigned, $nu uniform-rule lines; $(clean $1 |
        grep -aoE 'RX_CH_[0-9] G [0-9]+ ids [0-9]+ start [0-9]+' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    case $2 in
        id) [ $na -gt 0 ] && [ $nu = 0 ] ;;
        *) [ $nu -gt 0 ] && [ $na = 0 ] ;;
    esac || { echo "!!! the logged IDs are not router $2's"; return 1; }
}
slots() {  # log W,F,R,F1,R1: the VC1 split's logged slots
    local w f r f1 r1 v0 v1
    IFS=, read w f r f1 r1 <<< "$2"
    v0="VC0 sender slots {$w, $f, $f, $f} receiver slots {$r}"; v1="VC1 sender slots {$f1, $f1, $f1} receiver slots {$r1}"
    clean $1 | grep -aqF "$v0" && clean $1 | grep -aqF "$v1" && { echo "   slots: $2"; return 0; }
    echo "!!! slots not $2; logged:"; clean $1 | grep -aoE 'VC[0-2] sender slots \{[^}]*\} receiver slots \{[^}]*\}' | sort | uniq -c
    return 1
}
budget() { [ $SECONDS -le ${TRID_BUDGET:-1300} ] || { echo "=== budget: not starting $1 or later at ${SECONDS} s"; stop; }; }
row() {  # router row env...
    local r=$1 w=$2 l=${TAG}${P}_$1_$2 c=/home/aagarwal/.tt-metal-cache-$TAG-$1 rc log
    shift 2
    budget $l
    case $w in
        vc1 | vc1f)
            local e="TT_FABRIC_EXP_DYN_SLOTS=1" want=30,14,16,7,6
            [ $w = vc1 ] && e="$e TT_FABRIC_EXP_DYN_SLOTS_POLICY=vc0_receiver_floor=off" || want=26,10,32,7,6
            VC1CAP_EXTRA_ENV="TT_FABRIC_EXP_LOG_SLOTS=1 $e $*" VC1CAP_SPLIT_YAML=$SA/yamls/sls_vc1_all_3072.yaml \
                VC1CAP_SPLIT_FINISHES=82 VC1CAP_SPLIT_TIMEOUT=300 run_vc1_cap $l $c-$w O3 || stop
            log=$out/${l}_13129088_vc1.log
            slots $log $want || stop ;;
        *)
            local y n=""
            case $w in
                ring) y=$CON/frgap/p150_it_ring.yaml ;;
                t24) y=fabric_exp/torus/torusxy_2x4_sizes.yaml ;;
                t42) y=fabric_exp/torus/torusxy_4x2_sizes.yaml; set -- "$@" $MGD ;;
                e2d) y=fabric_exp/slotstudy/sls_2d_p150.yaml; n=14 ;;
                e1d) y=fabric_exp/slotstudy/sls_1d_p150.yaml; n=6 ;;
            esac
            rm -f generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv
            run_cap $l $y $c 600 TT_FABRIC_EXP_LOG_SLOTS=1 "$@"; rc=$?
            [ -f generated/fabric/bandwidth_results_blackhole.csv ] && cp generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv
            [ $rc = 2 ] && { echo "!!! $l: hang (captured): stopping"; stop; }
            [ -z "$n" ] || clean $out/$l.log | grep -aq "Tests: $n executed, 0 failed" || rc=1
            [ $rc = 0 ] || { echo "!!! $l failed (rc $rc)"; stop; }
            log=$out/$l.log ;;
    esac
    ids $log $r || stop
}
for r in $R; do
    case $r in a11) v=v4txa11; env=() ;; idoff) v=v4txa11id; env=() ;; id) v=v4txa11id; env=(TT_FABRIC_EXP_TRID_ALLOC=1) ;;
               *) echo "!!! unknown router $r"; exit 1 ;; esac
    echo "##### pass $P, router $r ($v ${env[*]}) (job at ${SECONDS} s)"
    c=/home/aagarwal/.tt-metal-cache-$TAG-$r
    rm -rf "${c:?}" "${c:?}"-vc1 "${c:?}"-vc1f; mkdir -p $c
    python3 $CON/swap.py apply fabric_exp/variants/$v $BK || stop
    for w in $ROWS; do row $r $w "${env[@]}"; done
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
echo "ALL_DONE pass $P (job at ${SECONDS} s)"
