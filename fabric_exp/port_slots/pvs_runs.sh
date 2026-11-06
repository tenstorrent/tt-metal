#!/bin/bash
# Port against study on one P150 LoudBox (an lb2 job; SLOT_ALLOCATOR.md section 12.12, RECONCILE_HOST.md): the same rows
# on both routers, slot rows matched, two passes with the arms reversed (pass a: study then port per suite; pass b:
# suites reversed, port then study).
#   STUDY  the lead v4txa11 swapped into this tree's v2 slot, with the study's standard settings (cap_lib.sh run_cap:
#          TT_FABRIC_KERNEL_VERSION=2, flags 13129088, keepalive 1000, -O3, BW telemetry off) and its default slots.
#   PORT   22cdb1e55c3 + fabric_exp/port_slots/slot_override.patch (port_lib.sh: its own settings, BW telemetry off),
#          the slot override set to the study's row for each suite (PVS_ROWS below, from the hosts' mock captures,
#          RECONCILE_HOST.md); everything else as the port passes it (the differences are listed in RECONCILE_HOST.md).
#   PORTDEC  PORT with TT_FABRIC_EXP_DECODE_BOTH=1 (slot_override_decode.patch: the route decode reads both bytes;
#          SLOT_ALLOCATOR.md section 12.13); its own kernel cache. PVS_ARMS picks the arms (default "study port"; the
#          decode A/B: "study port portdec"); pass a runs them in that order per suite, pass b reversed.
# Per run: every test must pass; the slot rows both hosts log for a suite must be the same multiset (a mismatch is
# reported with "!!! rows differ", the run kept). A failure or a hang stops the job.
#   pvs_runs.sh [SUITE...]     labels pvs<a|b>_<study|port>_<suite>: results/<label>.log, .csv, _tests.csv
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
D=fabric_exp/port_slots
source $D/port_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS_VC2 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D \
    TT_FABRIC_EXP_TIER_SLOTS TT_FABRIC_EXP_DYN_SLOTS TT_FABRIC_EXP_DYN_SLOTS_POLICY TT_FABRIC_EXP_TRID_ALLOC \
    TT_FABRIC_EXP_COMPLETION_GROUP TT_MESH_GRAPH_DESC_PATH
MGD42T=TT_MESH_GRAPH_DESC_PATH=/home/aagarwal/tt-metal/fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto
# suite | yaml | extra env (both arms) | the port's slot override (the study's row on these routers)
PVS_ROWS="ns|pvs_ns|-|TT_FABRIC_EXP_SLOTS_2D=@MESH@
t42ns|pvs_t42ns|$MGD42T|TT_FABRIC_EXP_SLOTS_2D=@TORUS42@
c2d|pvs_c2d|-|TT_FABRIC_EXP_SLOTS_2D=@MESH@
c1lin|pvs_c1lin|-|TT_FABRIC_EXP_SLOTS_1D=@LINE@
c1ring|pvs_c1ring|-|TT_FABRIC_EXP_SLOTS_1D=@RING@
c1ne|pvs_c1ne|-|@NE@"
PVS_ROWS=${PVS_ROWS//@MESH@/${PVS_MESH:-17,16,16}}
PVS_ROWS=${PVS_ROWS//@TORUS42@/${PVS_TORUS42:-17,16,16}}
PVS_ROWS=${PVS_ROWS//@LINE@/${PVS_LINE:-26,25,32}}
PVS_ROWS=${PVS_ROWS//@RING@/${PVS_RING:-26,25,32}}
PVS_ROWS=${PVS_ROWS//@NE@/${PVS_NE:--}}
want=" ${*:-ns t42ns c2d c1lin c1ring c1ne} "
mapfile -t SUITES < <(echo "$PVS_ROWS" | while IFS='|' read -r s y e r; do [[ $want == *" $s "* ]] && echo "$s|$y|$e|$r"; done)
[ ${#SUITES[@]} -gt 0 ] || { echo "no suites: $*"; exit 1; }
tests_of() { python3 -c "import json,sys; print(json.load(open('$D/yamls/manifest.json'))[sys.argv[1]]['tests'])" $1; }
for l in "${SUITES[@]}"; do IFS='|' read -r s y e r <<< "$l"; [ -f $D/yamls/$y.yaml ] || { echo "no $y"; exit 1; }; done
port_tree_setup
CP=/home/aagarwal/.tt-metal-cache-pvs-port; CD=/home/aagarwal/.tt-metal-cache-pvs-portdec; CS=/home/aagarwal/.tt-metal-cache-pvs-study
rm -rf "${CP:?}" "${CD:?}" "${CS:?}"; mkdir -p $CP $CD $CS
ARMS_A=${PVS_ARMS:-study port}
ARMS_B=$(echo $ARMS_A | tr ' ' '\n' | tac | tr '\n' ' ')
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
rows_of() { clean "$1" | grep -aoE 'VC[0-9] sender slots \{[^}]*\} receiver slots \{[^}]*\}' | sort | uniq -c | tr -s ' \n' ' '; }
one() {  # pass arm suite yaml extra-env port-row
    local P=$1 arm=$2 s=$3 y=$4 e=$5 r=$6 l=pvs$1_$2_$3 n env=() rc
    n=$(tests_of $y)
    [ "$e" = - ] || env+=($e)
    echo "##### $l ($n tests; job at ${SECONDS} s)"
    if [ $arm = study ]; then
        run_cap $l $D/yamls/$y.yaml $CS 900 TT_FABRIC_EXP_LOG_SLOTS=1 "${env[@]}"; rc=$?
        [ -f generated/fabric/bandwidth_results_blackhole.csv ] && cp generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv
    else
        [ "$r" = - ] || env+=($r)
        C=$CP
        [ $arm = portdec ] && { C=$CD; env+=(TT_FABRIC_EXP_DECODE_BOTH=1); }
        run_port $l $D/yamls/$y.yaml 900 "${env[@]}"; rc=$?
    fi
    [ $rc = 2 ] && { echo "!!! $l: hang (captured): stopping"; stop; }
    clean $out/$l.log | grep -aq "Tests: $n executed, 0 failed" || { echo "!!! $l failed (rc $rc, want '$n executed, 0 failed')"; stop; }
    echo "   rows: $(rows_of $out/$l.log)"
    if [ $arm != study ]; then  # the decode switch reached exactly the portdec arm's router builds (compiles: pass a)
        local nc nd
        nc=$(clean $out/$l.log | grep -a "compile cmd" | grep -ac "fabric_erisc_router")
        nd=$(clean $out/$l.log | grep -a "compile cmd" | grep -a "fabric_erisc_router" | grep -ac -- "-DTT_FABRIC_EXP_DECODE_BOTH")
        echo "   router compiles: $nc, with the decode define: $nd"
        [ $arm = port ] && [ $nd != 0 ] && { echo "!!! $l: the decode define reached the port arm"; stop; }
        [ $arm = portdec ] && [ $nc != 0 ] && [ $nd = 0 ] && { echo "!!! $l: no decode define in the portdec arm"; stop; }
    fi
    return 0
}
check_rows() {  # pass suite: every arm's logged rows against the first arm's
    local a b arm first=${ARMS_A%% *}
    a=$(rows_of $out/pvs$1_${first}_$2.log)
    for arm in $ARMS_A; do
        [ $arm = $first ] && continue
        b=$(rows_of $out/pvs$1_${arm}_$2.log)
        if [ "$a" = "$b" ]; then echo "   rows match ($2, $first / $arm)"; else echo "!!! rows differ ($2): $first $a | $arm $b"; fi
    done
}
mapfile -t SUITES_B < <(printf '%s\n' "${SUITES[@]}" | tac)
rm -f generated/fabric/bandwidth_results_blackhole.csv
python3 $CON/swap.py apply fabric_exp/variants/v4txa11 $BK || stop
for P in a b; do
    if [ $P = a ]; then order=("${SUITES[@]}"); arms=$ARMS_A; else order=("${SUITES_B[@]}"); arms=$ARMS_B; fi
    for l in "${order[@]}"; do
        IFS='|' read -r s y e r <<< "$l"
        for arm in $arms; do
            rm -f generated/fabric/bandwidth_results_blackhole.csv
            one $P $arm $s $y "$e" "$r"
        done
        check_rows $P $s
    done
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
echo "ALL_DONE pvs ${SUITES[*]%%|*} (job at ${SECONDS} s)"
