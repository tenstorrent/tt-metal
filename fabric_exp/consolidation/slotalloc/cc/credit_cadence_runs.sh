#!/bin/bash
# Credit-cadence A/B (TRID_ALLOCATION.md §6.4, user go 2026-11-04): why 16 transaction IDs lose ~2 cycles/packet on
# the torus N/S rows. Broker job on a09u32 (host library with TT_FABRIC_EXP_TRID_ALLOC, broker 649), no router build:
# variants/v4txa11id swapped in once; arm idoff = the switch off (8 IDs on the N/S routers' 16a receivers), arm id =
# TT_FABRIC_EXP_TRID_ALLOC=1 (16 IDs). -O3, study settings, BW telemetry off, hang capture armed (a hang stops the job).
#   credit_cadence_runs.sh [safety|ab|all]      (default all)
# Rows (slotalloc/cc/yamls, mk from torus/torusxy_4x2_sizes.yaml; 4x2 XY torus, study descriptor): 1-hop N/S write,
# its bidirectional pair, the wrap 1-hop N/S, at 4352 and 3072 B, 120M packets per sender (~13 s of traffic each).
# Beside each benchmark run txq_snap.py reads every connected Ethernet core's TXQ0 / TXQ1 TRANSFER_CNT and RESEND_CNT
# once a second (read-only, tt-exalens safe mode, after the test logs "Running Test:"); cc_report.py takes the
# steady window per data link and gives credit transfers (the receiver's TXQ1) per data packet.
#   safety  the same row and arm (idoff, ns 4352) without and with the snapshot loop: plain, snap, snap, plain
#   ab      pass a: idoff then id, rows in order; pass b: id then idoff, rows reversed
# Labels <TAG><s|a|b>_<arm>_<row>_<size>[_p1|_s1|...] (results/<label>.log, _tests.csv, _txq.json), TAG default cc.
# Each run's logged transaction IDs must be its arm's (idoff: uniform-rule lines only; id: assigned only).
# Runs are not started past CC_BUDGET s (default 1300) of job time. Only this script restores the tree: once at the end
# (or in stop()), and through jobs_lib.sh's EXIT trap; no bash subshell runs while the swap is in place (a "( ... ) &"
# subshell runs that EXIT trap when it exits: job 688), and guard() checks the swap before every run.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
MODE=${1:-all}
case $MODE in safety|ab|all) ;; *) echo "usage: credit_cadence_runs.sh [safety|ab|all]"; exit 1 ;; esac
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS_VC2 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D \
    TT_FABRIC_EXP_TIER_SLOTS TT_FABRIC_EXP_DYN_SLOTS TT_FABRIC_EXP_DYN_SLOTS_POLICY TT_FABRIC_EXP_TRID_ALLOC \
    TT_FABRIC_EXP_DRISC_DELAY TT_MESH_GRAPH_DESC_PATH TT_FABRIC_PHASE_PROFILE TT_FABRIC_PHASE_PROFILE_CSV
grep -aq TT_FABRIC_EXP_TRID_ALLOC build_Release/tt_metal/libtt_metal.so || { echo "!!! no TT_FABRIC_EXP_TRID_ALLOC in the host library"; exit 1; }
TAG=${TAG:-cc}
CC=$CON/slotalloc/cc
MGD=TT_MESH_GRAPH_DESC_PATH=$PWD/fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto
ROWS="ns_4352 nsbidir_4352 wrapns_4352 ns_3072 nsbidir_3072 wrapns_3072"
for r in $ROWS; do [ -f $CC/yamls/cc_$r.yaml ] || { echo "no $CC/yamls/cc_$r.yaml"; exit 1; }; done
python_env/bin/python -c "import ttexalens" || { echo "!!! python_env has no tt-exalens"; exit 1; }
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
# The swapped router must still be in place before every run: the backup's manifest exists and every variant file is
# the tree's (job 688: a mid-job restore made the next build fail after device open, which leaves the ERISCs held).
guard() {
    python3 - "$PWD/fabric_exp/variants/v4txa11id" "$BK" <<'PY' || { echo "!!! the swap is no longer in place before $1: stopping"; stop; }
import filecmp, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "fabric_exp/consolidation"))
from offline_build import ROOT, resolve
variant, bk = sys.argv[1], sys.argv[2]
if not os.path.exists(os.path.join(bk, "manifest.json")):
    sys.exit(f"no {bk}/manifest.json")
bad = [dst for src, dst in resolve(variant) if not filecmp.cmp(src, os.path.join(ROOT, dst), shallow=False)]
if bad:
    sys.exit(f"{len(bad)} variant files differ from the tree, first {bad[0]}")
PY
}
clean() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
ids() {  # log arm
    local na nu
    na=$(clean $1 | grep -ac "receiver transaction IDs (TT_FABRIC_EXP_TRID_ALLOC)")
    nu=$(clean $1 | grep -ac "receiver transaction IDs (uniform rule)")
    echo "   transaction IDs: $na assigned, $nu uniform-rule lines; $(clean $1 |
        grep -aoE 'RX_CH_[0-9] G [0-9]+ ids [0-9]+ start [0-9]+' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    case $2 in id) [ $na -gt 0 ] && [ $nu = 0 ] ;; *) [ $nu -gt 0 ] && [ $na = 0 ] ;; esac ||
        { echo "!!! the logged IDs are not arm $2's"; return 1; }
}
one() {  # label arm row snap(0|1)
    local l=$1 a=$2 r=$3 s=$4 c=/home/aagarwal/.tt-metal-cache-$TAG-$2 rc env=($MGD TT_FABRIC_EXP_LOG_SLOTS=1)
    [ $SECONDS -le ${CC_BUDGET:-1300} ] || { echo "=== budget: not starting $l or later at ${SECONDS} s"; stop; }
    guard $l
    [ $a = id ] && env+=(TT_FABRIC_EXP_TRID_ALLOC=1)
    rm -f generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv $out/${l}_txq.json $out/$l.log
    if [ $s = 1 ]; then
        # The test runs in this shell (a background subshell would run jobs_lib.sh's EXIT trap, which restores the
        # swapped router: job 688); the snapshot tool runs beside it and stops at the test's "Tests: " line.
        timeout 400 python_env/bin/python $CC/txq_snap.py $out/${l}_txq.json --start-log $out/$l.log --max-s 120 \
            > $out/${l}_txq.log 2>&1 &
        local sp=$! i
        run_cap $l $CC/yamls/cc_$r.yaml $c 600 "${env[@]}"; rc=$?
        for i in $(seq 30); do kill -0 $sp 2>/dev/null || break; sleep 1; done
        kill $sp 2>/dev/null   # only if the test ended before "Running Test:" (the tool is still waiting, device closed)
        wait $sp 2>/dev/null
        grep -E '^txq_snap: [0-9]+ snapshots|never|Error|error|Traceback' $out/${l}_txq.log
    else
        run_cap $l $CC/yamls/cc_$r.yaml $c 600 "${env[@]}"; rc=$?
    fi
    [ -f generated/fabric/bandwidth_results_blackhole.csv ] && cp generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv
    [ $rc = 2 ] && { echo "!!! $l: hang (captured): stopping"; stop; }
    clean $out/$l.log | grep -aq "Tests: 1 executed, 0 failed" || { echo "!!! $l failed (rc $rc)"; stop; }
    awk -F, 'NR>1 {printf "   %s %s %s B %.2f cycles/packet\n", $1, $6, $11, 1.35e9/$14}' $out/${l}_tests.csv 2>/dev/null
    ids $out/$l.log $a || stop
}
python3 $CON/swap.py apply fabric_exp/variants/v4txa11id $BK || stop
for a in idoff id; do rm -rf "/home/aagarwal/.tt-metal-cache-$TAG-$a"; mkdir -p /home/aagarwal/.tt-metal-cache-$TAG-$a; done
if [ $MODE != ab ]; then
    echo "##### safety: idoff ns_4352 without / with the snapshot loop (job at ${SECONDS} s)"
    one ${TAG}s_idoff_ns_4352_p1 idoff ns_4352 0
    one ${TAG}s_idoff_ns_4352_s1 idoff ns_4352 1
    one ${TAG}s_idoff_ns_4352_s2 idoff ns_4352 1
    one ${TAG}s_idoff_ns_4352_p2 idoff ns_4352 0
fi
if [ $MODE != safety ]; then
    for P in a b; do
        if [ $P = a ]; then arms="idoff id"; rows=$ROWS; else arms="id idoff"; rows=$(echo $ROWS | tr ' ' '\n' | tac | tr '\n' ' '); fi
        for a in $arms; do
            echo "##### pass $P, arm $a (job at ${SECONDS} s)"
            for r in $rows; do one ${TAG}${P}_${a}_$r $a $r 1; done
        done
    done
fi
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "ALL_DONE $MODE (job at ${SECONDS} s)"
