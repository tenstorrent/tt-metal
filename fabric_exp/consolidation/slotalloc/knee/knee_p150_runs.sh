#!/bin/bash
# Slot-need sweep on one P150 (SLOT_ALLOCATOR.md section 12): the arms of the given sets (knee_arms.sh) in two passes,
# pass a in table order and pass b reversed, all on this host (an lb2 job, or a broker job on a09u32). v4txa11 swapped
# in once (-O3, study settings, BW telemetry off); every arm sets its explicit row (TT_FABRIC_EXP_SLOTS_2D / _1D /
# _VC1), and its logged depths, its test count ("Tests: N executed, 0 failed", knee/yamls/manifest.json) and, for the
# VC1 split, both ranks' finishes are checked; a mismatch, a failure or a hang (captured) stops the job.
#   knee_p150_runs.sh SET...        e.g. A | B | "C D F" | E
# KNEE_VARIANT: the router variant swapped in (default v4txa11; v4txa11id for the completion-group arms). An arm's
# optional ENV field (knee_arms.sh) adds its environment, e.g. TT_FABRIC_EXP_COMPLETION_GROUP or a mesh descriptor.
# Labels kn<a|b>_<set>_<arm>: results/<label>.log, _tests.csv (vc1: <label>_13129088_vc1.*); KNEE_PREFIX replaces "kn"
# (kq: the a09u32 broker runs, kept apart from the lb2 rows; compare rows within one host only). One fresh kernel cache
# per job (/home/aagarwal/.tt-metal-cache-knee-<sets>). No bash subshell runs while the router is swapped in (a
# "( ... ) &" subshell runs jobs_lib.sh's EXIT trap, which restores the tree: job 688).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
[ $# -gt 0 ] || { echo "usage: knee_p150_runs.sh SET..."; exit 1; }
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS_VC2 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D \
    TT_FABRIC_EXP_TIER_SLOTS TT_FABRIC_EXP_DYN_SLOTS TT_FABRIC_EXP_DYN_SLOTS_POLICY TT_FABRIC_EXP_TRID_ALLOC \
    TT_FABRIC_EXP_DRISC_DELAY TT_MESH_GRAPH_DESC_PATH TT_FABRIC_PHASE_PROFILE TT_FABRIC_PHASE_PROFILE_CSV \
    TT_FABRIC_EXP_COMPLETION_GROUP
K=$CON/slotalloc/knee
source $K/knee_arms.sh
sets=" $* "
mapfile -t ARMS < <(knee_arms | while read -r s a k su ro ex; do [[ $sets == *" $s "* ]] && echo "$s $a $k $su $ro ${ex:--}"; done)
[ ${#ARMS[@]} -gt 0 ] || { echo "no arms for sets: $*"; exit 1; }
for line in "${ARMS[@]}"; do
    read -r s a k su ro ex <<< "$line"
    case $k in 2d|1d|vc1|val) ;; *) echo "!!! $s $a: kind $k is not a P150 kind"; exit 1 ;; esac
    [ -f $K/yamls/$su.yaml ] || { echo "no $K/yamls/$su.yaml (mk_knee_yamls.py)"; exit 1; }
done
tests_of() { python3 -c "import json,sys; print(json.load(open('$K/yamls/manifest.json'))[sys.argv[1]]['tests'])" $1; }
C=/home/aagarwal/.tt-metal-cache-knee-$(echo $* | tr ' ' '_')
rm -rf "${C:?}" "${C:?}"-vc1-*; mkdir -p $C
clean() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
check_slots() {  # log kind row
    local w f r f1 r1 want=()
    IFS=, read w f r f1 r1 <<< "$3"
    if [ $2 = 1d ]; then want=("VC0 sender slots {$w, $f} receiver slots {$r}"); else want=("VC0 sender slots {$w, $f, $f, $f} receiver slots {$r}"); fi
    [ -n "${f1:-}" ] && want+=("VC1 sender slots {$f1, $f1, $f1} receiver slots {$r1}")
    local n; for n in "${want[@]}"; do
        clean $1 | grep -aqF "$n" || { echo "!!! slots not $3; logged:"; clean $1 | grep -aoE 'VC[01] sender slots \{[^}]*\} receiver slots \{[^}]*\}' | sort | uniq -c; return 1; }
    done
    echo "   slots: $3"
}
check_group() {  # log G: every receiver channel the host logged has groups of G (TT_FABRIC_EXP_COMPLETION_GROUP)
    local got; got=$(clean $1 | grep -aoE 'RX_CH_[0-9] G [0-9]+' | sort | uniq -c | tr -s ' \n' ' ')
    [ -n "$got" ] && ! clean $1 | grep -aoE 'RX_CH_[0-9] G [0-9]+' | grep -vq " G $2\$" ||
        { echo "!!! completion groups not $2; logged: ${got:-none}"; return 1; }
    echo "   groups: $got"
}
arm() {  # pass set arm kind suite row extra-env
    local P=$1 s=$2 a=$3 k=$4 su=$5 ro=$6 ex=${7:--} l=${KNEE_PREFIX:-kn}$1_$2_$3 n env=() rc
    n=$(tests_of $su)
    [ "$ro" = - ] || case $k in 2d) env=(TT_FABRIC_EXP_SLOTS_2D=$ro) ;; 1d) env=(TT_FABRIC_EXP_SLOTS_1D=$ro) ;; vc1) env=(TT_FABRIC_EXP_SLOTS_VC1=$ro) ;; esac
    [ "$ex" = - ] || IFS=';' read -ra exv <<< "$ex"
    [ "$ex" = - ] || env+=("${exv[@]}")
    echo "##### $l: ${env[*]:-default slots} $su ($n tests; job at ${SECONDS} s)"
    if [ $k = vc1 ]; then
        VC1CAP_EXTRA_ENV="TT_FABRIC_EXP_LOG_SLOTS=1 ${env[*]}" VC1CAP_SPLIT_YAML=$K/yamls/$su.yaml VC1CAP_SPLIT_FINISHES=$((2 * n)) \
            VC1CAP_SPLIT_TIMEOUT=600 run_vc1_cap $l $C-vc1-$P-$a O3 || stop
        check_slots $out/${l}_13129088_vc1.log 2d $ro || stop
        return 0
    fi
    rm -f generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv
    run_cap $l $K/yamls/$su.yaml $C 600 TT_FABRIC_EXP_LOG_SLOTS=1 "${env[@]}"; rc=$?
    [ -f generated/fabric/bandwidth_results_blackhole.csv ] && cp generated/fabric/bandwidth_results_blackhole.csv $out/${l}_tests.csv
    [ $rc = 2 ] && { echo "!!! $l: hang (captured): stopping"; stop; }
    clean $out/$l.log | grep -aq "Tests: $n executed, 0 failed" || { echo "!!! $l failed (rc $rc, want '$n executed, 0 failed')"; stop; }
    [ "$ro" = - ] || check_slots $out/$l.log $k $ro || stop
    [[ $ex == *TT_FABRIC_EXP_COMPLETION_GROUP=* ]] &&
        { check_group $out/$l.log "$(grep -oE 'TT_FABRIC_EXP_COMPLETION_GROUP=[0-9]+' <<< "$ex" | cut -d= -f2)" || stop; }
    return 0
}
mapfile -t ARMS_B < <(printf '%s\n' "${ARMS[@]}" | tac)   # before the swap: no subshell runs while it is in place
guard() {  # the swapped router is still in place (job 688)
    python3 - "$PWD/fabric_exp/variants/${KNEE_VARIANT:-v4txa11}" "$BK" <<'PY' || { echo "!!! the swap is no longer in place before $1: stopping"; stop; }
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
python3 $CON/swap.py apply fabric_exp/variants/${KNEE_VARIANT:-v4txa11} $BK || stop
for P in a b; do
    if [ $P = a ]; then order=("${ARMS[@]}"); else order=("${ARMS_B[@]}"); fi
    for line in "${order[@]}"; do
        guard "$line"
        read -r s a k su ro ex <<< "$line"
        arm $P $s $a $k $su $ro $ex
    done
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "ALL_DONE sets $* (job at ${SECONDS} s)"
