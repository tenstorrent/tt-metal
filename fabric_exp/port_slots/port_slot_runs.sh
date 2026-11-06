#!/bin/bash
# The port slot measurements (SLOT_ALLOCATOR.md section 12.12) on one P150 LoudBox (an lb2 job): the arms of the given
# sets (port_arms.sh) in two passes, pass a in table order and pass b reversed (the data-checked set V in pass a only),
# all in this job. Router and host: the port, commit 22cdb1e55c3 plus fabric_exp/port_slots/slot_override.patch (the
# study's explicit-row switches and slot log, never committed), built in /home/aagarwal/tt-metal-port-slots and staged
# lean at PORT_TREE_SRC; this job copies it to the same path on the node (the binaries' RUNPATH) and checks the md5s.
# Each run: the port's own settings (as its run library: fast dispatch, TT_METAL_CLEAR_L1=1, profiler off, BW
# telemetry off, compile commands logged), no study kernel or flag switches; TT_FABRIC_EXP_LOG_SLOTS=1; the arm's row
# (TT_FABRIC_EXP_SLOTS_2D / _1D) and extra env. Checked per run: "Tests: N executed, 0 failed" (yamls/manifest.json)
# and the row's depths in the slot log; a failure, a mismatch or a hang (past 600 s: killed, captured with
# hang_capture.py into results/<label>_hangcap) stops the job.
#   port_slot_runs.sh SET...      labels ${PS_PREFIX:-ps}<a|b>_<set>_<arm>: results/<label>.log, .csv, _tests.csv
cd /home/aagarwal/tt-metal || exit 1
CON=fabric_exp/consolidation
out=fabric_exp/results
source $CON/vc1_capture_lib.sh
D=fabric_exp/port_slots
source $D/port_arms.sh
[ $# -gt 0 ] || { echo "usage: port_slot_runs.sh SET..."; exit 1; }
source $D/port_lib.sh
port_tree_setup
sets=" $* "
mapfile -t ARMS < <(port_arms | while read -r s a k su ro ex; do [[ $sets == *" $s "* ]] && echo "$s $a $k $su $ro ${ex:--}"; done)
[ ${#ARMS[@]} -gt 0 ] || { echo "no arms for sets: $*"; exit 1; }
tests_of() { python3 -c "import json,sys; print(json.load(open('$D/yamls/manifest.json'))[sys.argv[1]]['tests'])" $1; }
for line in "${ARMS[@]}"; do read -r s a k su ro ex <<< "$line"; [ -f $D/yamls/$su.yaml ] || { echo "no $su"; exit 1; }; done
C=/home/aagarwal/.tt-metal-cache-portslots-$(echo $* | tr ' ' '_')
rm -rf "${C:?}"; mkdir -p $C
clean() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }

arm() {  # pass set arm kind suite row extra-env
    local P=$1 s=$2 a=$3 k=$4 su=$5 ro=$6 ex=${7:--} l=${PS_PREFIX:-ps}$1_$2_$3 n env=() exv=() want="" w f r rc
    n=$(tests_of $su)
    if [ "$ro" != - ]; then
        IFS=, read w f r <<< "$ro"
        case $k in 2d) env=(TT_FABRIC_EXP_SLOTS_2D=$ro); want="VC0 sender slots {$w, $f, $f, $f} receiver slots {$r}" ;;
                   1d) env=(TT_FABRIC_EXP_SLOTS_1D=$ro); want="VC0 sender slots {$w, $f} receiver slots {$r}" ;; esac
    fi
    [ "$ex" = - ] || { IFS=';' read -ra exv <<< "$ex"; env+=("${exv[@]}"); }
    echo "##### $l: ${env[*]:-default slots} $su ($n tests; job at ${SECONDS} s)"
    run_port $l $D/yamls/$su.yaml 600 "${env[@]}"; rc=$?
    [ $rc = 2 ] && { echo "!!! $l: hang (captured): stopping"; exit 1; }
    clean $out/$l.log | grep -aq "Tests: $n executed, 0 failed" || { echo "!!! $l failed (rc $rc, want '$n executed, 0 failed')"; exit 1; }
    if [ -n "$want" ]; then
        clean $out/$l.log | grep -aqF "$want" || { echo "!!! $l: slots not $ro; logged:"; clean $out/$l.log |
            grep -aoE 'VC0 sender slots \{[^}]*\} receiver slots \{[^}]*\}' | sort | uniq -c; exit 1; }
        echo "   slots: $ro"
    fi
}
mapfile -t ARMS_B < <(printf '%s\n' "${ARMS[@]}" | tac | grep -v '^V ')
for P in a b; do
    if [ $P = a ]; then order=("${ARMS[@]}"); else order=("${ARMS_B[@]}"); fi
    for line in "${order[@]}"; do
        read -r s a k su ro ex <<< "$line"
        arm $P $s $a $k $su $ro $ex
    done
done
echo "ALL_DONE port slot sets $* (job at ${SECONDS} s)"
