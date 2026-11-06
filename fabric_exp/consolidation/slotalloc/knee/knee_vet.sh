#!/bin/bash
# Mock vet of every slot-need arm (knee_arms.sh): each arm's suite's first test on the mock cluster
# (slotstudy/sls_mock_vet.sh, nice 10) with the arm's explicit row; it must reach launch with the row's depths logged.
# P150 arms on the a09u32 descriptor, Galaxy arms on the Galaxy descriptor; the P150 VC1 split arms on the Galaxy 2x2
# split (the host computes the same depths from the explicit row; no P150 split mock mode). An arm's optional ENV field
# (knee_arms.sh) is passed to the run; with TT_FABRIC_EXP_COMPLETION_GROUP=G every logged receiver must have groups of G.
#   [TREE=/data/aagarwal/frs-glx/trees/v4txa11_1104] [V=<out dir>] [JOBS=4] knee_vet.sh [SET...]   (default: every set)
set -uo pipefail
cd /home/aagarwal/tt-metal
TREE=${TREE:-/data/aagarwal/frs-glx/trees/v4txa11_1104}
V=${V:-/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/kneevet}
K=fabric_exp/consolidation/slotalloc/knee
source $K/knee_arms.sh
mkdir -p $V/yamls
one() {  # set arm kind suite row [extra env]
    local set=$1 arm=$2 kind=$3 suite=$4 row=$5 ex=${6:--} mode=single desc=() sw="" want=() w f r f1 r1 out n exv=()
    [ "$ex" = - ] || IFS=';' read -ra exv <<< "$ex"
    python3 - $K/yamls/$suite.yaml $V/yamls/$suite.yaml <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); d["Tests"] = d["Tests"][:1]; yaml.safe_dump(d, open(sys.argv[2], "w"), sort_keys=False)
PY
    case $kind in
        2d|1d|val) desc=(MOCK_DESC=$PWD/fabric_exp/a09u32_cluster_desc.yaml) ;;
        gmval) ;;
        vc1|gvc1) mode=split ;;
        gvc1x44|gx44san) mode=split4x4 ;;
    esac
    [ "$row" = - ] && row_is_default=1
    case $kind in 2d|g2d) sw=TT_FABRIC_EXP_SLOTS_2D=$row ;; 1d) sw=TT_FABRIC_EXP_SLOTS_1D=$row ;; vc1|gvc1|gvc1x44|gx44san) sw=TT_FABRIC_EXP_SLOTS_VC1=$row ;; esac
    [ "$row" = - ] && sw=""
    if [ "$row" != - ]; then
        IFS=, read w f r f1 r1 <<< "$row"
        if [ $kind = 1d ]; then want=("VC0 sender slots {$w, $f} receiver slots {$r}"); else want=("VC0 sender slots {$w, $f, $f, $f} receiver slots {$r}"); fi
        [ -n "${f1:-}" ] && want+=("VC1 sender slots {$f1, $f1, $f1} receiver slots {$r1}")
    fi
    out=$V/$set-$arm
    env "${desc[@]}" SCR=$out TREE=$TREE P=1 bash fabric_exp/slotstudy/sls_mock_vet.sh $mode $V/yamls/$suite.yaml $sw "${exv[@]}" > $out.out 2>&1
    local st=ok
    grep -qE "launch reached by [1-9]" $out.out || st="NO LAUNCH"
    sed 's/\x1b\[[0-9;]*m//g' $out/*/*.log 2>/dev/null > $out.clean
    for n in "${want[@]}"; do grep -qaF "$n" $out.clean || st="MISSING '$n'"; done
    local g; g=$(grep -oE 'TT_FABRIC_EXP_COMPLETION_GROUP=[0-9]+' <<< "$ex" | cut -d= -f2)
    if [ -n "$g" ]; then  # every receiver channel the host logged has groups of g
        grep -aoE 'RX_CH_[0-9] G [0-9]+' $out.clean | grep -q . || st="NO GROUP LOG"
        grep -aoE 'RX_CH_[0-9] G [0-9]+' $out.clean | grep -vq " G $g\$" && st="GROUPS NOT $g"
    fi
    printf '%-3s %-12s %-8s %-18s %-16s %s\n' $set $arm $kind $suite $row "$st"
}
export -f one; export V K TREE
sets=" ${*:-A B C D E F G1 G2} "
knee_arms | while read -r s a k su ro ex; do [[ $sets == *" $s "* ]] && echo "$s $a $k $su $ro ${ex:--}"; done |
    xargs -P ${JOBS:-4} -L 1 bash -c 'one "$@"' _ | tee $V/summary.txt
bad=$(grep -vc " ok$" $V/summary.txt)
echo "$(wc -l < $V/summary.txt) arms, $bad not ok"
[ "$bad" = 0 ] && echo KNEE_VET_OK || { echo KNEE_VET_MISMATCH; exit 1; }
