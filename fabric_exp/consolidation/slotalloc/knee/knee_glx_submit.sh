#!/bin/bash
# Submit one Galaxy set of the slot-need sweep (G1 or G2; knee_arms.sh) as one dyn_glx.sbatch job, both passes in it
# (pass a in table order, pass b reversed), on TREE (default v4txa11_1104) after copying the knee yamls into it.
# Counts the study's Galaxy allocations first: at most 4 running + pending on bh_sc36_6 / bh_110a_sc16, the quad
# (glxq) excepted. Excludes the standing nodes plus a02u14 (rings) and a03u14 / a08u20 (multi-process tt-run).
#   knee_glx_submit.sh G1|G2 [--test-only]
set -euo pipefail
cd /home/aagarwal/tt-metal
SET=${1:?usage: knee_glx_submit.sh G1|G2 [--test-only]}; MODE=${2:-}
TREE=${TREE:-v4txa11_1104}
EXCLUDE=${EXCLUDE:-bh-glx-110-a01u14,bh-glx-110-a02u02,bh-glx-110-a02u08,bh-glx-110-a02u14,bh-glx-110-a03u14,bh-glx-110-a08u20}
PARTS=${GLX_PARTS:-bh_sc36_6,bh_110a_sc16}   # per-invocation override only (bh_sc36_2 needs GLX_SC36_2_EXCEPTION=1)
K=fabric_exp/consolidation/slotalloc/knee
source $K/knee_arms.sh
kind_of() { case $1 in g2w_4352) echo kn2w ;; g2f_4352) echo kn2f ;; gvc1_4352) echo vc1kn ;; gvc1x44_4352) echo vc1knx44 ;;
                       gvc1x44san_4352) echo vc1x44san ;; gvc1_3072) echo vc1kn3 ;; gvc1x44_3072) echo vc1knx443 ;;
                       gvc1x44san_3072) echo vc1x44san3 ;; merge_g8x4_bench) echo knm ;; merge_g8x4_val) echo knmval ;;
                       *) echo "!!! no Galaxy kind for $1" >&2; exit 1 ;; esac; }
mapfile -t ARMS < <(knee_arms | awk -v s=$SET '$1 == s')
[ ${#ARMS[@]} -gt 0 ] || { echo "no arms for $SET"; exit 1; }
step() {  # pass "set arm kind suite row [env]"
    local s a k su ro ex g=""; read -r s a k su ro ex <<< "$2"
    [[ ${ex:-} == *TT_FABRIC_EXP_COMPLETION_GROUP=* ]] && g=_g$(echo "$ex" | grep -oE 'TT_FABRIC_EXP_COMPLETION_GROUP=[124]' | cut -d= -f2)
    if [ "$ro" = - ]; then
        # the merge sweep's data-checked run: at the unsqueezed row, so its depths are checked too
        case $su in merge_g8x4_val) echo "$(kind_of $su):$1_$a:row=20,16,32" ;; *) echo "$(kind_of $su):$1_$a:def=19,9,16,5,4" ;; esac
    else echo "$(kind_of $su):$1_$a:row$g=$ro"; fi
}
STEPS=""
for l in "${ARMS[@]}"; do STEPS+="$(step a "$l") "; done
# pass b reversed; the data-checked sanity runs once, first (pass a only)
for l in $(printf '%s\n' "${ARMS[@]}" | tac | tr ' ' '|'); do l=${l//|/ }; [[ $l == *" - "* || $l == *" -" || $l == *san* ]] && continue; STEPS+="$(step b "$l") "; done
T=/data/aagarwal/frs-glx/trees/$TREE
mkdir -p $T/fabric_exp/consolidation/slotalloc/knee/yamls
for f in $K/yamls/*.yaml; do cp $f $T/$f.tmp && mv -f $T/$f.tmp $T/$f; done
n=$(squeue -u aagarwal -h -o "%P %j" | awk '$1 ~ /bh_sc36_6|bh_110a_sc16/ && $2 !~ /^glxq/' | wc -l)
CAP=${GLX_CAP:-4}   # per-invocation override only (e.g. the user's one-time 8 until 10:30 UTC on 2026-11-04)
echo "study Galaxy allocations now (running + pending, quad excepted): $n of $CAP"
echo "STEPS ($(echo $STEPS | wc -w)) = $STEPS"
echo "tree $TREE; partitions $PARTS; excluded $EXCLUDE"
[ "$MODE" = --test-only ] && { STEPS="$STEPS" TREE=$TREE sbatch --test-only --partition=$PARTS --exclude=$EXCLUDE --job-name=knee-$SET $K/../dyn_glx.sbatch; exit 0; }
[ $n -lt $CAP ] || { echo "!!! the cap of $CAP is reached: not submitting"; exit 1; }
# node-local kernel caches, removed per step (no /data inodes for caches; 2026-11-04)
LOCAL_CACHE=1 KEEP_CACHES=0 STEPS="$STEPS" TREE=$TREE sbatch --partition=$PARTS --exclude=$EXCLUDE --job-name=knee-$SET $K/../dyn_glx.sbatch
