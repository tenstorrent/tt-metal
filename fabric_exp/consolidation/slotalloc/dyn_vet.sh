#!/bin/bash
# Mock vet and code-size check of the dynamic slot allocator (SLOT_ALLOCATOR.md §8 step 3): CPU only, no device. For
# each configuration and arm (tier, dyn, dynw24), the first test of the configuration's yaml runs on the mock cluster
# (slotstudy/sls_mock_vet.sh, nice 10) from a /data tree with the dyn host build and v4txa11; it must reach program
# launch with the expected depths logged (TT_FABRIC_EXP_LOG_SLOTS). Then dyn_size.py measures every router ELF the
# runs built, per arm: largest E0 / E1 code against the 26,624 B kernel config buffer, least free stack.
#   [TREE=/data/aagarwal/frs-glx/trees/v4txa11_dyn] [V=<out dir>] dyn_vet.sh [NAME...]   (default: every configuration)
#   DYN_VET_CHECK_ONLY=1: no mock runs, re-check the logs and sizes of an earlier run in V
# Expected rows: slot_alloc_table.sh (default policy; worker_cap_slots=24 for dynw24), 2026-11-03. On a host library
# with the VC0 receiver floor (2026-11-04) dyn and dynw24 run with vc0_receiver_floor=off, and the dynfloor arm checks
# the floor's rows (SLOT_ALLOCATOR §10; "-" where it equals dyn). Without the floor the dynfloor arm is skipped.
set -uo pipefail
cd /home/aagarwal/tt-metal
TREE=${TREE:-/data/aagarwal/frs-glx/trees/v4txa11_dyn}
V=${V:-/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/dynvet}
P150=$PWD/fabric_exp/a09u32_cluster_desc.yaml
S=fabric_exp/slotstudy
Y=fabric_exp/consolidation/slotalloc/yamls
# DYN_VET_NO_LIBCHECK=1: run anyway (tests this script on a tree without the dyn host: its dyn arms must mismatch)
grep -aq TT_FABRIC_EXP_DYN_SLOTS_POLICY $TREE/build_Release/tt_metal/libtt_metal.so || [ -n "${DYN_VET_NO_LIBCHECK:-}" ] ||
    { echo "!!! $TREE's host library has no TT_FABRIC_EXP_DYN_SLOTS"; exit 1; }
FLOOR_OFF=""; grep -aq vc0_receiver_floor $TREE/build_Release/tt_metal/libtt_metal.so && FLOOR_OFF=vc0_receiver_floor=off
# name | mode | yaml | system (glx|p150) | extra env | expected tier ; dyn ; dynw24 ; dynfloor (W,F,R[,F1,R1][+W2,R2];
# "-" skips the arm)
CONFIGS="glx2d3k|single|$S/sls_2d_glx_3k.yaml|glx||35,16,32;27,24,16;24,25,16;23,20,32
glx1d|single|$S/sls_1d_glx.yaml|glx||16,16,32;26,25,32;24,27,32;-
glxvc1|split|$S/sls_glx_vc1.yaml|glx||51,4,8,2,4;19,9,16,5,4;-;-
glxvc13k|split|$S/sls_glx_vc1_3k.yaml|glx||85,4,8,2,4;30,14,16,7,6;24,16,16,7,6;26,10,32,7,6
glxvc2|single|$S/sls_vc2_glx_4k.yaml|glx|TT_METAL_ENABLE_FABRIC_VC2=1|43,7,11+2,4;17,14,16+4,2;-;-
glxvc23k|single|$S/sls_vc2_glx_3k.yaml|glx|TT_METAL_ENABLE_FABRIC_VC2=1|77,7,11+2,4;24,22,16+5,4;-;20,18,32+5,4
glxvc12|split|$S/sls_glx_vc1.yaml|glx|TT_METAL_ENABLE_FABRIC_VC2=1|45,4,8,2,4+2,4;17,8,16,5,4+3,2;-;-
p150_2d|single|$S/sls_2d_p150.yaml|p150||17,16,16;17,16,16;-;-
p150_2d3k|single|$Y/sls_2d_p150_3072.yaml|p150||35,16,32;27,24,16;24,25,16;23,20,32
p150_ring3k|single|$Y/exp_ring_3072.yaml|p150||32,32,32;43,42,32;-;-
p150_txy3k|single|$Y/torusxy_2x4_sizes_3072.yaml|p150||35,16,32;27,24,16;-;23,20,32"
want_lines() {  # row kind(1d|2d): the log lines the row must produce
    local e=$1 kind=$2 w f r f1 r1 w2 r2 v0
    IFS=, read w f r f1 r1 <<< "${e%+*}"
    [ $kind = 1d ] && v0="VC0 sender slots {$w, $f} receiver slots {$r}" || v0="VC0 sender slots {$w, $f, $f, $f} receiver slots {$r}"
    echo "$v0"
    [ -n "${f1:-}" ] && echo "VC1 sender slots {$f1, $f1, $f1} receiver slots {$r1}"
    if [[ $e == *+* ]]; then IFS=, read w2 r2 <<< "${e#*+}"; echo "VC2 sender slots {$w2} receiver slots {$r2}"; fi
}
mkdir -p $V/yamls
bad=0
while IFS='|' read -r name mode yaml sys extra rows; do
    [ $# -gt 0 ] && [[ " $* " != *" $name "* ]] && continue
    python3 - "$yaml" "$V/yamls/$name.yaml" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
d["Tests"] = d["Tests"][:1]
yaml.safe_dump(d, open(sys.argv[2], "w"), sort_keys=False)
PY
    kind=2d; [[ $name == *1d* || $name == *ring* ]] && kind=1d
    IFS=';' read -r e_tier e_dyn e_w24 e_floor <<< "$rows"
    for arm in tier dyn dynw24 dynfloor; do
        case $arm in
            tier) e=$e_tier; env=TT_FABRIC_EXP_TIER_SLOTS=1 ;;
            dyn) e=$e_dyn; env="TT_FABRIC_EXP_DYN_SLOTS=1${FLOOR_OFF:+ TT_FABRIC_EXP_DYN_SLOTS_POLICY=$FLOOR_OFF}" ;;
            dynw24) e=$e_w24; env="TT_FABRIC_EXP_DYN_SLOTS=1 TT_FABRIC_EXP_DYN_SLOTS_POLICY=worker_cap_slots=24${FLOOR_OFF:+,$FLOOR_OFF}" ;;
            dynfloor) e=$e_floor; env=TT_FABRIC_EXP_DYN_SLOTS=1; [ -n "$FLOOR_OFF" ] || e=- ;;
        esac
        [ "$e" = - ] && continue
        desc=(); [ $sys = p150 ] && desc=(MOCK_DESC=$P150)
        if [ -z "${DYN_VET_CHECK_ONLY:-}" ]; then
            env "${desc[@]}" SCR=$V/$arm/$name TREE=$TREE P=1 bash $S/sls_mock_vet.sh $mode $V/yamls/$name.yaml $extra $env \
                > $V/$arm-$name.out 2>&1
        fi
        logs=$(ls $V/$arm/$name/*/*.log 2>/dev/null | grep -v mock_mapping)
        launched=$(grep -qE "launch reached by [1-9]" $V/$arm-$name.out && echo yes || echo no)
        # The cleaned logs go to a file first: grep -q on a pipe under pipefail fails when it exits before sed is done.
        sed 's/\x1b\[[0-9;]*m//g' $logs > $V/$arm-$name.clean
        status=ok
        while read -r line; do
            grep -qaF "$line" $V/$arm-$name.clean || status="MISSING '$line'"
        done < <(want_lines "$e" $kind)
        [ $launched = yes ] || status="no launch ($status)"
        [ "$status" = ok ] || bad=1
        printf '%-12s %-7s %-22s %s\n' $name $arm "$e" "$status"
    done
done <<< "$CONFIGS"
echo
python3 fabric_exp/consolidation/slotalloc/dyn_size.py "tier=$V/tier/*/*/cache_*" "dyn=$V/dyn/*/*/cache_*" \
    "dynw24=$V/dynw24/*/*/cache_*" ${FLOOR_OFF:+"dynfloor=$V/dynfloor/*/*/cache_*"}
[ $bad = 0 ] && echo "DYN_VET_OK" || { echo "DYN_VET_MISMATCH"; exit 1; }
