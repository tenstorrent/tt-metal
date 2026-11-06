#!/bin/bash
# Mock vet of the transaction-ID allocation (TRID_ALLOCATION.md §5; CPU only, no device): v4txa11id from a /data tree
# whose host has TT_FABRIC_EXP_TRID_ALLOC (host_rebuild_tridalloc.sh, broker 649), each configuration's first test on
# the mock cluster (slotstudy/sls_mock_vet.sh, nice 10), with the switch on (rule "new") and off (rule "uniform").
# Each run must reach program launch; trid_vet_check.py then checks every router config's emitted RX_CH_* args against
# the rule for that config's own receivers, and dyn_size.py measures code size and stack per switch.
#   [TREE=/data/aagarwal/frs-glx/trees/v4txa11id_1104] [V=<out dir>] trid_vet.sh [NAME...]   (default: every config)
#   TRID_VET_CHECK_ONLY=1: no mock runs, re-check the logs and sizes of an earlier run in V
#   SWITCHES="on off" (default): which switch settings to run
#   CG=G: every run also sets TT_FABRIC_EXP_COMPLETION_GROUP=G (SLOT_ALLOCATOR.md section 12.10) and the check expects
#   groups of G on receivers without first-level acks
# Configurations: the study's default slots (no slot switch) on the P150 2D mesh, 1D line, ring, Y torus and XY torus,
# the Galaxy VC1 split (19,9,16,5,4) and VC2 on (17,14,16 + 2,4); the VC1 split at the dynamic slots with the VC0
# receiver floor at 3072 B (26,10,32,7,6: the TRID compile's 32 | 6 recapture).
set -uo pipefail
cd /home/aagarwal/tt-metal
TREE=${TREE:-/data/aagarwal/frs-glx/trees/v4txa11id_1104}
V=${V:-/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/tridvet}
P150=$PWD/fabric_exp/a09u32_cluster_desc.yaml
S=fabric_exp/slotstudy
Y=fabric_exp/consolidation/slotalloc/yamls
SA=fabric_exp/consolidation/slotalloc
grep -aq TT_FABRIC_EXP_TRID_ALLOC $TREE/build_Release/tt_metal/libtt_metal.so ||
    { echo "!!! $TREE's host library has no TT_FABRIC_EXP_TRID_ALLOC"; exit 1; }
[ -f $TREE/fabric_exp/slotstudy/glx_2x2_split_rank_bindings.yaml ] ||
    { echo "!!! $TREE has no fabric_exp/slotstudy (the split runs' rank bindings): cp -a fabric_exp/slotstudy into it"; exit 1; }
grep -q RX_CH_NUM_TRIDS $TREE/tt_metal/fabric/hw/inc/edm_fabric/router/*.hpp $TREE/tt_metal/fabric/impl/kernels/edm_fabric/*.hpp 2>/dev/null ||
    echo "note: RX_CH_NUM_TRIDS not found in $TREE's router headers (is it a v4txa11id tree?)"
# name | mode | yaml | system (glx|p150) | extra env
CONFIGS="p150_2d|single|$S/sls_2d_p150.yaml|p150|
p150_1d|single|$S/sls_1d_p150.yaml|p150|
p150_ring|single|fabric_exp/v4/val_ring_p150.yaml|p150|
p150_torusy|single|fabric_exp/v4/val_torus_y_p150.yaml|p150|
p150_txy|single|fabric_exp/torus/val_torus_xy_p150.yaml|p150|
glx_vc1|split|$S/sls_glx_vc1.yaml|glx|
glx_vc2|single|$S/sls_vc2_glx_4k.yaml|glx|TT_METAL_ENABLE_FABRIC_VC2=1
glx_vc13k_floor|split|$S/sls_glx_vc1_3k.yaml|glx|TT_FABRIC_EXP_DYN_SLOTS=1"
mkdir -p $V/yamls
bad=0
for sw in ${SWITCHES:-on off}; do
    rule=new; env=TT_FABRIC_EXP_TRID_ALLOC=1
    [ $sw = off ] && { rule=uniform; env=TT_FABRIC_EXP_TRID_OFF=1; }   # a no-op variable keeps the argument list shape
    cg=(); [ -n "${CG:-}" ] && cg=(TT_FABRIC_EXP_COMPLETION_GROUP=$CG)
    while IFS='|' read -r name mode yaml sys extra; do
        [ $# -gt 0 ] && [[ " $* " != *" $name "* ]] && continue
        python3 - "$yaml" "$V/yamls/$name.yaml" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
d["Tests"] = d["Tests"][:1]
yaml.safe_dump(d, open(sys.argv[2], "w"), sort_keys=False)
PY
        desc=(); [ $sys = p150 ] && desc=(MOCK_DESC=$P150)
        if [ -z "${TRID_VET_CHECK_ONLY:-}" ]; then
            env "${desc[@]}" SCR=$V/$sw/$name TREE=$TREE P=1 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 \
                bash $S/sls_mock_vet.sh $mode $V/yamls/$name.yaml $extra $env "${cg[@]}" > $V/$sw-$name.out 2>&1
        fi
        logs=$(ls $V/$sw/$name/*/*.log 2>/dev/null | grep -v mock_mapping)
        launched=$(grep -qE "launch reached by [1-9]" $V/$sw-$name.out && echo yes || echo no)
        echo "=== $sw $name (rule $rule): launch $launched"
        [ $launched = yes ] || bad=1
        python3 $SA/trid_vet_check.py --rule $rule --group ${CG:-0} $logs > $V/$sw-$name.check 2>&1 || bad=1
        sed 's/^/    /' $V/$sw-$name.check
    done <<< "$CONFIGS"
done
echo
python3 $SA/dyn_size.py "on=$V/on/*/*/cache_*" "off=$V/off/*/*/cache_*"
[ $bad = 0 ] && echo "TRID_VET_OK" || { echo "TRID_VET_MISMATCH"; exit 1; }
