#!/bin/bash
# Cycle breakdown, phase 2 on the P150 (fabric_exp/report/BREAKDOWN_PLAN.md): PC samples (pc_sampler.py over the debug
# bus: no router code, no measured perturbation) of both ERISCs of every router while one long row runs.
#   - points 1 and 3 (2D 1-hop write 4352 B: pcs3/ew4352.yaml; 2D multicast write 4352 B: bd_rows/mcast.yaml) on main's
#     router as shipped (census main_fw: kernel v1, flags 0, firmware keepalive) and on the new router (NEW_VARIANT
#     swapped in, kernel v2 slot, study flags, keepalive 1000), in two passes in reversed router order:
#     pass a = main then new, pass b = new then main;
#   - between the two, the costliest-rows profile on the new router only (2D multicast scatter, half-ring multicast
#     scatter, ring 1-hop scatter, unidirectional linear multicast write and scatter, Y-torus 1-hop N/S scatter; 4352 B).
#
#   bd_p150_runs.sh NEW_VARIANT TAG          a broker job (workspace /home/aagarwal; exclusive device)
#   DRY=1 bash bd_p150_runs.sh NEW TAG       print the steps only: no tree change, no device, nothing sourced
#
# One kernel cache per router (~/.tt-metal-cache-bd-TAG-{main,new}; fresh at the start; KEEP them: the -g ELFs in them
# are what bd_map.py maps the samples through). The first row of each topology is a short compile row that logs the
# compile commands (the config hashes restrict build matching). While each router is in the tree its sources are
# snapshotted (results/bd_TAG_src_{main,new}.tgz): the mapping reads those, never the live tree.
# Labels bd_TAG_<pass>_<router>_<row> (pass a, b; x for the costliest rows; c for compile rows), samples
# results/<label>_samples.{bin,json}, rates results/<label>.csv. Any failed row stops the job (the tree is restored).
# Then: fabric_exp/report/scripts/bd_phase2_sets.py p150 TAG NEW_VARIANT, and build_breakdown.py (BREAKDOWN_PLAN.md).
set -u
NEW=${1:?usage: bd_p150_runs.sh NEW_VARIANT TAG}; TAG=${2:?usage: bd_p150_runs.sh NEW_VARIANT TAG}
NEWV=fabric_exp/variants/$NEW
L=bd_${TAG}
CM=/home/aagarwal/.tt-metal-cache-bd-$TAG-main
CN=/home/aagarwal/.tt-metal-cache-bd-$TAG-new
P3=fabric_exp/pcs3
BR=fabric_exp/consolidation/bd_rows
V1="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
SRC="tt_metal/fabric/hw/inc tt_metal/fabric/impl/kernels/edm_fabric tt_metal/hw/inc tt_metal/hostdevcommon"

if [ "${DRY:-0}" = 1 ]; then
    cd /home/aagarwal/tt-metal
    out=fabric_exp/results; CON=fabric_exp/consolidation; BK=/home/aagarwal/.tt-metal-p1-swap-backup
    [ -d $NEWV ] || { echo "no variant $NEWV"; exit 1; }
    run_cap() { echo "   run_cap $1 $2 cache=${3##*/} timeout=$4 ${KA_ENV+KA_ENV=(empty: firmware keepalive)} ${*:5}"; }
    swap() { echo "   swap.py $*"; }
    snap() { echo "   snapshot $SRC -> $out/${L}_src_$1.tgz"; }
    sampler() { echo "   pc_sampler.py $out/$1_samples (beside $1)"; }
else
    source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
    source $CON/cap_lib.sh
    [ -d $NEWV ] || { echo "no variant $NEWV"; exit 1; }
    unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
    swap() { python3 $CON/swap.py "$@"; }
    snap() {  # router: the router sources as they are in the tree now (what the JIT compiled)
        tar czf $out/${L}_src_$1.tgz $SRC && echo "   sources: $out/${L}_src_$1.tgz ($(du -k $out/${L}_src_$1.tgz | cut -f1) KB;" \
            "swap manifest $( [ -f $BK/manifest.json ] && md5sum < $BK/manifest.json | cut -c1-12 || echo none))"
    }
    sampler() {  # label
        timeout 300 python_env/bin/python $CON/pc_sampler.py $out/$1_samples --start-log $out/$1.log \
            --start-delay-s 2 --max-s 90 2>&1 | grep --line-buffered -E '^sampler: |Error|error|Traceback'
    }
fi
stop() { echo "!!! $1"; [ "${DRY:-0}" = 1 ] || swap restore $BK; echo STOPPED; exit 1; }
rate() { awk -F, -v l=$1 'NR>1 {printf "   %s: %s %s B %.2f cycles/packet\n", l, $1, $7, 1.35e9/$10}' $out/$1.csv 2>/dev/null; }
compile() {  # router label yaml [env...]: a short row that compiles that topology's routers into the router's cache
    local c=$CN; [ $1 = main ] && c=$CM
    local lab=${L}_c_$1_$2 y=$3; shift 3
    if [ ${c##*-} = main ]; then KA_ENV= run_cap $lab $y $c 600 $V1 "$@"; else run_cap $lab $y $c 600 "$@"; fi
    [ $? = 0 ] || stop "$lab failed"
}
sample() {  # pass router row yaml [env...]: the row under the sampler
    local p=$1 r=$2 row=$3 y=$4 c=$CN rc; shift 4
    [ $r = main ] && c=$CM
    local lab=${L}_${p}_${r}_${row}
    [ "${DRY:-0}" = 1 ] || rm -f $out/$lab.log $out/${lab}_samples.bin $out/${lab}_samples.json
    # The run goes in a subshell (the sampler runs beside it). The subshell must not inherit jobs_lib.sh's EXIT trap,
    # which restores the tree: it would put the tree back after the new router's first sampled row (job 518).
    if [ $r = main ]; then ( trap - EXIT; KA_ENV= run_cap $lab $y $c 900 $V1 "$@" ) & else ( trap - EXIT; run_cap $lab $y $c 900 "$@" ) & fi
    local rp=$!
    sampler $lab
    wait $rp; rc=$?
    [ "${DRY:-0}" = 1 ] && return 0
    rate $lab
    [ $rc = 0 ] || stop "$lab failed (exit $rc)"
}

echo "== cycle breakdown, P150: main as shipped vs $NEW; tag $TAG ($(date +%T))"
[ "${DRY:-0}" = 1 ] || { rm -rf "${CM:?}" "${CN:?}"; mkdir -p $CM $CN; }

echo "== pass a: main, then $NEW"
compile main 2d $P3/compile2d.yaml
snap main
sample a main ew4352 $P3/ew4352.yaml
sample a main mcast $BR/mcast.yaml
swap apply $NEWV $BK || stop "swap apply failed"
compile new 2d $P3/compile2d.yaml
snap new
sample a new ew4352 $P3/ew4352.yaml
sample a new mcast $BR/mcast.yaml

echo "== pass b: $NEW, then main"
sample b new ew4352 $P3/ew4352.yaml
sample b new mcast $BR/mcast.yaml

echo "== costliest rows, $NEW only"
sample x new mcast_scatter $BR/mcast_scatter.yaml
compile new ring $P3/compilering.yaml
sample x new halfring_scatter $BR/halfring_scatter.yaml
sample x new ring1_scatter $BR/ring1_scatter.yaml
compile new 1d $P3/compile1d.yaml
sample x new unidirlin_write $BR/unidirlin_write.yaml
sample x new unidirlin_scatter $BR/unidirlin_scatter.yaml
compile new torusy $P3/compiletorusy.yaml
sample x new torusns_scatter $P3/torusns_scatter.yaml
swap restore $BK || { echo STOPPED; exit 1; }
snap main_after  # the restored tree; must equal the pass-a snapshot (checked below)

sample b main ew4352 $P3/ew4352.yaml
sample b main mcast $BR/mcast.yaml
if [ "${DRY:-0}" != 1 ]; then
    cmp -s <(tar xzf $out/${L}_src_main.tgz -O | md5sum) <(tar xzf $out/${L}_src_main_after.tgz -O | md5sum) \
        && echo "   main's sources equal in both passes" || echo "!!! main's sources differ between pass a and pass b"
    rm -f $M
fi
echo ALL_DONE
