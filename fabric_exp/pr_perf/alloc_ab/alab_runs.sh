#!/bin/bash
# QB2 torus regression A/B on one P150 LoudBox (an lb2 job; never a09u32), coordinator 2026-11-05 (local, never
# committed): does the staged slot allocator (07322400fd6) slow a 2x2 torus at 14400 B against its parent (57c55f98118)?
#   A  57c55f98118 (tier table: one 2x2 torus router at 14,496 B slots has TX 5,4,4,4 RX 8; alloc_mock.py slots)
#   B  07322400fd6 (staged allocator: TX 6,5,5,5 RX 4)
# Each tree is staged whole by stage_tree.sh (no local diff: its own test_tt_fabric reads TT_FABRIC_NO_BW_TELEMETRY) at
# /data/aagarwal/lb2/trees/alab_<sha>.tar.zst, unpacked here at its build path (the binaries' RUNPATH), checked (sha256,
# MD5SUMS, COMMIT). One pass, A then B, each in a fresh kernel cache, deleted after its run. Every run: run_port's
# settings (fast dispatch, TT_METAL_CLEAR_L1=1, TT_FABRIC_NO_BW_TELEMETRY=1), one 2x2 of the cube (TT_VISIBLE_DEVICES =
# chip ids 0,1,2,3), the tree's own p300_x2 MGD, alab_rows.yaml (14400 B: torus XY 1-hop E/W bidir, torus all-four ring
# neighbors, FABRIC_2D mesh 1-hop E/W bidir as the control). Labels alab_<A|B>: results/<label>.log, .csv, _tests.csv.
# A hang (killed, captured) or a run with no result line stops the job.   DRY=1: print only. LOCAL=1: on a09u32 through
# its broker, from the clean worktrees themselves (no unpack).
cd /home/aagarwal/tt-metal || exit 1
CON=fabric_exp/consolidation
out=fabric_exp/results
D=fabric_exp/pr_perf/alloc_ab
DRY=${DRY:-0}
G=/data/aagarwal/lb2/trees
declare -A TREE=([A]=/home/aagarwal/tt-metal-alloc-57c5 [B]=/home/aagarwal/tt-metal-alloc-0732)
declare -A TAR=([A]=$G/alab_57c5.tar.zst [B]=$G/alab_0732.tar.zst)
declare -A WANT=([A]=57c55f98118 [B]=07322400fd6)
ARMS=(A B)
PFX=${ALAB_PREFIX:-alab}
# ALAB_ARMS="NAME:TREE:SHA ..." (LOCAL=1 only): other arms in that order, e.g. the bubble-receiver floor fix
# (coordinator 2026-11-05: nofix de8a2ac17a8, fix 6a4a961aeab); ALAB_TEL_PATCH=1 lets each worktree carry exactly
# fabric_exp/port_tune/local_patches/test_tt_fabric_no_bw_telemetry.patch (commits whose test_tt_fabric lacks the switch)
if [ -n "${ALAB_ARMS:-}" ]; then
    ARMS=()
    for a in $ALAB_ARMS; do IFS=: read -r n t w <<< "$a"; ARMS+=($n); TREE[$n]=$t; WANT[$n]=$w; done
fi
TELP=/home/aagarwal/tt-metal/fabric_exp/port_tune/local_patches/test_tt_fabric_no_bw_telemetry.patch
norm() { grep -vE '^(index |@@ )' | sed -n '/^diff --git/,/^-- $/p' | grep -v '^-- $'; }
Y=$D/alab_rows.yaml
N=3
MGD=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
[ -f $Y ] || { echo "!!! no $Y"; exit 1; }
[ "${LOCAL:-0}" = 1 ] || for r in A B; do [ -f ${TAR[$r]} ] && [ -f ${TAR[$r]}.sha256 ] || { echo "!!! no ${TAR[$r]}"; exit 1; }; done

unpack() {  # arm (pr_perf_runs.sh's)
    local r=$1 T=${TREE[$1]} f=${TAR[$1]}
    if [ -e $T ] && [ ! -e $T/.lb2_pr_owned ]; then echo "!!! $T exists and is not this harness's: not touched"; exit 1; fi
    (cd $(dirname $f) && sha256sum -c --quiet $(basename $f).sha256) || { echo "!!! $f fails its sha256"; exit 1; }
    rm -rf "${T:?}.new" && mkdir -p $T.new && tar -I zstd -xf $f -C $T.new || { echo "!!! unpack of $f failed"; exit 1; }
    (cd $T.new && md5sum -c --quiet MD5SUMS) || { echo "!!! $r: binaries differ from MD5SUMS"; exit 1; }
    case $(cat $T.new/COMMIT) in ${WANT[$r]}*) ;; *) echo "!!! $r tree at $(cat $T.new/COMMIT), want ${WANT[$r]}"; exit 1 ;; esac
    [ -s $T.new/PATCH ] && { echo "!!! $r tree has a local patch"; exit 1; }
    [ -f $T.new/$MGD ] || { echo "!!! $r tree has no $MGD"; exit 1; }
    ln -sfn build_Release $T.new/build
    echo "created by $(hostname) job ${SLURM_JOB_ID:-?} $(date -u +%FT%TZ) from $f" > $T.new/.lb2_pr_owned
    rm -rf "${T:?}" && mv $T.new $T || { echo "!!! could not put the tree at $T"; exit 1; }
    echo "$r: $(cat $T/COMMIT | cut -c1-11), no local patch, at $T ($(du -sh $T | cut -f1))"
}
if [ "$DRY" = 1 ]; then
    for r in "${ARMS[@]}"; do echo "   ${PFX}_$r: ${TREE[$r]} $Y ($N tests) TT_VISIBLE_DEVICES=0,1,2,3 TT_MESH_GRAPH_DESC_PATH=${TREE[$r]}/$MGD"; done
    exit 0
fi
source $CON/vc1_capture_lib.sh
source fabric_exp/port_slots/port_lib.sh
mkdir -p $out
if [ "${LOCAL:-0}" = 1 ]; then
    # a09u32's broker (coordinator 2026-11-05): the worktrees themselves, which must be clean at their commits
    [ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! LOCAL=1 is for a09u32's worktrees"; exit 1; }
    for r in "${ARMS[@]}"; do
        T=${TREE[$r]}
        case $(git -C $T rev-parse HEAD) in ${WANT[$r]}*) ;; *) echo "!!! $T is not at ${WANT[$r]}"; exit 1 ;; esac
        if [ "${ALAB_TEL_PATCH:-0}" = 1 ]; then
            diff <(git -C $T diff HEAD | norm) <(norm < $TELP) > /dev/null || { echo "!!! $T's local diff is not exactly $TELP"; exit 1; }
        else
            [ -z "$(git -C $T status --porcelain --untracked-files=no)" ] || { echo "!!! $T has local changes"; exit 1; }
        fi
        [ -f $T/$MGD ] && [ $T/build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric -nt $T/tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp ] ||
            { echo "!!! $T: no MGD or a stale test binary"; exit 1; }
        echo "$r: $(git -C $T rev-parse --short=11 HEAD), worktree at $T (local diff: $(git -C $T diff HEAD --stat | tail -1))"
    done
else
    [ "$(hostname)" != bh-lb-120-a09u32 ] || { echo "!!! a09u32 holds the build worktrees: run this on an lb2 node (or LOCAL=1)"; exit 1; }
    for r in A B; do unpack $r; done
fi
for r in "${ARMS[@]}"; do
    PT=${TREE[$r]} C=/home/aagarwal/.tt-metal-cache-$PFX-$r label=${PFX}_$r
    rm -rf "${C:?}"; mkdir -p $C
    run_port $label $Y 600 TT_VISIBLE_DEVICES=0,1,2,3 TT_MESH_GRAPH_DESC_PATH=$PT/$MGD TT_METAL_RUNTIME_ROOT=$PT; rc=$?
    [ $rc = 2 ] && { echo "!!! $label hung (captured): stopping"; exit 1; }
    roots=$(clean $out/$label.log | grep -a "compile cmd:" | grep -aoE " -I[^ ]*/tt_metal/hw/inc " | sort | uniq -c)
    nr=$(grep -c . <<< "$roots"); ok=$(grep -cF -- " -I$PT/tt_metal/hw/inc " <<< "$roots")
    if [ -n "$roots" ] && { [ "$nr" != 1 ] || [ "$ok" != 1 ]; }; then echo "!!! $label compiled from another root: $roots"; exit 1; fi
    got=$(clean $out/$label.log | grep -aoE 'Tests: [0-9]+ executed' | tail -1 | grep -oE '[0-9]+')
    [ -n "$got" ] || { echo "!!! $label: no result line: stopping"; exit 1; }
    [ "$got" = "$N" ] || echo "!!! $label: $got executed, want $N"
    [ $rc = 0 ] || echo "!!! $label failed (rc $rc)"
    clean $out/$label.log | grep -aE "Running Test Group|axis topologies|torus along" | sort -u | head -6 | sed 's/^/   | /'
    rm -rf "${C:?}"
done
echo "ALL_DONE (job at ${SECONDS} s)"
