#!/bin/bash
# The bubble-flow-control injection rule (RING_JAM_FIX.md §10) on the P150x8 (a09u32 broker), base against branch:
#   base   /home/aagarwal/tt-metal-mainline-bfcbase  4db4f45790b, built like the branch (the regress /data tree
#          new_4db4f45 the first plan used was removed on 2026-11-04 08:00; /home/aagarwal/tt-metal-new has moved on)
#   bfc    /home/aagarwal/tt-metal-mainline-bfc  branch aagarwal/fabric-router-mainline-bfc (T = max(2, R - 16))
#
#   bfc_p150_runs.sh val    data-checked validation of the branch: mesh sanity (the tree's yaml), val1d_m1, val_ring_p150,
#                           val_torus_y_p150, val_torus_xy_p150 on the 2x4 and on the 4x2 (yamls/p150_x8_4x2_torus_xy_mgd)
#   bfc_p150_runs.sh perf   benchmark rows, pass a (base, bfc) then pass b (bfc, base): yamls/p150_it_ring.yaml,
#                           torusxy_2x4_sizes.yaml, torusxy_4x2_sizes.yaml (4x2 descriptor)
# No flow-control full-ring row on a LoudBox (FULLRING_MCAST_GAP.md §8.16). Every run: TT_FABRIC_NO_BW_TELEMETRY=1
# (bfc_lib.sh), the host's default opt level and slots. One kernel cache per tree and job kind. Labels bfcv_<row>
# (validation) and bfcp<PASS>_<tree>_<row> (perf). A hang stops the job (captured first); a failed validation stops it
# before any perf run.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/bfc/bfc_lib.sh
BASE=/home/aagarwal/tt-metal-mainline-bfcbase
BFC=/home/aagarwal/tt-metal-mainline-bfc
Y=$VAL/yamls
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
MGD="TT_MESH_GRAPH_DESC_PATH=$Y/p150_x8_4x2_torus_xy_mgd.textproto"
K=${1:-}
case $K in val | perf) ;; *) echo "usage: bfc_p150_runs.sh val|perf"; exit 1 ;; esac
[ -x $BFC/build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric ] || { echo "!!! no test_tt_fabric in $BFC"; exit 1; }
if [ $K = perf ]; then  # the base runs only in perf
    [ -x $BASE/build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric ] || { echo "!!! no test_tt_fabric in $BASE"; exit 1; }
    [ "$(git -C $BASE rev-parse HEAD)" = 4db4f45790b8383aaf8288dfd9efd84ffea2f67d ] && [ -z "$(git -C $BASE status --porcelain --untracked-files=no)" ] ||
        { echo "!!! $BASE is not a clean 4db4f45790b tree"; exit 1; }
fi
echo "##### $K: base $(git -C $BASE rev-parse --short HEAD 2>/dev/null), bfc $(git -C $BFC rev-parse --short HEAD) (dirty: $(git -C $BFC status --porcelain --untracked-files=no | wc -l))"
FAILED=""
name_of() { [ $1 = $BASE ] && echo base || echo bfc; }
one() {  # tree label yaml timeout [ENV...]
    local tree=$1 label=$2 y=$3 to=$4 c rc
    shift 4
    c=/home/aagarwal/.tt-metal-cache-bfc-$K-$(name_of $tree)
    if ! budget_ok "$to"; then echo "=== $label: skipped (job budget)"; FAILED="$FAILED $label(skipped)"; return 1; fi
    W=$tree
    cd $W || exit 1
    run_cap $label "$y" $c $to LD_LIBRARY_PATH=$(readlink -f $tree/build)/lib "$@"; rc=$?
    [ $rc = 2 ] && { echo "STOPPED (hang in $label)"; exit 1; }
    [ $rc = 0 ] || FAILED="$FAILED $label"
    return $rc
}
rm -rf /home/aagarwal/.tt-metal-cache-bfc-$K-base /home/aagarwal/.tt-metal-cache-bfc-$K-bfc
if [ $K = val ]; then
    one $BFC bfcv_mesh $SAN/test_fabric_mesh_sanity_common.yaml 300
    one $BFC bfcv_val1d $Y/val1d_m1.yaml 300
    one $BFC bfcv_ring $Y/val_ring_p150.yaml 120
    one $BFC bfcv_torusy $Y/val_torus_y_p150.yaml 120
    one $BFC bfcv_t24 $Y/val_torus_xy_p150.yaml 300
    one $BFC bfcv_t42 $Y/val_torus_xy_p150.yaml 300 $MGD
else
    for p in a b; do
        [ $p = a ] && order="$BASE $BFC" || order="$BFC $BASE"
        for t in $order; do
            n=$(name_of $t)
            one $t bfcp${p}_${n}_ring $Y/p150_it_ring.yaml 300 || { echo "STOPPED (bfcp${p}_${n}_ring failed)"; exit 1; }
            one $t bfcp${p}_${n}_t24 $Y/torusxy_2x4_sizes.yaml 300 || { echo "STOPPED (bfcp${p}_${n}_t24 failed)"; exit 1; }
            one $t bfcp${p}_${n}_t42 $Y/torusxy_4x2_sizes.yaml 300 $MGD || { echo "STOPPED (bfcp${p}_${n}_t42 failed)"; exit 1; }
        done
    done
fi
echo "failed:${FAILED:- none}"
echo "ALL_DONE (job at ${SECONDS} s)"
