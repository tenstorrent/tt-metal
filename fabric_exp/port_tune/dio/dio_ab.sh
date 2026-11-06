#!/bin/bash
# deliver_other: out of line (noinline, as built) against inlined (FORCE_INLINE), on the port 6dad5fec638.
#   NOINLINE  router_local_delivery.hpp as built.
#   INLINE    the same with deliver_other's `__attribute__((noinline))` replaced by FORCE_INLINE; nothing else.
# One host binary (/home/aagarwal/tt-metal-port-gap/build_Release, 6dad5fec638). The routers and test kernels
# JIT-compile from each run's runtime root (/home/aagarwal/tt-metal-port-gap-roots/dio_<arm>_<sender>: symlinks into
# the worktree, its own generated/; the inline roots carry their own copy of that one header). Both arms run under two
# test senders, since main's sender sets the 1-hop write rows' rate (RECONCILE_DEVICE.md section 8.1):
#   main  main's test sender as built (send_packets_stateful; roots/asbuilt/tests);
#   loop  the study's stream loop in that sender (roots/loop/tests; the quick test of job 798).
# Rows: dio_ew.yaml (2x4 mesh 1-hop E/W: writes 16 / 4352 B, atomic, scatter, fused; 50,000 packets each).
# Pass a: per sender (main, loop) noinline then inline; pass b: senders and arms reversed. Fresh JIT cache per root.
# Broker job on bh-lb-120-a09u32 (workspace /home/aagarwal, timeout 1500 s): bash fabric_exp/port_tune/dio/dio_ab.sh
# DRY=1: the checks, then the runs it would do.
set -uo pipefail
cd /home/aagarwal/tt-metal
CON=fabric_exp/consolidation
D=fabric_exp/port_tune/dio
source $CON/vc1_capture_lib.sh
source fabric_exp/port_slots/port_lib.sh
R=/home/aagarwal/tt-metal-port-gap-roots
TREE=/home/aagarwal/tt-metal-port-gap
YAML=$D/dio_ew.yaml
H=tt_metal/fabric/hw/inc/edm_fabric/router/router_local_delivery.hpp
K=tests/tt_metal/tt_fabric/test_infra/kernels/tt_fabric_test_sender.cpp
RUN_TIMEOUT=240
out=/home/aagarwal/tt-metal/fabric_exp/results/dio_$(date +%Y%m%d-%H%M%S)
(while sleep 30; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
echo "dio_ab: $(hostname), results $out"

# --- checks before any device run
[ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! expected bh-lb-120-a09u32"; exit 1; }
[ "$(git -C $TREE rev-parse --short=11 HEAD)" = 6dad5fec638 ] || { echo "!!! $TREE is not at 6dad5fec638"; exit 1; }
[ -z "$(git -C $TREE status --porcelain -- tt_metal)" ] || { echo "!!! $TREE has tt_metal changes"; exit 1; }
want_h=$(git -C $TREE show HEAD:$H | md5sum | cut -d' ' -f1)
want_inline=$(git -C $TREE show HEAD:$H | sed 's/^__attribute__((noinline)) void deliver_other(/FORCE_INLINE void deliver_other(/' | md5sum | cut -d' ' -f1)
want_main=$(git -C $TREE show HEAD:$K | md5sum | cut -d' ' -f1)
want_loop=5d091b123a0051587deb27824edc0265
for arm in noinline inline; do
    for s in main loop; do
        r=$R/dio_${arm}_$s
        h=$(md5sum < $r/$H | cut -d' ' -f1); k=$(md5sum < $r/$K | cut -d' ' -f1)
        wh=$want_h; [ $arm = inline ] && wh=$want_inline
        wk=$want_main; [ $s = loop ] && wk=$want_loop
        [ "$h" = "$wh" ] && [ "$k" = "$wk" ] || { echo "!!! $r: header $h (want $wh), sender $k (want $wk)"; exit 1; }
        [ "$(readlink $r/build_Release)" = "$TREE/build_Release" ] || { echo "!!! $r/build_Release"; exit 1; }
    done
done
[ "$want_h" != "$want_inline" ] || { echo "!!! the inline edit did not apply"; exit 1; }
echo "roots: header md5 noinline $want_h / inline $want_inline; sender md5 main $want_main / loop $want_loop"

declare -A CACHE
for arm in noinline inline; do for s in main loop; do CACHE[${arm}_$s]=/home/aagarwal/.tt-metal-cache-dio-${arm}-$s; done; done
if [ -n "${DRY:-}" ]; then
    run_port() { echo "DRY: run_port $* (TT_METAL_HOME=$PT, cache $C)"; }
else
    mkdir -p $out
    for c in "${CACHE[@]}"; do rm -rf "${c:?}"; done
fi

# --- the runs
rc=0
for pass in a b; do
    order="main:noinline main:inline loop:noinline loop:inline"
    [ $pass = b ] && order="loop:inline loop:noinline main:inline main:noinline"
    for x in $order; do
        s=${x%%:*}; arm=${x#*:}
        need=$((RUN_TIMEOUT + 45 + 60 + VC1CAP_CAPTURE_TIMEOUT + 120 + 20))
        if [ $((SECONDS + need)) -gt 1450 ]; then
            echo "=== stopping before dio${pass}_${s}_$arm: job at ${SECONDS} s, a hang could need ${need} s more"; rc=1; break 2
        fi
        PT=$R/dio_${arm}_$s; C=${CACHE[${arm}_$s]}
        run_port dio${pass}_${s}_$arm $YAML $RUN_TIMEOUT TT_METAL_RUNTIME_ROOT=$PT; r=$?
        if [ $r != 0 ]; then echo "!!! dio${pass}_${s}_$arm failed (rc $r): stopping"; rc=1; break 2; fi
    done
done
[ -n "${DRY:-}" ] && { echo "DRY done (${SECONDS} s)"; exit 0; }

# --- which images the edit changed, from this job's caches
for s in main loop; do
    echo "--- sender $s: noinline vs inline"
    python3 fabric_exp/port_tune/elf_load_cmp.py ${CACHE[noinline_$s]} ${CACHE[inline_$s]} fabric_erisc_router \
        tt_fabric_test_sender tt_fabric_test_receiver
done
for arm in noinline inline; do
    echo "--- $arm: main vs loop sender (routers must be identical)"
    python3 fabric_exp/port_tune/elf_load_cmp.py ${CACHE[${arm}_main]} ${CACHE[${arm}_loop]} fabric_erisc_router \
        || { echo "!!! $arm routers differ between the sender roots"; rc=1; }
done
python3 $D/dio_report.py $out | tee $out/report.md
echo "dio_ab done (rc $rc, ${SECONDS} s)"
exit $rc
