#!/bin/bash
# Quick test of the port's write gap (RECONCILE_DEVICE.md section 8): is the test sender kernel the difference?
#   ASBUILT  the port 6dad5fec638 as built (main's test sender: send_packets_stateful in the single-config benchmark
#            branch).
#   LOOP     the same tree with that branch running the study's one-segment stream-benchmark loop instead (local copy of
#            the connection, free-slot checks only when the known count runs out, header replay by the credit-only
#            update, copy written back; no new compile or runtime args, no stream-counter write). Uncommitted, kernel
#            only (tests/tt_metal/tt_fabric/test_infra/kernels/tt_fabric_test_sender.cpp).
# One host binary (/home/aagarwal/tt-metal-port-gap/build_Release); the test kernels JIT-compile from each arm's
# runtime root (/home/aagarwal/tt-metal-port-gap-roots/<arm>: symlinks to the worktree, its own copy of tests/ and its
# own generated/), so the arms differ only in that one source file (checked below). The router images of the two arms
# were identical in a mock build (elf_load_cmp.py) and are compared again from this job's caches at the end.
# Rows: fabric_exp/port_slots/yamls/pvs_ns.yaml (P150 2x4 mesh, 1-hop N/S and E/W unicast writes at 16 and 4352 B,
# 4352 B atomic increments as a control; 50,000 packets; slots 17,16,16 as the study's row, checked by a mock probe).
# Pass a: asbuilt then loop; pass b: loop then asbuilt. Fresh JIT cache per arm. BW telemetry off.
# Broker job on bh-lb-120-a09u32 (workspace /home/aagarwal, timeout 1500 s): bash fabric_exp/port_tune/gap_loop_ab.sh
# Results: fabric_exp/results/gaploop_<stamp>/gl<a|b>_<arm>.log, .csv, _tests.csv; report gap_loop_report.py.
# DRY=1: the checks, then the runs it would do (no device, caches untouched).
set -uo pipefail
cd /home/aagarwal/tt-metal
CON=fabric_exp/consolidation
D=fabric_exp/port_tune
source $CON/vc1_capture_lib.sh
source fabric_exp/port_slots/port_lib.sh
R=/home/aagarwal/tt-metal-port-gap-roots
TREE=/home/aagarwal/tt-metal-port-gap
YAML=fabric_exp/port_slots/yamls/pvs_ns.yaml
K=tests/tt_metal/tt_fabric/test_infra/kernels/tt_fabric_test_sender.cpp
RUN_TIMEOUT=240
out=/home/aagarwal/tt-metal/fabric_exp/results/gaploop_$(date +%Y%m%d-%H%M%S)
(while sleep 30; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
echo "gap_loop_ab: $(hostname), results $out"

# --- checks before any device run
[ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! expected bh-lb-120-a09u32"; exit 1; }
[ "$(git -C $TREE rev-parse --short=11 HEAD)" = 6dad5fec638 ] || { echo "!!! $TREE is not at 6dad5fec638"; exit 1; }
want_asbuilt=$(git -C $TREE show HEAD:$K | md5sum | cut -d' ' -f1)
want_loop=5d091b123a0051587deb27824edc0265
got_asbuilt=$(md5sum < $R/asbuilt/$K | cut -d' ' -f1)
got_loop=$(md5sum < $R/loop/$K | cut -d' ' -f1)
echo "sender source md5: asbuilt $got_asbuilt (HEAD $want_asbuilt), loop $got_loop (expected $want_loop)"
[ "$got_asbuilt" = "$want_asbuilt" ] && [ "$got_loop" = "$want_loop" ] || { echo "!!! sender sources changed"; exit 1; }
d=$(diff -rq $R/asbuilt/tests $R/loop/tests)
[ "$d" = "Files $R/asbuilt/$K and $R/loop/$K differ" ] || { echo "!!! roots differ beyond the sender:"; echo "$d" | head; exit 1; }
for arm in asbuilt loop; do
    for e in $(ls -A $R/$arm); do
        case $e in tests|generated) [ -d $R/$arm/$e ] && [ ! -L $R/$arm/$e ] || { echo "!!! $R/$arm/$e"; exit 1; } ;;
            *) [ "$(readlink $R/$arm/$e)" = "$TREE/$e" ] || { echo "!!! $R/$arm/$e is not the worktree's"; exit 1; } ;;
        esac
    done
done
echo "roots: only the sender kernel differs; everything else is the worktree's ($TREE)"

declare -A CACHE=([asbuilt]=/home/aagarwal/.tt-metal-cache-gaploop-asbuilt [loop]=/home/aagarwal/.tt-metal-cache-gaploop-loop)
if [ -n "${DRY:-}" ]; then
    run_port() { echo "DRY: run_port $* (TT_METAL_HOME=$PT, cache $C, timeout per run $3 s)"; }
else
    mkdir -p $out
    rm -rf "${CACHE[asbuilt]:?}" "${CACHE[loop]:?}"
fi

# --- the runs
rc=0
for pass in a b; do
    arms="asbuilt loop"; [ $pass = b ] && arms="loop asbuilt"
    for arm in $arms; do
        need=$((RUN_TIMEOUT + 45 + 60 + VC1CAP_CAPTURE_TIMEOUT + 120 + 20))
        if [ $((SECONDS + need)) -gt 1450 ]; then
            echo "=== stopping before gl${pass}_$arm: job at ${SECONDS} s, a hang could need ${need} s more"; rc=1; break 2
        fi
        PT=$R/$arm; C=${CACHE[$arm]}
        run_port gl${pass}_$arm $YAML $RUN_TIMEOUT TT_METAL_RUNTIME_ROOT=$PT; r=$?
        if [ $r != 0 ]; then echo "!!! gl${pass}_$arm failed (rc $r): stopping"; rc=1; break 2; fi
    done
done

[ -n "${DRY:-}" ] && { echo "DRY done (${SECONDS} s)"; exit 0; }
# --- which source each arm's sender came from, and the images of the two arms
for arm in asbuilt loop; do
    src=$(grep -ah "tt_fabric_test_sender.cpp" ${CACHE[$arm]}/*/kernels/tt_fabric_test_sender/*/kernel_includes.hpp 2>/dev/null | grep -o '"[^"]*"' | sort -u | tr '\n' ' ')
    echo "$arm sender built from: ${src:-none found}"
done
python3 $D/elf_load_cmp.py ${CACHE[asbuilt]} ${CACHE[loop]} fabric_erisc_router tt_fabric_test_sender \
    && echo "router images identical between the arms" || { echo "!!! router images differ between the arms"; rc=1; }
python3 $D/gap_loop_report.py $out | tee $out/report.md
echo "gap_loop_ab done (rc $rc, ${SECONDS} s)"
exit $rc
