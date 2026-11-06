#!/bin/bash
# Completion-proof counts (diagnostic; the runs' cycles are not scored, only the counters): how many status reads a
# group's proof takes and how often it hits the cap (COMPLETION_PROOF_MAX_CHECKS = 32), and how often a single-slot
# (G 1) check defers.
#   Router: 6dad5fec638 with LOCAL counters in router_receiver.hpp (runtime root
#           /home/aagarwal/tt-metal-port-gap-roots/proof_count: symlinks into /home/aagarwal/tt-metal-port-testinfra,
#           its own copy of that header). Counters in core-local memory (.bss), written at the end of the receiver
#           loop into the first slot of the VC0 receiver ring; E0 and test kernel images are identical to the
#           worktree's (mock check).
#   Host and test infra: /home/aagarwal/tt-metal-port-testinfra (6dad5fec638 + fabric_exp/streams/port/
#           test_infra_6dad.patch: the stream rows, and the stream-loop sender for single-pattern benchmarks, as in the
#           receiver tuning sweep).
#   Rows: port_tune/proof/yamls/*.yaml, one test each, one run each: the forwarded fused / atomic / scatter / write
#           rows (where c128 moved), E/W fused, 1-hop E/W write 16 / 4352 B, scatter, atomic, 2D 3-hop write, half
#           ring multicast write (G 1). Pass a in name order, pass b reversed.
#   After each run (every process has exited): hang_capture.py (read-only L1 dump of every Ethernet core), then
#   proof_counts.py; the L1 dumps are deleted after parsing (index.json and the parsed JSON stay).
# Broker job on bh-lb-120-a09u32 (workspace /home/aagarwal, timeout 1500 s): bash fabric_exp/port_tune/proof/proof_runs.sh
# DRY=1: the checks and the runs it would do.
set -uo pipefail
cd /home/aagarwal/tt-metal
CON=fabric_exp/consolidation
D=fabric_exp/port_tune/proof
source $CON/vc1_capture_lib.sh
source fabric_exp/port_slots/port_lib.sh
R=/home/aagarwal/tt-metal-port-gap-roots/proof_count
TREE=/home/aagarwal/tt-metal-port-testinfra
H=tt_metal/fabric/hw/inc/edm_fabric/router/router_receiver.hpp
RUN_TIMEOUT=180
out=/home/aagarwal/tt-metal/fabric_exp/results/proof_$(date +%Y%m%d-%H%M%S)
(while sleep 30; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
echo "proof_runs: $(hostname), results $out"

[ "$(hostname)" = bh-lb-120-a09u32 ] || { echo "!!! expected bh-lb-120-a09u32"; exit 1; }
[ "$(git -C $TREE rev-parse --short=11 HEAD)" = 6dad5fec638 ] || { echo "!!! $TREE is not at 6dad5fec638"; exit 1; }
[ -z "$(git -C $TREE status --porcelain -- tt_metal)" ] || { echo "!!! $TREE has tt_metal changes"; exit 1; }
[ "$(md5sum < $TREE/build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric | cut -d' ' -f1)" = \
  cc6302303a10010ff5571c12501aa0f7 ] || { echo "!!! the test binary changed"; exit 1; }
[ "$(md5sum < $R/$H | cut -d' ' -f1)" = 77e7e5a0754f31aa543d0cc921191474 ] || { echo "!!! the counter header changed"; exit 1; }
[ "$(readlink $R/tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp)" = "$TREE/$(dirname $H)/router_sender.hpp" ] \
    || { echo "!!! the counter root's other headers are not the worktree's"; exit 1; }
mapfile -t ROWS < <(ls $D/yamls/*.yaml | xargs -n1 basename | sed 's/\.yaml$//')
echo "rows (${#ROWS[@]}): ${ROWS[*]}"
C=/home/aagarwal/.tt-metal-cache-proof-count
if [ -n "${DRY:-}" ]; then
    run_port() { echo "DRY: run_port $* (TT_METAL_HOME=$PT, cache $C)"; }
else
    mkdir -p $out
    rm -rf "${C:?}"
fi

rc=0
prev=""
for pass in a b; do
    order=("${ROWS[@]}")
    [ $pass = b ] && mapfile -t order < <(printf '%s\n' "${ROWS[@]}" | tac)
    for row in "${order[@]}"; do
        label=pr${pass}_$row
        need=$((RUN_TIMEOUT + 45 + 60 + VC1CAP_CAPTURE_TIMEOUT + 120 + 20))
        if [ $((SECONDS + need)) -gt 1450 ]; then
            echo "=== stopping before $label: job at ${SECONDS} s, a hang could need ${need} s more"; rc=1; break 2
        fi
        PT=$R
        run_port $label $D/yamls/$row.yaml $RUN_TIMEOUT TT_METAL_RUNTIME_ROOT=$PT; r=$?
        if [ $r != 0 ]; then echo "!!! $label failed (rc $r): stopping"; rc=1; break 2; fi
        [ -n "${DRY:-}" ] && { echo "DRY: capture + proof_counts.py for $label"; continue; }
        cap=$out/${label}_cap
        env -u TT_VISIBLE_DEVICES timeout 200 python_env/bin/python $CON/hang_capture.py $cap --no-workers --no-txq \
            --time-budget 150 --meta label=$label 2>&1 | grep --line-buffered -E "^l1:|error|Error" | sed 's/^/   | /'
        python3 $D/proof_counts.py $cap ${prev:+--prev $prev} --json $out/${label}_proof.json \
            > $out/${label}_proof.txt 2>&1 || echo "   !!! proof_counts.py failed"
        echo "   $label: $(python3 -c "import json; r=json.load(open('$out/${label}_proof.json')); print(len(r), 'records,', sum(1 for x in r if x['stale']), 'stale')" 2>/dev/null)"
        prev=$out/${label}_proof.json
        rm -f $cap/chip*/eth*_l1.bin
    done
done
[ -n "${DRY:-}" ] && { echo "DRY done (${SECONDS} s)"; exit 0; }
python3 $D/proof_report.py $out | tee $out/report.md
echo "proof_runs done (rc $rc, ${SECONDS} s)"
exit $rc
