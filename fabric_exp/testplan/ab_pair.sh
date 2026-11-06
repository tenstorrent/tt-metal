#!/bin/bash
# One A/B unit as one broker job: one pass of GROUP per router, in the order given, no device reset in between.
#   ab_pair.sh GROUP PASS BUDGET_S ROUTER [ROUTER ...]     e.g. ab_pair.sh LB-P 2 600 new main
# Trees: main -> manifest tree `main`; new -> $NEW_TREE (default new_288afd2). Label: $AB_LABEL (default ab_288afd2).
# Resets: never inside a broker job (a09u32 rule, coordinator 2026-11-04: resets only through tt_device_reset, the
# broker's post-job health gate handles recovery). AB_RESET=1 asks for `tt-smi -r` before each arm and is refused on a
# broker host (the manifest's systems.*.broker.node, the check run_group.py's job guard uses); off-broker hosts only.
# AB_OUT: the results root (run_group.py run --out). On an lb2 node: AB_RESET=1, AB_ANY_NODE=1,
# AB_OUT=/data/aagarwal/regress/results-lb2.
# Exit: 3 if any arm had a step killed (the device needs a reset), 5 if refused, else the largest arm rc.
set -u
G=$1; P=$2; B=$3; shift 3
HERE=/home/aagarwal/tt-metal/fabric_exp/testplan
RG="python3 $HERE/run_group.py"
LABEL=${AB_LABEL:-ab_288afd2}
NEW_TREE=${NEW_TREE:-new_288afd2}
OUT=${AB_OUT:+--out $AB_OUT}   # results root (lb2 jobs: /data/aagarwal/regress/results-lb2, collected from there)
ANY=${AB_ANY_NODE:+--any-node}  # an lb2 node is neither the broker host nor (to run_group) a Slurm job: lb2.sbatch strips SLURM_*
on_broker_host() {
    python3 - "$HERE/manifest.yaml" <<'EOF'
import socket, sys
try:
    import yaml
    man = yaml.safe_load(open(sys.argv[1]))
    brokers = {(s.get("broker") or {}).get("node") for s in man["systems"].values()} - {None}
except Exception:
    sys.exit(0)  # cannot tell: treat as a broker host (fail safe)
sys.exit(0 if socket.gethostname() in brokers else 1)
EOF
}
if [ "${AB_RESET:-0}" = 1 ] && on_broker_host; then
    echo "ab_pair: refusing AB_RESET=1 on a broker host ($(hostname)): no tt-smi inside broker jobs; use tt_device_reset between jobs"
    exit 5
fi
worst=0; killed=0
for r in "$@"; do
    case $r in main) t=main;; new) t=$NEW_TREE;; *) echo "ab_pair: unknown router $r"; exit 2;; esac
    if [ "${AB_RESET:-0}" = 1 ]; then
        echo "[ab_pair] $(date -u +%H:%M:%S) reset before $G pass $P router $r (off-broker host)"
        tt-smi -r > /dev/null 2>&1 || { echo "[ab_pair] tt-smi -r failed"; exit 4; }
    fi
    echo "[ab_pair] $(date -u +%H:%M:%S) $G pass $P router $r"
    # shellcheck disable=SC2086
    $RG run --group "$G" --tree "$t" --router "$r" --label "$LABEL" --pass "$P" --budget-s "$B" --resume $OUT $ANY
    rc=$?
    echo "[ab_pair] $(date -u +%H:%M:%S) $G pass $P router $r rc=$rc"
    [ $rc -eq 3 ] && killed=1
    [ $rc -gt $worst ] && worst=$rc
done
[ $killed -eq 1 ] && exit 3
exit $worst
