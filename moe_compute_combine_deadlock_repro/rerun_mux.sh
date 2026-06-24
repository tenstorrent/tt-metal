#!/bin/bash
# Run on HOST. Re-run specific mux boxes with a post-reset SETTLE and a retry when the
# smoke dies on the intermittent post-reset fabric-init FATAL (set_fabric_config), so the
# verdict reflects the mux, not reset flakiness.
set -u
C=tt-xla-ird-mvasiljev
D=/home/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
SETTLE="${SETTLE:-25}"
BOXES=("$@")
[ ${#BOXES[@]} -eq 0 ] && BOXES=("3,0,4,7" "3,1,4,2")

run_once() {
  local box="$1"
  tt-smi -r >/dev/null 2>&1; echo "   reset rc=$? ; settle ${SETTLE}s"; sleep "$SETTLE"
  docker exec --user 4123:4123 "$C" /bin/bash -lc "export SMOKE_MUX='$box'; bash $D/run_verdict.sh mux_re_${box//,/_}" 2>&1
}

for box in "${BOXES[@]}"; do
  echo "########## mux=($box) ##########"
  for attempt in 1 2 3; do
    out=$(run_once "$box")
    verdict=$(echo "$out" | grep -oE 'VERDICT: [A-Z_]+' | head -1)
    placement=$(echo "$out" | grep -oE 'selected tilize cores [0-9]+, combine cores [0-9]+, matmul cores [0-9]+' | head -1)
    if echo "$out" | grep -q "could not fit in the discovered physical topology"; then
      echo "   attempt $attempt: FABRIC-INIT FLAKE (retrying)"; continue
    fi
    echo "   attempt $attempt: $verdict | $placement"
    break
  done
done
echo "=== final reset ==="; tt-smi -r >/dev/null 2>&1; echo "reset rc=$?"
