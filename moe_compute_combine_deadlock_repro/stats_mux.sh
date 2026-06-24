#!/bin/bash
# Run on HOST. Repeated-trial characterization of the fused-combine deadlock.
# Consistent procedure: reset + settle before EVERY run; retry only on the post-reset
# fabric-init flake (set_fabric_config FATAL). Records PASS/HANG per trial so we can
# measure hang RATE per (mux, seed) and decide: deterministic-by-data vs race vs mux.
set -u
C=tt-xla-ird-mvasiljev
D=/home/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
SETTLE="${SETTLE:-25}"
TS=$(date +%Y%m%d_%H%M%S)
SUM="/data/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro/logs/stats_mux_${TS}.txt"

# batch: "label|mux|seed|N"   seed empty = random routing each run
BATCHES=(
  "A_defmux_seed1234|3,0,4,7|1234|4"
  "B_defmux_random|3,0,4,7||5"
  "C_modelmux_random|1,1,3,3||5"
)

trial() {  # $1=mux $2=seed -> echoes verdict
  local mux="$1" seed="$2" envs
  for attempt in 1 2 3; do
    tt-smi -r >/dev/null 2>&1; sleep "$SETTLE"
    envs="export SMOKE_MUX='$mux';"
    [ -n "$seed" ] && envs="$envs export SMOKE_SEED='$seed';"
    out=$(docker exec --user 4123:4123 "$C" /bin/bash -lc "$envs bash $D/run_verdict.sh stat" 2>&1)
    if echo "$out" | grep -q "could not fit in the discovered physical topology"; then continue; fi
    echo "$out" | grep -oE 'VERDICT: [A-Z_]+' | head -1 | sed 's/VERDICT: //'
    return
  done
  echo "FLAKE3X"
}

echo "=== stats @ $TS  settle=${SETTLE}s ===" | tee "$SUM"
for b in "${BATCHES[@]}"; do
  IFS='|' read -r label mux seed N <<<"$b"
  echo "" | tee -a "$SUM"
  echo "### $label  mux=($mux) seed='${seed:-random}' N=$N" | tee -a "$SUM"
  pass=0; hang=0; other=0; line=""
  for i in $(seq 1 "$N"); do
    v=$(trial "$mux" "$seed")
    case "$v" in
      PASS) pass=$((pass+1));;
      HANG|TIMEOUT_HANG) hang=$((hang+1));;
      *) other=$((other+1));;
    esac
    line="$line $i:$v"
    echo "   trial$i -> $v   (running: PASS=$pass HANG=$hang OTHER=$other)" | tee -a "$SUM"
  done
  echo "   == $label RESULT: PASS=$pass HANG=$hang OTHER=$other  [$line ]" | tee -a "$SUM"
done
echo "" | tee -a "$SUM"
echo "=== final reset ===" | tee -a "$SUM"; tt-smi -r >/dev/null 2>&1; echo "reset rc=$?" | tee -a "$SUM"
echo ">>> STATS DONE -> $SUM" | tee -a "$SUM"
