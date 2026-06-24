#!/bin/bash
# Run on HOST. Control batch: NO settle after reset (run the smoke immediately), default
# "bad" mux, random routing. If this produces HANGs/early-exits while the +settle campaign
# was 13/13 PASS, the post-reset fabric-readiness race (not the mux) is the driver.
# Does NOT retry on fabric flake -- we WANT to observe the failure distribution.
set -u
C=tt-xla-ird-mvasiljev
D=/home/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
MUX="${MUX:-3,0,4,7}"; N="${N:-6}"
TS=$(date +%Y%m%d_%H%M%S)
SUM="/data/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro/logs/nosettle_${TS}.txt"
echo "=== NO-SETTLE control @ $TS mux=($MUX) N=$N ===" | tee "$SUM"
pass=0; hang=0; ee=0; flake=0; line=""
for i in $(seq 1 "$N"); do
  tt-smi -r >/dev/null 2>&1   # NO settle: run immediately
  out=$(docker exec --user 4123:4123 "$C" /bin/bash -lc "export SMOKE_MUX='$MUX'; bash $D/run_verdict.sh nosettle" 2>&1)
  if echo "$out" | grep -q "could not fit in the discovered physical topology"; then v="FABRIC_FLAKE"; flake=$((flake+1))
  else v=$(echo "$out" | grep -oE 'VERDICT: [A-Z_]+' | head -1 | sed 's/VERDICT: //')
    case "$v" in PASS) pass=$((pass+1));; HANG|TIMEOUT_HANG) hang=$((hang+1));; EARLY_EXIT) ee=$((ee+1));; esac
  fi
  line="$line $i:$v"
  echo "   trial$i -> $v   (PASS=$pass HANG=$hang EARLY=$ee FABRICFLAKE=$flake)" | tee -a "$SUM"
done
echo "   == NO-SETTLE RESULT: PASS=$pass HANG=$hang EARLY=$ee FABRICFLAKE=$flake  [$line ]" | tee -a "$SUM"
echo "=== final reset ==="; tt-smi -r >/dev/null 2>&1; echo "reset rc=$?" | tee -a "$SUM"
echo ">>> NOSETTLE DONE -> $SUM" | tee -a "$SUM"
