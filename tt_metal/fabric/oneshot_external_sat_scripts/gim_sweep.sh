#!/usr/bin/env bash
# Raw one-shot performance sweep: gimsatul (clause-sharing) at 1 vs 16 threads across the 2x4 base-embedding CNFs.
# These CNFs are the HARD encoding only (NO_MINHOST) -- no soft/preferred, no host-min objective.
set -u
SC=/tmp/claude-4010/-data-rsong-tt-metal2/00fa4a3b-aa18-4f70-b82a-ae18fdf91071/scratchpad
G="$SC/gimsatul/gimsatul"
RES="$SC/gim_sweep_results.txt"; : > "$RES"
CAP=420
printf "%-6s %-8s %-10s %-12s %-12s %-12s\n" size vars gim_t1 gim_t8 gim_t16 gim_t32 | tee -a "$RES"
run() { # size threads
  local f="$SC/2x4_$1.cnf" T=$2
  [ -f "$f" ] || { echo "TIMEOUT"; return; }
  local start=$(date +%s%3N)
  local s=$(timeout $CAP "$G" "$f" --threads=$T 2>&1 | grep -aoE "^s [A-Z]+" | head -1)
  local end=$(date +%s%3N)
  if [ -z "$s" ]; then echo ">${CAP}s"; else echo "$(( end-start ))ms"; fi
}
for N in 64 80 96 112 128 144; do
  V=$(head -1 "$SC/2x4_$N.cnf" 2>/dev/null | awk '{print $3}')
  t1=$(run $N 1); t8=$(run $N 8); t16=$(run $N 16); t32=$(run $N 32)
  printf "%-6s %-8s %-10s %-12s %-12s %-12s\n" "$N" "$V" "$t1" "$t8" "$t16" "$t32" | tee -a "$RES"
done
echo "[$(date -u +%T)] GIM SWEEP DONE" | tee -a "$RES"
