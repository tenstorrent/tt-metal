#!/usr/bin/env bash
# One-shot standalone CaDiCaL 2.2.1 (SAME version we link) on the SAME dumped CNFs. This isolates SOLVER vs MODE:
# our production CaDiCaL runs INCREMENTAL (ilb=2, blocking-clause enumeration) which restricts variable elimination;
# standalone runs one-shot with full preprocessing. If one-shot CaDiCaL is also fast, the 205s was the mode, not gimsatul.
set -u
SC=/tmp/claude-4010/-data-rsong-tt-metal2/00fa4a3b-aa18-4f70-b82a-ae18fdf91071/scratchpad
CAD="$SC/cadical_src/build/cadical"
RES="$SC/cad_sweep_results.txt"; : > "$RES"
CAP=420
printf "%-6s %-8s %-12s %-10s %-14s\n" size vars cad_oneshot verdict elim_vars | tee -a "$RES"
for N in 64 80 96 112 128 144; do
  f="$SC/2x4_$N.cnf"; V=$(head -1 "$f" | awk '{print $3}')
  start=$(date +%s%3N)
  out=$(timeout $CAP "$CAD" "$f" 2>&1)
  end=$(date +%s%3N)
  s=$(echo "$out" | grep -aoE "^s [A-Z]+" | head -1)
  # CaDiCaL prints "c eliminated <n>" style stats; capture eliminated var count if present
  elim=$(echo "$out" | grep -aoiE "eliminated:?[ ]+[0-9]+" | grep -aoE "[0-9]+" | head -1)
  w=$([ -n "$s" ] && echo "$(( end-start ))ms" || echo ">${CAP}s")
  printf "%-6s %-8s %-12s %-10s %-14s\n" "$N" "$V" "$w" "${s:-TIMEOUT}" "${elim:-?}" | tee -a "$RES"
done
echo "[$(date -u +%T)] CAD ONE-SHOT SWEEP DONE" | tee -a "$RES"
