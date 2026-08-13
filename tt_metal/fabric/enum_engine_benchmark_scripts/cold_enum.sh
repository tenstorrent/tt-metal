#!/usr/bin/env bash
# Repeated-cold-gimsatul enumeration: solve CNF -> parse model -> append full-model blocking clause -> re-solve cold.
# Usage: cold_enum.sh <cnf> <N> <threads>. Prints per-solution wall time (incl. DIMACS round-trip overhead).
set -u
CNF_BASE=$1; N=$2; T=${3:-8}
GIM=/data/rsong/gimsatul_backup/gimsatul_src/gimsatul
WORK=$(mktemp --suffix=.cnf); cp "$CNF_BASE" "$WORK"
t0=$(date +%s.%N); prev=0
for i in $(seq 1 "$N"); do
  out=$(mktemp)
  $GIM --threads=$T "$WORK" > "$out" 2>/dev/null
  if ! grep -qa '^s SATISFIABLE' "$out"; then echo "cold sol $i: NOT SAT"; rm -f "$out"; break; fi
  t=$(date +%s.%N); el=$(echo "$t - $t0"|bc); d=$(echo "$el - $prev"|bc)
  printf "cold sol %d: %.2fs (delta %.2fs)\n" "$i" "$el" "$d"; prev=$el
  # full-model blocking clause = negate every model literal
  block=$(grep -a '^v ' "$out" | sed 's/^v//' | tr ' ' '\n' | grep -E '^-?[0-9]+$' | grep -v '^0$' | awk '{print -$1}' | tr '\n' ' ')
  V=$(head -1 "$WORK"|awk '{print $3}'); C=$(head -1 "$WORK"|awk '{print $4}'); newC=$((C+1))
  sed -i "1s/.*/p cnf $V $newC/" "$WORK"
  echo "$block 0" >> "$WORK"
  rm -f "$out"
done
echo "cold TOTAL: $(echo "$(date +%s.%N) - $t0"|bc)s"
rm -f "$WORK"
