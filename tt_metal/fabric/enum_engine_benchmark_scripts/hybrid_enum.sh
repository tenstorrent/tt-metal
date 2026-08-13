#!/usr/bin/env bash
set -u
SC=/tmp/claude-4010/-data-rsong-tt-metal2/b9a92f3d-7cfe-426a-b13a-95af54b11ca9/scratchpad
CNF=$1; N=$2; T=${3:-8}; SEED=${4:-0}
GIM=/data/rsong/gimsatul_backup/gimsatul_src/gimsatul
WORK=$(mktemp --suffix=.cnf); cp "$CNF" "$WORK"
t0=$(date +%s.%N); out=$(mktemp)
$GIM --threads=$T "$WORK" > "$out" 2>/dev/null
grep -qa '^s SATISFIABLE' "$out" || { echo "gimsatul #1 NOT SAT"; exit 1; }
printf "hybrid sol 1 (gimsatul cold): %.2fs\n" "$(echo "$(date +%s.%N) - $t0"|bc)"
block=$(grep -a '^v ' "$out" | sed 's/^v//' | tr ' ' '\n' | grep -E '^-?[0-9]+$' | grep -v '^0$' | awk '{print -$1}' | tr '\n' ' ')
V=$(head -1 "$WORK"|awk '{print $3}'); C=$(head -1 "$WORK"|awk '{print $4}')
sed -i "1s/.*/p cnf $V $((C+1))/" "$WORK"; echo "$block 0" >> "$WORK"
"$SC/warm_enum" "$WORK" $((N-1)) "$SEED"
rm -f "$WORK" "$out"
