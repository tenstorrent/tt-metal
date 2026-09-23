#!/bin/bash
# Function-level presence: which p150 defs are absent from main?
cd /local/ttuser/gtobar || exit 1
P=p150_reference/models/demos/wormhole/bge_m3
M=tt-metal/models/demos/wormhole/bge_m3
printf "%-38s %5s %5s %5s  %s\n" FILE P150 MAIN MISS STATUS
for f in "$@"; do
  if [ ! -f "$P/$f" ]; then printf "%-38s %5s %5s %5s  %s\n" "$f" "-" "-" "-" "NOT-IN-P150"; continue; fi
  if [ ! -f "$M/$f" ]; then printf "%-38s %5s %5s %5s  %s\n" "$f" "?" "-" "?" "NEW-FROM-P150"; continue; fi
  grep -oE '^[[:space:]]*def [a-zA-Z_0-9]+' "$P/$f" | sed 's/.*def //' | sort -u > /tmp/_p.txt
  grep -oE '^[[:space:]]*def [a-zA-Z_0-9]+' "$M/$f" | sed 's/.*def //' | sort -u > /tmp/_m.txt
  np=$(wc -l < /tmp/_p.txt); nm=$(wc -l < /tmp/_m.txt)
  comm -23 /tmp/_p.txt /tmp/_m.txt > /tmp/_miss.txt
  miss=$(wc -l < /tmp/_miss.txt)
  if [ "$miss" -eq 0 ]; then st="ALL-IN-MAIN"; else st="MISSING: $(tr '\n' ' ' < /tmp/_miss.txt)"; fi
  printf "%-38s %5s %5s %5s  %s\n" "$f" "$np" "$nm" "$miss" "$st"
done
