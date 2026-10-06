#!/bin/bash
# qualify.sh: read-only. Prints "<idle_min> <node> <LastBusyTime>" for each dit node (Wan/DiT team rows
# 120-B23 and 120-B45 of the Markham Allocation Weekly Schedule) that is State=IDLE now and has
# LastBusyTime >= MIN_IDLE (default 120) minutes ago, most idle first. Exit 0 if any qualifies, else 1.
# With -v, also prints every dit node's state to stderr.
NODES="bh-glx-120-b02u02 bh-glx-120-b02u08 bh-glx-120-b03u02 bh-glx-120-b03u08 bh-glx-120-b04u02 bh-glx-120-b04u08 bh-glx-120-b05u02 bh-glx-120-b05u08"
MIN_IDLE=${MIN_IDLE:-120}
now=$(date +%s); out=""
for n in $NODES; do
  l=$(scontrol show node -o "$n" 2>/dev/null) || continue
  st=$(grep -oP ' State=\K\S+' <<<"$l"); lb=$(grep -oP 'LastBusyTime=\K\S+' <<<"$l")
  [ "$1" = -v ] && echo "$n State=$st LastBusyTime=$lb" >&2
  [ "$st" = IDLE ] && [ -n "$lb" ] && [ "$lb" != None ] || continue
  m=$(( (now - $(date -d "$lb" +%s)) / 60 ))
  [ $m -ge $MIN_IDLE ] && out+="$m $n $lb"$'\n'
done
[ -n "$out" ] || exit 1
printf %s "$out" | sort -rn
