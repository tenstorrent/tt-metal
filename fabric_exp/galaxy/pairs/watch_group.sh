#!/bin/bash
# watch_group.sh GROUP: wait until a sibling of pair group GROUP starts (or all are gone), print the state, then wait
# until no sibling is left in the queue. squeue errors count as "still queued".
G=$1
first=""
while true; do
    out=$(timeout 60 squeue -u aagarwal -h -o "%i|%j|%T|%N" 2>/dev/null) || { sleep 60; continue; }
    grp=$(grep "glxp-$G-" <<< "$out")
    run=$(grep -E "\|(RUNNING|COMPLETING)\|" <<< "$grp")
    if [ -n "$run" ] && [ -z "$first" ]; then
        first=1; echo "STARTED $(date +%T):"; echo "$grp"
    fi
    if [ -z "$grp" ]; then echo "GROUP_DONE $(date +%T)"; break; fi
    sleep 60
done
