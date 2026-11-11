#!/bin/bash
# Kill this user's prefill/ttrun processes on this node and clear its stale shm; print what is left.
me=$(id -u); pat='prefill_runner|prefill_producer|ttrun.py|prterun|prted|run_multirank_pcc'
pkill -u $me -f "$pat"; sleep 4; pkill -9 -u $me -f "$pat"; sleep 1
find /dev/shm -maxdepth 1 -user $me \( -name '*prefill*' -o -name 'TT_UMD_LOCK*' \) -delete 2>/dev/null
echo "$(hostname -s) left=$(pgrep -u $me -f "$pat" | grep -vc "^$$\$")"
