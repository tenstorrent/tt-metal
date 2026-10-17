#!/bin/bash
# usage: device_queue.sh [lock-file]
# How many processes hold or wait on this box's device lock (default /tmp/tt-device-$(hostname).lock), whatever
# form took it — `flock <file> <cmd>` or `exec 9>file; flock 9` (legs.sh, perf_ab.sh). Prints the count; use it
# for the §2 queue cap (at most 4 waiting). Exit 0.
set -u
lock=${1:-${TT_DEVICE_LOCK:-/tmp/tt-device-$(hostname).lock}}
[ -e "$lock" ] || { echo 0; exit 0; }
n=$( { lsof -t -- "$lock" 2>/dev/null || fuser "$lock" 2>/dev/null | tr ' ' '\n'; } | sort -u | grep -c . )
echo "$n"
