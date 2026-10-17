#!/bin/bash
# usage: device_queue.sh [lock-file]
# How many processes hold or wait on this box's device lock (default /tmp/tt-device-$(hostname).lock), whatever
# form took it — `flock <file> <cmd>` or `exec 9>file; flock 9` (legs.sh, perf_ab.sh). Prints the count; use it
# for the §2 queue cap (at most 4 waiting). Exit 0.
# Uses lsof or fuser when present; otherwise scans /proc/*/fd (boxes without either used to report 0 and the
# cap never fired: aus-wh-08 reached a 54-min holder + 5 waiters + 44 remote legs that way).
set -u
lock=${1:-${TT_DEVICE_LOCK:-/tmp/tt-device-$(hostname).lock}}
[ -e "$lock" ] || { echo 0; exit 0; }
real=$(readlink -f "$lock")
if command -v lsof > /dev/null 2>&1; then
    n=$(lsof -t -- "$real" 2>/dev/null | sort -u | grep -c .)
elif command -v fuser > /dev/null 2>&1; then
    n=$(fuser "$real" 2>/dev/null | tr ' ' '\n' | sort -u | grep -c .)
else
    n=$(find /proc/[0-9]*/fd -maxdepth 1 -lname "$real" 2>/dev/null | cut -d/ -f3 | sort -u | grep -c .)
fi
echo "$n"
