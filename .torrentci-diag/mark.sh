#!/bin/sh
# TEMPORARY (torrentci#2377 diagnostic): one timeline line with the cgroup's
# cumulative CPU time, to compute cores used per step.
cpu=$(sed -n 's/^usage_usec //p' /sys/fs/cgroup/cpu.stat 2>/dev/null)
echo "MARK $1 $(date -u +%Y-%m-%dT%H:%M:%S.%3NZ) cpu_usec=${cpu:-na} load=$(cut -d' ' -f1-3 /proc/loadavg)"
