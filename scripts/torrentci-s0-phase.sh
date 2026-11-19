#!/bin/sh
# torrentci#3196 S0: run one phase and log container cgroup CPU counters and
# wall time as one grep-able "S0 arm=... phase=..." line. Evaluation only.
# Usage: torrentci-s0-phase.sh <phase> <command...>
phase=$1
shift
stat=/sys/fs/cgroup/cpu.stat
get() { awk -v k="$1" '$1 == k { print $2 }' "$stat"; }
u0=$(get usage_usec); p0=$(get nr_periods); n0=$(get nr_throttled); t0=$(get throttled_usec)
w0=$(date +%s.%N)
l0=$(cut -d' ' -f1 /proc/loadavg)
echo "S0 arm=$S0_ARM phase=$phase event=start load1=$l0 cpu.stat: $(tr '\n' ' ' < "$stat")"
rc=0
"$@" || rc=$?
w1=$(date +%s.%N)
l1=$(cut -d' ' -f1 /proc/loadavg)
u1=$(get usage_usec); p1=$(get nr_periods); n1=$(get nr_throttled); t1=$(get throttled_usec)
echo "S0 arm=$S0_ARM phase=$phase event=end cpu.stat: $(tr '\n' ' ' < "$stat")"
awk -v arm="$S0_ARM" -v ph="$phase" -v j="${CMAKE_BUILD_PARALLEL_LEVEL-unset}" -v rc="$rc" \
  -v w0="$w0" -v w1="$w1" -v u0="$u0" -v u1="$u1" -v p0="${p0:-0}" -v p1="${p1:-0}" \
  -v n0="${n0:-0}" -v n1="${n1:-0}" -v t0="${t0:-0}" -v t1="${t1:-0}" -v l0="$l0" -v l1="$l1" \
  -v ncpu="$(grep -c ^processor /proc/cpuinfo)" 'BEGIN {
    wall = w1 - w0; du = u1 - u0; dp = p1 - p0; dn = n1 - n0
    printf "S0 arm=%s phase=%s j=%s rc=%s wall_s=%.1f d_usage_usec=%.0f cores=%.2f d_nr_periods=%d d_nr_throttled=%d thr_ratio=%.3f d_throttled_usec=%.0f load1_start=%s load1_end=%s node_cpus=%s\n",
      arm, ph, j, rc, wall, du, (wall > 0 ? du / 1e6 / wall : 0), dp, dn, (dp > 0 ? dn / dp : 0), t1 - t0, l0, l1, ncpu
  }' | tee -a /tmp/s0-summary.txt
exit "$rc"
