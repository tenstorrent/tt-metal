#!/usr/bin/env bash
# Prints what decides how fast an idle CPU wakes up: virtualization, the cpuidle driver and states, and kernel knobs.
p() { echo "IDLE_INFO $*"; }
p "hostname=$(hostname) nproc=$(nproc) cpuset=$(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null)"
p "kernel=$(uname -r) cmdline=$(cat /proc/cmdline 2>/dev/null)"
lscpu | grep -E 'Model name|Hypervisor|Virtualization|Thread\(s\)|NUMA node\(s\)' | sed 's/  */ /g' | while read -r l; do p "$l"; done
d=/sys/devices/system/cpu/cpuidle
p "cpuidle driver=$(cat $d/current_driver 2>/dev/null || echo none) governor=$(cat $d/current_governor_ro 2>/dev/null || cat $d/current_governor 2>/dev/null || echo none)"
for s in /sys/devices/system/cpu/cpu0/cpuidle/state*; do
    [ -d "$s" ] || continue
    p "cpu0 $(basename "$s") name=$(cat "$s"/name) latency_us=$(cat "$s"/latency) residency_us=$(cat "$s"/residency) disable=$(cat "$s"/disable) usage=$(cat "$s"/usage)"
done
for f in /sys/module/haltpoll/parameters/* /sys/module/cpuidle_haltpoll/parameters/*; do
    [ -f "$f" ] && p "$f=$(cat "$f")"
done
p "cpufreq governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo none)"
p "loadavg=$(cat /proc/loadavg)"
