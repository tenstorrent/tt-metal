#!/bin/sh
# TEMPORARY (torrentci#2377 diagnostic): what happened before the first step.
echo "TIMELINE now $(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
echo "TIMELINE pid1_start $(ps -o lstart= -p 1) (local tz $(date +%Z))"
echo "TIMELINE uptime_s $(cut -d' ' -f1 /proc/uptime) pid1_starttime_ticks $(cut -d' ' -f22 /proc/1/stat) hz $(getconf CLK_TCK)"
for f in /workspace /workspace/.git /workspace/.git/HEAD /workspace/.git/index /workspace/.git/packed-refs /workspace/.git/FETCH_HEAD /workspace/tt_metal; do
  [ -e "$f" ] && echo "TIMELINE stat $f birth=$(stat -c %w "$f") modify=$(stat -c %y "$f")"
done
du -sh /workspace/.git 2>/dev/null | sed 's/^/TIMELINE git_size /'
git -C /workspace count-objects -vH 2>/dev/null | sed 's/^/TIMELINE /'
git -C /workspace rev-parse --is-shallow-repository 2>/dev/null | sed 's/^/TIMELINE shallow /'
nproc | sed 's/^/TIMELINE nproc /'
cat /sys/fs/cgroup/cpu.max 2>/dev/null | sed 's/^/TIMELINE cpu.max /'
grep -m1 'model name' /proc/cpuinfo | sed 's/^/TIMELINE /'
