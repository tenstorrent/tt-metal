#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Host CPU / DRAM utilization pane. Deliberately dependency-free (reads /proc
# directly) so it works on a bare node without sysstat/htop installed.
#
# Usage: host_stats.sh            (REFRESH overridable via env, default 5s)
#
# Unlike the other scripts this one does not source common.sh — it is not tied to
# a run or a log dir, it just watches the box the stress loop is running on.

REFRESH="${REFRESH:-5}"
# How many of the hungriest processes (by RSS) to list.
TOP_N="${TOP_N:-6}"

NCPU=$(nproc)

# Aggregate CPU busy/total jiffies from /proc/stat's first line. Utilization is a
# delta between two samples, so a single read is meaningless — the loop keeps the
# previous sample and reports the busy fraction over the interval.
cpu_sample() {
  local _ user nice sys idle iowait irq softirq steal
  read -r _ user nice sys idle iowait irq softirq steal _ < /proc/stat
  local busy=$((user + nice + sys + irq + softirq + steal))
  echo "$busy $((busy + idle + iowait))"
}

# Horizontal bar: bar <percent> <width>
bar() {
  local pct=${1%.*} width="$2" filled i out=""
  [ "$pct" -gt 100 ] && pct=100
  filled=$((pct * width / 100))
  for ((i = 0; i < width; i++)); do
    if [ "$i" -lt "$filled" ]; then out+="█"; else out+="·"; fi
  done
  printf '%s' "$out"
}

read -r prev_busy prev_total <<<"$(cpu_sample)"

while true; do
  sleep "$REFRESH"
  read -r busy total <<<"$(cpu_sample)"
  d_busy=$((busy - prev_busy))
  d_total=$((total - prev_total))
  prev_busy=$busy
  prev_total=$total
  cpu_pct=0
  [ "$d_total" -gt 0 ] && cpu_pct=$((100 * d_busy / d_total))

  # /proc/meminfo is in kB. MemAvailable is the kernel's own estimate of what a
  # new allocation can get, which is what actually matters here — MemFree alone
  # ignores reclaimable page cache and reads alarmingly low on a warm box.
  while read -r key val _; do
    case "$key" in
      MemTotal:) mem_total=$val ;;
      MemAvailable:) mem_avail=$val ;;
      SwapTotal:) swap_total=$val ;;
      SwapFree:) swap_free=$val ;;
    esac
  done < /proc/meminfo
  mem_used=$((mem_total - mem_avail))
  mem_pct=$((100 * mem_used / mem_total))
  swap_pct=0
  [ "${swap_total:-0}" -gt 0 ] && swap_pct=$((100 * (swap_total - swap_free) / swap_total))

  gib() { awk -v k="$1" 'BEGIN{printf "%.1f", k/1048576}'; }

  clear
  echo "══ HOST  $(hostname -s)  ${NCPU} cpu  $(date '+%H:%M:%S')  refresh ${REFRESH}s ═════════"
  printf "  CPU  %3d%%  [%s]  load %s\n" "$cpu_pct" "$(bar "$cpu_pct" 28)" \
    "$(cut -d' ' -f1-3 /proc/loadavg)"
  printf "  DRAM %3d%%  [%s]  %s / %s GiB used\n" "$mem_pct" "$(bar "$mem_pct" 28)" \
    "$(gib "$mem_used")" "$(gib "$mem_total")"
  if [ "${swap_total:-0}" -gt 0 ]; then
    printf "  SWAP %3d%%  [%s]  %s / %s GiB used\n" "$swap_pct" "$(bar "$swap_pct" 28)" \
      "$(gib $((swap_total - swap_free)))" "$(gib "$swap_total")"
  fi
  echo "  ─────────────────────────────────────────────────────────────"
  printf "  %-8s %6s %6s  %s\n" "PID" "%CPU" "RSS_G" "COMMAND"
  # --sort=-rss, not -%cpu: the failure mode this pane exists to catch is the
  # weight load / cache load eating DRAM, not a busy core.
  ps -eo pid=,pcpu=,rss=,comm= --sort=-rss 2>/dev/null | head -"$TOP_N" |
    while read -r pid pcpu rss comm; do
      printf "  %-8s %6s %6.1f  %s\n" "$pid" "$pcpu" "$(awk -v k="$rss" 'BEGIN{print k/1048576}')" "$comm"
    done
done
