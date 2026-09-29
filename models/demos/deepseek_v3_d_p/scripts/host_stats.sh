#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Host resource pane: CPU / DRAM, the 1 GB hugepage pool, and the pinning limits
# of the live pytest process. Deliberately dependency-free (reads /proc and sysfs
# directly) so it works on a bare node without sysstat/htop installed.
#
# Usage: host_stats.sh [log_name]      (REFRESH / SNAP_SECS / TOP_N via env)
#
# The hugepage + limit rows exist because of the 2026-08-12 soak failures: eight
# devices logged "tenstorrent 0000:XX:00.0: pin_user_pages_longterm failed: -14"
# (EFAULT) and every run died with SIGBUS in a native thread. tt-kmd pins a
# hugepage-backed host buffer per device for DMA, so the things worth watching are
# how many 1 GB pages are still free and what RLIMIT_MEMLOCK the process actually
# has — neither is visible in meminfo's HugePages_* rows, which only describe the
# default 2 MB pool.
#
# With a log_name argument, a TSV row is appended to <log dir>/host_stats.tsv every
# SNAP_SECS (60s) so the state leading up to a failure survives the crash. It does
# not source common.sh (it is not tied to a run point), so LOG_DIR is recomputed
# here the same way common.sh does it.

REFRESH="${REFRESH:-5}"
SNAP_SECS="${SNAP_SECS:-60}"
# How many of the hungriest processes (by RSS) to list.
TOP_N="${TOP_N:-6}"

NCPU=$(nproc)
HP1G=/sys/kernel/mm/hugepages/hugepages-1048576kB
HP2M=/sys/kernel/mm/hugepages/hugepages-2048kB
# 1 GB hugepages needed per run = PCIe-attached devices (32 on an 8x4 box). tt-kmd
# pins one host DMA buffer per device, so this is the number free_hugepages has to
# reach before an iteration can start.
NDEV=$(ls /sys/bus/pci/drivers/tenstorrent/ 2>/dev/null | grep -c '^0000:')

LOG_NAME="${1:-}"
SNAP_FILE=""
if [ -n "$LOG_NAME" ]; then
  SNAP_DIR="/data/$USER/$LOG_NAME"
  mkdir -p "$SNAP_DIR" 2>/dev/null
  SNAP_FILE="$SNAP_DIR/host_stats.tsv"
fi

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

rd() { cat "$1" 2>/dev/null || echo "-"; }
gib() { awk -v k="$1" 'BEGIN{printf "%.1f", k/1048576}'; }        # from kB
gib_b() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }   # from bytes

# The pytest process under test, matched by node id rather than the bare word
# "pytest" (which would also catch this pane's own pgrep). Empty between outer
# iterations.
#
# Skipping shells is not cosmetic: stress.sh runs pytest via `bash -c "$ENV_VARS
# pytest -vs <target> |& tee ..."`, so the wrapper's command line contains the same
# node id and sorts first by pid. Reporting it would show rss=3 MB / fds=5 / pin=0
# for the whole run — the wrapper's numbers, not the model's.
find_pytest() {
  local p
  for p in $(pgrep -f 'pytest.*test_prefill_transformer_chunked' 2>/dev/null); do
    case "$(cat "/proc/$p/comm" 2>/dev/null)" in
      bash | sh | dash | tee | timeout | pgrep) continue ;;
    esac
    echo "$p"
    return
  done
}

# Soft limit of one /proc/PID/limits row, in bytes ("unlimited" passes through).
proc_limit() {
  awk -F'  +' -v want="$2" '$1 == want {print $2}' "/proc/$1/limits" 2>/dev/null || echo "-"
}

read -r prev_busy prev_total <<<"$(cpu_sample)"
last_snap=0

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
  # Hugetlb is the total carved out for hugepages: it is NOT part of MemAvailable,
  # which is why a box can look 90% free while device pinning still fails.
  hugetlb=0
  while read -r key val _; do
    case "$key" in
      MemTotal:) mem_total=$val ;;
      MemAvailable:) mem_avail=$val ;;
      SwapTotal:) swap_total=$val ;;
      SwapFree:) swap_free=$val ;;
      Hugetlb:) hugetlb=$val ;;
    esac
  done < /proc/meminfo
  mem_used=$((mem_total - mem_avail))
  mem_pct=$((100 * mem_used / mem_total))
  swap_pct=0
  [ "${swap_total:-0}" -gt 0 ] && swap_pct=$((100 * (swap_total - swap_free) / swap_total))

  hp_nr=$(rd $HP1G/nr_hugepages); hp_free=$(rd $HP1G/free_hugepages)
  hp_resv=$(rd $HP1G/resv_hugepages); hp_surp=$(rd $HP1G/surplus_hugepages)
  hp2_nr=$(rd $HP2M/nr_hugepages); hp2_free=$(rd $HP2M/free_hugepages)
  pid=$(find_pytest)

  # A run needs one 1 GB page per PCIe device, so free=0 is the NORMAL state while a
  # run holds them all — flagging that would make this pane cry wolf for the whole
  # soak. What actually predicts the tt-kmd EFAULT is pages still held when no run
  # owns them: the next iteration then cannot get its 32 and dies with SIGBUS.
  hp_flag=""
  if [ -n "$pid" ]; then
    [ "$hp_free" = "0" ] && hp_flag="  (all $hp_nr held by run — normal)"
  elif [ "$hp_nr" != "0" ] && [ "$hp_free" != "$hp_nr" ]; then
    hp_flag="  <<< LEAK? $((hp_nr - hp_free)) page(s) held with no run alive"
  fi

  sh_lock=$(ulimit -l); sh_nofile=$(ulimit -n)
  if [ -n "$pid" ]; then
    p_lock=$(proc_limit "$pid" "Max locked memory")
    p_nofile=$(proc_limit "$pid" "Max open files")
    p_rss=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null)
    p_lck=$(awk '/^VmLck:/{print $2}' "/proc/$pid/status" 2>/dev/null)
    p_pin=$(awk '/^VmPin:/{print $2}' "/proc/$pid/status" 2>/dev/null)
    p_fds=$(ls "/proc/$pid/fd" 2>/dev/null | wc -l)
  else
    p_lock="-"; p_nofile="-"; p_rss=0; p_lck=0; p_pin=0; p_fds="-"
  fi

  clear
  echo "══ HOST  $(hostname -s)  ${NCPU} cpu  $(date '+%H:%M:%S')  refresh ${REFRESH}s ═════════"
  printf "  CPU  %3d%%  [%s]  load %s\n" "$cpu_pct" "$(bar "$cpu_pct" 24)" \
    "$(cut -d' ' -f1-3 /proc/loadavg)"
  printf "  DRAM %3d%%  [%s]  %s / %s GiB  (+%s GiB hugetlb, outside MemAvailable)\n" \
    "$mem_pct" "$(bar "$mem_pct" 24)" "$(gib "$mem_used")" "$(gib "$mem_total")" "$(gib "$hugetlb")"
  if [ "${swap_total:-0}" -gt 0 ]; then
    printf "  SWAP %3d%%  [%s]  %s / %s GiB\n" "$swap_pct" "$(bar "$swap_pct" 24)" \
      "$(gib $((swap_total - swap_free)))" "$(gib "$swap_total")"
  fi
  echo "  ─────────────────────────────────────────────────────────────"
  printf "  HUGE 1G  nr=%-4s free=%-4s resv=%-3s surplus=%-3s need=%s/run%s\n" \
    "$hp_nr" "$hp_free" "$hp_resv" "$hp_surp" "$NDEV" "$hp_flag"
  printf "  HUGE 2M  nr=%-4s free=%-4s\n" "$hp2_nr" "$hp2_free"
  # ulimit -l is in kB; print GiB so it is not mistaken for bytes or pages. EFAULT
  # (-14) from the driver is a bad-address failure, not a limit failure, so these
  # rows are here to rule memlock out rather than because it is the likely cause.
  printf "  ULIMIT   memlock=%s GiB  nofile=%s   (pane shell)\n" \
    "$(gib "$sh_lock")" "$sh_nofile"
  if [ -n "$pid" ]; then
    # lck/pin come from the process mm and read 0 for this workload — tt-kmd's
    # pinning is not charged to it — so they are shown only to prove that, and the
    # hugepage rows above are the signal that matters.
    printf "  PYTEST   pid=%-8s rss=%sG fds=%-6s lck=%sG pin=%sG (mm, not tt-kmd)\n" \
      "$pid" "$(gib "${p_rss:-0}")" "$p_fds" "$(gib "${p_lck:-0}")" "$(gib "${p_pin:-0}")"
    printf "           memlock=%s GiB  nofile=%s\n" "$(gib_b "${p_lock:-0}")" "$p_nofile"
  else
    printf "  PYTEST   (not running — between outer iterations)\n"
  fi
  echo "  ─────────────────────────────────────────────────────────────"
  printf "  %-8s %6s %6s  %s\n" "PID" "%CPU" "RSS_G" "COMMAND"
  # --sort=-rss, not -%cpu: the failure mode this pane exists to catch is host
  # memory / pinned-page pressure, not a busy core.
  ps -eo pid=,pcpu=,rss=,comm= --sort=-rss 2>/dev/null | head -"$TOP_N" |
    while read -r p pcpu rss comm; do
      printf "  %-8s %6s %6.1f  %s\n" "$p" "$pcpu" "$(awk -v k="$rss" 'BEGIN{print k/1048576}')" "$comm"
    done

  # Periodic snapshot. Appended, never rewritten, so it survives the SIGBUS that
  # takes the run down — the point is having the 60s before a failure on disk.
  if [ -n "$SNAP_FILE" ]; then
    now=$(date +%s)
    if [ $((now - last_snap)) -ge "$SNAP_SECS" ]; then
      last_snap=$now
      if [ ! -s "$SNAP_FILE" ]; then
        printf 'ts\tcpu_pct\tload1\tmem_used_kb\tmem_total_kb\thugetlb_kb\tswap_used_kb\thp1g_nr\thp1g_free\thp1g_resv\thp1g_surp\thp2m_nr\thp2m_free\tsh_memlock\tsh_nofile\tpid\tp_rss_kb\tp_lck_kb\tp_pin_kb\tp_fds\tp_memlock\tp_nofile\n' \
          > "$SNAP_FILE"
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date '+%F %T')" "$cpu_pct" "$(cut -d' ' -f1 /proc/loadavg)" \
        "$mem_used" "$mem_total" "$hugetlb" "$((swap_total - swap_free))" \
        "$hp_nr" "$hp_free" "$hp_resv" "$hp_surp" "$hp2_nr" "$hp2_free" \
        "$sh_lock" "$sh_nofile" "${pid:--}" "${p_rss:-0}" "${p_lck:-0}" "${p_pin:-0}" \
        "$p_fds" "$p_lock" "$p_nofile" >> "$SNAP_FILE"
    fi
    printf "  snapshot: %s  every %ss  (%s rows)\n" "$SNAP_FILE" "$SNAP_SECS" \
      "$(($(wc -l < "$SNAP_FILE" 2>/dev/null || echo 1) - 1))"
  fi
done
