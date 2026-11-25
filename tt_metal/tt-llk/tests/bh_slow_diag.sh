#!/usr/bin/env bash
# Trial only (perf-trial/bh-slow-*): print the host and cgroup state of a perf leg.
# Usage: bh_slow_diag.sh <start|compile_end|end>   or   bh_slow_diag.sh sampler <seconds>
# Every line starts with "BHDIAG <mode>" so the job log can be parsed.
set -u
MODE="${1:-snap}"
STATE=/tmp/bhdiag
mkdir -p "$STATE"
CG=/sys/fs/cgroup

cpustat_file() {
  if [[ -f $CG/cpu.stat ]]; then echo $CG/cpu.stat
  elif [[ -f $CG/cpu/cpu.stat ]]; then echo $CG/cpu/cpu.stat
  elif [[ -f $CG/cpu,cpuacct/cpu.stat ]]; then echo $CG/cpu,cpuacct/cpu.stat
  fi
}
# usage_usec user_usec system_usec nr_periods nr_throttled throttled_usec (v1: throttled_time ns -> usec)
cpustat_vals() {
  local f; f=$(cpustat_file)
  [[ -n $f ]] || { echo "NA NA NA NA NA NA"; return; }
  awk '{v[$1]=$2} END {
    tu = ("throttled_usec" in v) ? v["throttled_usec"] : (("throttled_time" in v) ? int(v["throttled_time"]/1000) : "NA");
    printf "%s %s %s %s %s %s\n", ("usage_usec" in v)?v["usage_usec"]:"NA", ("user_usec" in v)?v["user_usec"]:"NA",
      ("system_usec" in v)?v["system_usec"]:"NA", ("nr_periods" in v)?v["nr_periods"]:"NA",
      ("nr_throttled" in v)?v["nr_throttled"]:"NA", tu }' "$f"
}
# node-wide /proc/stat: total idle iowait steal (jiffies)
procstat() { awk '/^cpu /{t=0; for(i=2;i<=NF;i++) t+=$i; print t, $5, $6, $9; exit}' /proc/stat; }
mhz() { awk -F: '/^cpu MHz/{s+=$2; n++} END {if (n) printf "%.0f", s/n; else printf "NA"}' /proc/cpuinfo; }
psi_cg() { [[ -f $CG/cpu.pressure ]] && awk '{printf "%s:%s:%s ", $1, $2, $5}' $CG/cpu.pressure || printf "NA"; }
psi_node() { [[ -f /proc/pressure/cpu ]] && awk '{printf "%s:%s:%s ", $1, $2, $5}' /proc/pressure/cpu || printf "NA"; }

if [[ $MODE == sampler ]]; then
  INT="${2:-15}"
  read -r t0 i0 w0 s0 < <(procstat); read -r u0 _ _ p0 n0 th0 < <(cpustat_vals); T0=$(date +%s)
  trap 'kill $SLP 2>/dev/null; exit 0' TERM INT
  while :; do
    sleep "$INT" & SLP=$!; wait $SLP
    read -r t1 i1 w1 s1 < <(procstat); read -r u1 _ _ p1 n1 th1 < <(cpustat_vals)
    dt=$((t1 - t0)); [[ $dt -gt 0 ]] || dt=1
    busy=$(awk -v a=$dt -v b=$((i1 - i0 + w1 - w0)) 'BEGIN{printf "%.1f", 100*(a-b)/a}')
    steal=$(awk -v a=$dt -v b=$((s1 - s0)) 'BEGIN{printf "%.1f", 100*b/a}')
    if [[ $u1 != NA ]]; then
      cores=$(awk -v u=$((u1 - u0)) -v s="$INT" 'BEGIN{printf "%.2f", u/1e6/s}')
      thr="$((n1 - n0))"; thu="$((th1 - th0))"
    else cores=NA; thr=NA; thu=NA; fi
    echo "BHDIAG sample t=$(( $(date +%s) - T0 )) load=$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' ,) node_busy_pct=$busy steal_pct=$steal cg_cores_used=$cores nr_throttled_d=$thr throttled_usec_d=$thu mhz=$(mhz) psi_cg=$(psi_cg | tr ' ' ,) psi_node=$(psi_node | tr ' ' ,)"
    t0=$t1; i0=$i1; w0=$w1; s0=$s1; u0=$u1; n0=$n1; th0=$th1
  done
  exit 0
fi

P="BHDIAG $MODE"
echo "$P time=$(date -u +%FT%TZ) epoch=$(date +%s) hostname=$(hostname) runner=${RUNNER_NAME:-NA}"
env | grep -E '^(NODE|K8S|KUBE|POD|MY_NODE|ACTIONS_RUNNER_POD|RUNNER_NAME)' | sed "s/^/$P env /" || true
echo "$P nproc=$(nproc) online=$(cat /sys/devices/system/cpu/online 2>/dev/null) mhz_avg=$(mhz)"
lscpu 2>/dev/null | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket|NUMA node\(s\)|CPU (max |min )?MHz|Hypervisor)' | sed "s/^/$P lscpu /" | tr -s ' '
if [[ -f $CG/cpu.max ]]; then
  echo "$P cgroup=v2 cpu.max=$(cat $CG/cpu.max) cpu.weight=$(cat $CG/cpu.weight 2>/dev/null) cpuset.effective=$(cat $CG/cpuset.cpus.effective 2>/dev/null) memory.max=$(cat $CG/memory.max 2>/dev/null)"
else
  for d in $CG/cpu $CG/cpu,cpuacct; do [[ -d $d ]] && echo "$P cgroup=v1 dir=$d cfs_quota_us=$(cat $d/cpu.cfs_quota_us 2>/dev/null) cfs_period_us=$(cat $d/cpu.cfs_period_us 2>/dev/null) cpu.shares=$(cat $d/cpu.shares 2>/dev/null)" && break; done
  echo "$P cpuset=$(cat $CG/cpuset/cpuset.effective_cpus $CG/cpuset/cpuset.cpus 2>/dev/null | head -1) memory.limit=$(cat $CG/memory/memory.limit_in_bytes 2>/dev/null)"
fi
echo "$P self_cgroup=$(tr '\n' ' ' < /proc/self/cgroup)"
f=$(cpustat_file); [[ -n $f ]] && echo "$P cpu.stat $(tr '\n' ' ' < "$f")"
echo "$P psi_cg=$(psi_cg) psi_node=$(psi_node)"
echo "$P loadavg=$(cat /proc/loadavg)"
read -r a0 b0 c0 d0 < <(procstat); sleep 2; read -r a1 b1 c1 d1 < <(procstat)
echo "$P node_busy_2s_pct=$(awk -v a=$((a1 - a0)) -v b=$((b1 - b0 + c1 - c0)) 'BEGIN{printf "%.1f", 100*(a-b)/a}') steal_2s_pct=$(awk -v a=$((a1 - a0)) -v b=$((d1 - d0)) 'BEGIN{printf "%.1f", 100*b/a}')"
free -m 2>/dev/null | sed "s/^/$P free /" | tr -s ' '
read -r u _ _ p n th < <(cpustat_vals)
echo "$(date +%s) $u $p $n $th" > "$STATE/$MODE"
if [[ $MODE != start && -f $STATE/start ]]; then
  read -r e0 u0 p0 n0 th0 < "$STATE/start"; read -r e1 u1 p1 n1 th1 < "$STATE/$MODE"
  if [[ $u1 != NA ]]; then
    echo "$P delta_from_start wall_s=$((e1 - e0)) cpu_s=$(( (u1 - u0) / 1000000 )) avg_cores=$(awk -v u=$((u1 - u0)) -v w=$((e1 - e0)) 'BEGIN{printf "%.2f", u/1e6/(w?w:1)}') nr_periods_d=$((p1 - p0)) nr_throttled_d=$((n1 - n0)) throttled_s=$(awk -v t=$((th1 - th0)) 'BEGIN{printf "%.1f", t/1e6}')"
  else
    echo "$P delta_from_start wall_s=$((e1 - e0)) cpu.stat=NA"
  fi
fi
# Fixed CPU work: one process, then nproc processes at once (seconds of wall time; lower = faster).
python3 - "$P" <<'EOF' || true
import multiprocessing as mp, os, sys, time
def work(_=None):
    t = time.perf_counter(); x = 0
    for i in range(15_000_000): x += i * i % 7
    return time.perf_counter() - t
t1 = work()
n = os.cpu_count() or 1
t = time.perf_counter()
with mp.Pool(n) as p: r = p.map(work, range(n))
print(f"{sys.argv[1]} bench single_s={t1:.3f} parallel_n={n} parallel_wall_s={time.perf_counter() - t:.3f} parallel_task_max_s={max(r):.3f} parallel_task_med_s={sorted(r)[n // 2]:.3f}")
EOF
exit 0
