#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# fleet/gate.sh: read-only idle gate of this host, run by run.sh --mode before every phase. Nothing here opens a device.
# Refuses when: a process outside this run holds /dev/tenstorrent or owns a chip's UMD CHIP_IN_USE lock in /dev/shm
# (tt-telemetry and its tt_telemetry_server are ignored; the authoritative test); a local device workload of anyone else runs, matched on its
# executable only (wl below); a wan container runs, or a container that maps /dev/tenstorrent has a process with a device
# node open (a container that only maps it, with idle or no device-holding processes, gets a WARNING and does not count);
# a wan pod is scheduled on
# this node (the kubelet's /var/log/pods/wan_* dirs, where the host has them; local reads only, no kubectl).
#   gate.sh [OWN_PGID [RUN_PID]]   prints "gate ok" or "gate busy: <why>"; exit 0 ok, 10 device held or another broker job, 11 other run/container/pod
# In a tt-device-mcp broker job: the job's own processes never count, another broker job (a ttdev-job-*.scope
#   that is not this one) is busy (10), and a workload matched by name only is CPU load (holds no device) printed to stderr
#   as "host load: ..." and not counted. Outside a broker job the name check below still refuses (11).
# env: FLEET_GATE_DEVICE_ONLY=1 (opt-in, default unset: skips the host-wide workload-name check below, so only
#   device holders and the tray-test run lock count; for hosts where a device broker serializes the device. Never set on quad hosts.)
# env: TRAY_REPRO_SYS, FLEET_PROC (test hook, default /proc), FLEET_DEVD (default /dev/tenstorrent), FLEET_SHM (default /dev/shm), FLEET_PODLOG (default /var/log/pods),
#   TRAY_RUN_LOCK (run.sh's host-wide run lock, default /run/lock/tray-test-run.lock), TRAY_RUN_LOCK_HELD (set by the run.sh that holds it),
#   FLEET_SELF_CGROUP (test hook: this gate's cgroup file, default /proc/$$/cgroup)
set -u
own=${1:-}; run=${2:-}; PROC=${FLEET_PROC:-/proc}; DEVD=${FLEET_DEVD:-/dev/tenstorrent}; SHM=${FLEET_SHM:-/dev/shm}
. "$(dirname "$(readlink -f "$0")")/../lib/sudo_l.sh"   # never sudo -n true; find, crictl and docker take varying arguments
S=(); [ "$(id -u)" = 0 ] || { sudo_all && S=(sudo -n); }
jobof() { sed -n 's#.*/ttdev-job-\([^/]*\)\.scope.*#\1#p' "$1" 2>/dev/null | head -1; }   # $1 a cgroup file: its broker job id
myjob=$(jobof "${FLEET_SELF_CGROUP:-/proc/$$/cgroup}")   # the broker job running this test, or empty
ours() {  # $1 pid: in this run's process group or broker job, or this gate itself
  local g; [ "$1" = $$ ] && return 0; [ -n "$myjob" ] && [ "$(jobof "$PROC/$1/cgroup")" = "$myjob" ] && return 0; [ -z "$own" ] && return 1
  g=$(ps -o pgid= -p "$1" 2>/dev/null | tr -d ' '); [ "$g" = "$own" ]
}
. "$(dirname "$(readlink -f "$0")")/agentp.sh"
why=""; rc=0
# when the only holder is tt-fabric-manager-agent, pass at once with a WARNING (default
# wait 0 s; FLEET_AGENT_WAIT_S>0 polls that long first). Any other holder still refuses (10). The agent is never signalled. env: FLEET_AGENT_WAIT_S (0), FLEET_AGENT_IVL (15)
AW=${FLEET_AGENT_WAIT_S:-0}; AI=${FLEET_AGENT_IVL:-15}; t0=$SECONDS; warn=""
# A job of this run
# (setsid, so outside OWN_PGID) can still be closing the device for a moment after fl_watch saw its leader exit. A holder
# that descends from RUN_PID (the run.sh, 2nd argument) is this run's own: wait up to FLEET_GATE_OWN_WAIT_S (10) s for it to
# go, then count it as busy as before. A holder pid gone before it is named holds nothing: skipped.
OW=${FLEET_GATE_OWN_WAIT_S:-10}
mine() {  # $1 pid: a descendant of RUN_PID, by the ppid chain in $PROC/<pid>/stat
  local q=$1 h=0 s; [ -n "$run" ] || return 1
  while [[ $q =~ ^[0-9]+$ ]] && [ "$q" -gt 1 ] && [ $h -lt 64 ]; do
    [ "$q" = "$run" ] && return 0
    s=$(cat "$PROC/$q/stat" 2>/dev/null) || return 1; s=${s##*) }; read -r _ q _ <<< "$s"; h=$((h + 1))
  done; return 1
}
lockowners() {  # pids owning a chip's UMD lock, $SHM/*CHIP_IN_USE* (trays note 86589dd8893b7e39): a robust pthread mutex
  # at offset 0, not an flock, read as load_test.sh's lockowners: int __lock @0 (owner TID | OWNER_DIED 0x40000000), int
  # __owner @8 (owner TID; 0x7fffffff/0x7ffffffe: use __lock's TID); the TID's Tgid is the process. Unowned or stale: none.
  local f w lk tid
  for f in "$SHM"/*CHIP_IN_USE*; do [ -f "$f" ] || continue
    read -ra w < <(od -An -v -t u4 -N 12 "$f" 2>/dev/null); [ ${#w[@]} -ge 3 ] || continue
    lk=${w[0]}; tid=$(( w[2] & 0x3fffffff ))
    case ${w[2]} in 2147483647|2147483646) tid=$(( lk & 0x3fffffff )) ;; esac
    [ $tid = 0 ] && continue
    [ $(( lk >> 30 & 1 )) = 1 ] && [ $(( lk & 0x3fffffff )) = 0 ] && continue
    awk '$1 == "Tgid:" { print $2; exit }' "$PROC/$tid/status" 2>/dev/null
  done | sort -u
}
while :; do
  why=""; rc=0; other=0; agent=0; own_left=""
  lo=" $(lockowners | paste -sd' ') "
  fd=" $("${S[@]}" find "$PROC"/[0-9]*/fd -lname "$DEVD/*" 2>/dev/null | sed -n 's#.*/\([0-9][0-9]*\)/fd/.*#\1#p' | sort -u | paste -sd' ') "
  for p in $(tr ' ' '\n' <<< "$fd $lo" | grep . | sort -u); do
    c=$(cat "$PROC/$p/comm" 2>/dev/null); case $c in tt-telemetry*|tt_telemetry*) continue ;; esac; ours "$p" && continue   # tt-telemetry.service runs /usr/bin/tt_telemetry_server (comm tt_telemetry_se)
    h=""; [[ $fd == *" $p "* ]] && h="holds $DEVD"; [[ $lo == *" $p "* ]] && h="${h:+$h, }owns a $SHM CHIP_IN_USE lock"
    if mine "$p"; then own_left="$own_left; pid $p ($c, this run's own job, still open after ${OW}s) $h"; continue; fi
    [ -e "$PROC/$p" ] || continue   # gone since the scan: it holds nothing now
    if agentp "$p" "$c"; then agent=1; else other=1; fi
    why="$why; pid $p ($c, $(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')) $h"; rc=10
  done
  if [ -n "$own_left" ]; then   # this run's own job still closing: wait for it alone, else it counts as busy
    [ $other = 0 ] && [ $agent = 0 ] && [ $((SECONDS - t0)) -lt "$OW" ] && { sleep 1; continue; }
    why="$why$own_left"; rc=10; other=1
  fi
  [ $rc = 10 ] && [ $other = 0 ] && [ $agent = 1 ] || break
  if [ $((SECONDS - t0)) -ge "$AW" ]; then warn="WARNING: only tt-fabric-manager-agent held $DEVD (waited ${AW}s); going ahead"; why=""; rc=0; break; fi
  sleep "$AI"
done
[ -z "$warn" ] || echo "$warn" >&2
# wl: the executable a process runs, matched by name: argv[0]'s basename, or for
# an interpreter its script or -m module. Never a substring of the command line: a shell's -c text, ssh/quad-ssh to
# another host, git, ttp sandbox or an editor that only mention a tray-test or H3 path do not touch the local chips.
# pytest is not a name here: ttp checks and sandboxed test suites run it with no chip; a
# pytest on the device is busy by its /dev/tenstorrent fd or CHIP_IN_USE lock above.
wl() {  # $@ argv: 0 when it is a local device workload
  local a=${1##*/}
  case $a in python|python[0-9]*|bash|sh|dash|zsh|perl)
    a=""; shift
    while [ $# -gt 0 ]; do
      if [[ $1 =~ ^-[A-Za-z]*c[A-Za-z]*$ ]]; then return 1           # -c text: what it starts shows up as its own process
      elif [ "$1" = -m ]; then a=${2:-}; break
      elif [[ $1 =~ ^-[XWo]$ ]]; then shift 2
      elif [[ $1 = -* ]]; then shift
      else a=${1##*/}; break; fi
    done ;;
  esac
  case $a in
    tt-smi|tt_smi|tt-triage|tt-triage.py|triage.py|tt-exalens|exalens|tt-run|unit_tests*) return 0 ;;
    mpirun|mpiexec|prterun|prted|orted) return 0 ;;   # prted/orted: the OpenMPI daemon on the other hosts of a run
    run_repro.sh|run-thorough.sh|run-mid*|ub0_repro.py|load_test*|fabric_repro.py|h3shape_repro.py|ccl_load.py) return 0 ;;
    run_workers*|run_server*|uvicorn|h3_server*) return 0 ;;
    # the H3 repro kit (/data_models/h3/repro/quad*-dbg): its link check runs the cluster recover.sh (tt-smi
    # -glx_reset on all 4 hosts, then 60 s sleeps that hold no chip); its watchdog, triage and soak drive a live run
    *_link_check.sh|recover.sh|run_cluster_validation*|watchdog.sh|triage_all.sh|triage_host.sh|soak.sh|stop_dbg.sh) return 0 ;;
    on_timeout_in_container.sh) return 0 ;;
  esac
  return 1
}
# our deviceless CPU test shards (TRAY_TEST_DEVICELESS_SHARD=1 in their environ) are not a workload by
# name; a shard that holds a device is still caught by the holder check above (10)
shard() { { tr '\0' '\n' < "$PROC/$1/environ"; } 2>/dev/null | grep -qx 'TRAY_TEST_DEVICELESS_SHARD=1'; }   # an unreadable environ (job 970: Permission denied) prints nothing
[ "${FLEET_GATE_DEVICE_ONLY:-0}" = 1 ] || while read -r pid pgid user args; do
  [ "$pid" = $$ ] && continue; [ -n "$own" ] && [ "$pgid" = "$own" ] && continue
  read -ra av <<< "$args"; [ ${#av[@]} -gt 0 ] && wl "${av[@]}" || continue
  shard "$pid" && continue
  if [ -n "$myjob" ]; then ours "$pid" || echo "host load: pid $pid ($user): ${args:0:100} (holds no device; not counted in broker job $myjob)" >&2; continue; fi
  why="$why; pid $pid ($user): ${args:0:100}"; [ $rc = 0 ] && rc=11
done < <(ps -eo pid=,pgid=,user=,args=)
# another tt-device-mcp broker job on this host is busy whether or not it holds a device right now
while read -r j p; do
  why="$why; broker job $j runs (pid $p ($(cat "$PROC/$p/comm" 2>/dev/null), $(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')))"; rc=10
done < <(grep -Ho 'ttdev-job-[^/]*\.scope' "$PROC"/[0-9]*/cgroup 2>/dev/null \
  | sed -n 's#^.*/\([0-9][0-9]*\)/cgroup:ttdev-job-\(.*\)\.scope$#\2 \1#p' | awk -v m="$myjob" '$1 != m && !s[$1]++')
# a tray-test run.sh is busy by its host-wide run lock, never by the name run.sh: the stubbed run.sh copies
# of tray-test's own test suites lock their temp dir (TRAY_RUN_LOCK) and touch no chips. Probed with a shared flock -n on a
# read-only open, so a probe never blocks a run; a lock file that cannot be probed counts as held.
LK=${TRAY_RUN_LOCK:-/run/lock/tray-test-run.lock}
if [ -e "$LK" ] && [ "${TRAY_RUN_LOCK_HELD:-}" != "$LK" ] && ! flock -n -s 9 9< "$LK" 2> /dev/null; then
  h=$("${S[@]}" find /proc/[0-9]*/fd -lname "$(readlink -f "$LK")" 2>/dev/null | sed -n 's#^/proc/\([0-9]*\)/fd/.*#\1#p' | sort -un | head -3 \
    | while read -r p; do echo "pid $p ($(ps -o user=,args= -p "$p" 2>/dev/null | cut -c1-80))"; done | paste -sd, -)
  why="$why; a tray-test run holds $LK: ${h:-holder not visible}"; [ $rc = 0 ] && rc=11
fi
if command -v crictl > /dev/null; then
  n=$("${S[@]}" crictl ps --state running --label io.kubernetes.pod.namespace=wan -q 2>/dev/null | grep -c .)
  [ "$n" -gt 0 ] && { why="$why; $n wan container(s) running (crictl)"; [ $rc = 0 ] && rc=11; }
fi
if command -v docker > /dev/null; then
  for id in $("${S[@]}" docker ps -q 2>/dev/null); do
    "${S[@]}" docker inspect -f '{{.HostConfig.Devices}} {{.HostConfig.Binds}}' "$id" 2>/dev/null | grep -qF -e tenstorrent -e "$DEVD" || continue
    # a container that only maps the device is not busy, the same rule as for pods. It is
    # busy when a process in it has a device node open, or when its processes' fds cannot be read and one of them is
    # not idle (sleep, pause, init, tail -f /dev/null), or when its process list cannot be read at all.
    cb=""; cu=""; n=0; ch=0
    while read -r cp ca; do
      [[ $cp =~ ^[0-9]+$ ]] || continue; n=$((n + 1)); [ -n "$cu" ] || cu=$(stat -c %U "$PROC/$cp" 2>/dev/null)
      o=$("${S[@]}" find "$PROC/$cp/fd" -lname "$DEVD/*" 2>/dev/null); frc=$?
      if [ -n "$o" ]; then cb="$cb, pid $cp holds $DEVD"; ch=1
      elif [ $frc != 0 ] && ! grep -qE '^(sleep( infinity| [0-9.]+[smhd]?)?|/pause|/sbin/docker-init( .*)?|/dev/init( .*)?|tail -f /dev/null)$' <<< "$ca"; then
        cb="$cb, pid $cp ($(cut -c1-40 <<< "$ca")) fds unreadable"
      fi
    done < <("${S[@]}" docker top "$id" -eo pid,args 2>/dev/null | sed 1d | sed -E 's/ +$//')
    [ $n = 0 ] && cb=", process list unreadable"
    if [ -z "$cb" ]; then
      img=$("${S[@]}" docker inspect -f '{{.Config.Image}}' "$id" 2>/dev/null)
      echo "WARNING: container $id (image ${img:-?}, user ${cu:-?}) maps $DEVD but no process in it has a device open; not counted as busy" >&2
      continue
    fi
    # inside a Slurm job of our own, a container left by someone else whose fds we cannot
    # read is logged, not busy, when no process in it holds the device and the holder check above saw no other holder.
    if [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_USER:-$(id -un)}" = "$(id -un)" ] && [ $ch = 0 ] && [ $rc != 10 ]; then
      img=$("${S[@]}" docker inspect -f '{{.Config.Image}}' "$id" 2>/dev/null)
      echo "WARNING: container $id (image ${img:-?}, user ${cu:-?}) maps $DEVD${cb}; inside our Slurm job ${SLURM_JOB_ID}, no open device fd seen, not counted as busy" >&2
      continue
    fi
    why="$why; container $id maps $DEVD${cb}"; [ $rc = 0 ] && rc=11
  done
fi
PL=${FLEET_PODLOG:-/var/log/pods}   # wan pods on this node from the kubelet's own pod log dirs (wan_<pod>_<uid>): local reads only, no kubectl
if [ -d "$PL" ]; then
  w=$(ls -1 "$PL" 2>/dev/null | grep '^wan_' | cut -d_ -f2 | grep -v '^tt-fabric-manager-agent' | sort -u | paste -sd' ' -)
  [ -n "$w" ] && { why="$why; wan pod(s) on this node (kubelet pod logs $PL): $w"; [ $rc = 0 ] && rc=11; }
fi
[ $rc = 0 ] && echo "gate ok" || echo "gate busy: ${why#; }"
exit $rc
