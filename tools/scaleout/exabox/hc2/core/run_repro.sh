#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
# run_repro.sh — one command from a cold Blackhole galaxy to a verdict and an evidence tarball.
#
#   run_repro.sh [--dry-run] [--out DIR] [--wait-quiet-min N] [--kcache DIR|off]
#                [-- ub0_repro.py args...]
#
# 1. Checks the environment: TT_METAL_HOME (tt-metal ltx-rt b9f8587ce6c is the validated commit; others warn),
#    the python with ttnn and torch ($UB0_PYTHON, else $TT_METAL_HOME/python_env/bin/python, else python3),
#    ipmitool and BMC access.
# 2. Runs ub0_repro.py with its documented defaults (ub0: 4x8 ring, 60 s open idle unless --idle-s, 30 s burst), which
#    does the preflight (all 32 chips / 4 trays, else exit 2 before any load) and the cold-state check itself.
# 3. Right after it ends, captures in this order (the live signals first): tray/PDB CPLD registers, BMC SDR (tray
#    power, PSU, retimer temps), BMC SEL, lspci tray state, dmesg excerpt, BMC mc info. Without ipmitool or BMC
#    access it writes the exact commands to run into capture_commands.txt and prints them.
# 4. Packs the log, telemetry CSV, result JSON and captures into <out>/ub0_<host>_<UTC>.tar.gz and ends on one line:
#      VERDICT: NO TRIP | VERDICT: TRIP chip N tray T at +Xs | VERDICT: PREFLIGHT REFUSED (...) | VERDICT: ERROR (...)
#      | VERDICT: CCL HANG: chip unknown (32/32 still on PCIe), op ... (--load allgather only, exit 4)
#    A TRIP line ends " loss=DROP", " loss=CHIP_HANG" or " loss=UNKNOWN" (ub0_repro.py's loss_class: CHIP_HANG = one
#    chip fell off PCIe with its tray powered, DROP = a tray lost, UNKNOWN = not proven either way; a missing class
#    prints UNKNOWN, never DROP), then " transient=N,.." if chips answered again after
#    --settle-s. Both are a loss (exit 3).
#    A chip lost after the load, in ub0_repro.py's GO_LONG_IDLE window (result long_idle_chip_missing), keeps the exit
#    code 0 but ends "VERDICT: NO TRIP, POST-RUN CHIP LOSS chip N tray T ...": count it apart, neither trip nor clean.
#
# --dry-run: environment check and ub0_repro.py --dry-run (sysfs reads only: preflight, cold state, the plan); no
# device open, no ipmitool. Ends on "VERDICT: DRY RUN ..." with exit 0 if a real run would start, else 1 or 2.
#
# --kcache DIR|off: an opt-in shared tt-metal JIT cache (#906), for start-up time only: a pre-warmed cache spares the
# run its firmware/kernel compile at mesh open. It is NOT a fix for the chip-0 mesh_open hang: the cold-cache theory
# (#903) was ruled out 10-16 01:37 PDT (a warm-cache root run on a dev host at 01:16 PDT hung the same way,
# and root writes no cache entry). Default off: no cache is set for any user, root included, and tt-metal keeps its
# own default ($HOME/.cache/tt-metal-cache, or $TT_METAL_CACHE if the caller set it). --kcache DIR uses DIR for any
# user. With a cache set it exports TT_METAL_CACHE=DIR and, before
# ub0_repro.py opens the mesh, prewarms it OFF-DEVICE (no /dev/tenstorrent open; strace proof in #906):
#   a. seed: each build key in the invoking user's warm cache (~$SUDO_USER/.cache/tt-metal-cache/<key>/, or
#      $UB0_KCACHE_SEED) that DIR lacks is copied whole to DIR/tt-metal-cache<key>/ (tt-metal's layout for a
#      TT_METAL_CACHE value); keys DIR has are left as they are;
#   b. $TT_METAL_HOME/build_Release/tools/kernel_prewarm DIR: compiles DIR's kernel_prewarm.manifest, if any (a no-op
#      without one; runs record one only with TT_METAL_KERNEL_PREWARM=1, which this script does not set).
# Files a root run leaves in DIR are given back to DIR's owner (before the prewarm and after the run), so root and
# non-root runs share it. A failed prewarm step is logged and the run goes on. Only the cache moves: load, clock,
# throttle and stagger are untouched. --dry-run logs the export and the prewarm steps without running them.
#
# Run as root (sudo run_repro.sh, SysEng's command line), only the device process drops root (#902:
#   root as the device process triggers the chip-0 mesh_open CHIP_HANG): ub0_repro.py (every --load, matmul too) runs
#   under setpriv --reuid=<user> --regid=<device group> --groups <device group>,<the user's groups>, capabilities
#   gone (CapEff 0), in its own cwd (<out>/devhome, made with install -d -o <user> -g <group>) with HOME = <user>'s
#   own home, so tt-metal uses <user>'s existing JIT cache (~/.cache/tt-metal-cache, or $TT_METAL_CACHE if the
#   caller set it; the #903 cold-cache suspect was ruled out earlier, so the cache is not part of the workaround),
#   and only these
#   variables: HOME USER LOGNAME PATH LANG LC_* TZ and every TT_* UB0_* PYTHON* VIRTUAL_ENV LD_LIBRARY_PATH ARCH_NAME
#   IPMI_PASSWORD (no secret goes into argv).
#   <user> = $SUDO_USER (if set and not root), else $TRAY_REPRO_DEVICE_USER, else the owner of this script's tree
#   (e.g. the owner of /opt/tray-repro); <group> = `stat -c %G` of /dev/tenstorrent/<n>
#   (ttdev on the blx boxes), read at run time. Before the run, as <user>: every /dev/tenstorrent/<n>, the 1G
#   hugepage mount (/dev/hugepages-1G) and its files, /dev/shm and its CHIP_IN_USE locks must be readable and
#   writable (access checks, no open), else VERDICT: ERROR, exit 1: it never falls back to root. ub0_repro.py's
#   CPLD/SEL/lspci reads go through `sudo -n` when <user> has passwordless sudo, else a WARNING says that loss
#   evidence is missing. The wrapper's captures (CPLD, SDR, SEL, lspci, dmesg, mc info) stay root. The state file
#   ($UB0_STATE or --state-file, default /var/tmp/ub0_repro.state.json) must be readable, writable and owned by <user>
#   and its dir writable (ub0_repro.py replaces it by a rename), else the same refusal: a file another user owns is
#   lent to <user> for the run (chown <user>:<group>, u+rw) and its owner and mode are given back at exit.
#   The log line `dev_proc=uid:<n>(<user>) gid:<g> capeff=<hex|0> jit_cache=<dir>(warm|cold)` says who ran the device
#   process and with which cache (warm: the cache dir has entries; cold: missing or empty).
#   TRAY_REPRO_DEVICE_AS_ROOT=1 keeps the old root device process, for A/B runs only (off by default). A non-root
#   caller (a broker job, an sbatch job) is unchanged: no setpriv, label `... caller`.
#   --device-ident prints that label (after the same checks; exit 2 if they fail) and --device-exec -- CMD... runs
#   CMD as the device process (stdin kept): load_test.sh uses them for its own tt_umd clock and idle steps.
# env: TT_METAL_HOME (required for a real run); BMC_HOST, BMC_USER (default root), BMC_PASS for out-of-band BMC
#      reads (in-band ipmitool is used for the CPLD reads and when BMC_HOST is unset); UB0_PYTHON; UB0_OUT; IPMITOOL;
#      TRAY_REPRO_DEVICE_AS_ROOT, TRAY_REPRO_DEVICE_USER, TRAY_REPRO_DEV_CWD (the device process cwd, default
#      <out>/devhome). TRAY_REPRO_TEST_EUID, TRAY_REPRO_DEV_ROOT, TRAY_REPRO_HUGE_DIRS and TRAY_REPRO_SHM are test hooks.
# exit: as ub0_repro.py: 0 NO TRIP, 1 error (environment, script, timeout, interrupted), 2 PREFLIGHT REFUSED, 3 TRIP,
#       4 CCL HANG.
# Kills only what it started, by pid. Read-only towards the hardware apart from the test itself: no reset, no rescan.
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
EXPECT_COMMIT=b9f8587ce6c681f380e341753c6f0ea7f5ee441d
dry=0; out=${UB0_OUT:-$PWD/ub0_evidence}; quiet=""; pass=(); kcache=off; mode=run
usage() { sed -n '2,/^set -u/p' "$0" | sed '$d; s/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) dry=1 ;;
    --device-ident) mode=ident ;;
    --device-exec) mode=exec ;;
    --out) out=${2:?--out needs a directory}; shift ;;
    --wait-quiet-min) quiet=${2:?--wait-quiet-min needs minutes}; shift   # minutes, fractions too (load_test.sh: Q/2)
      [[ $quiet =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "--wait-quiet-min needs minutes, e.g. 1 or 0.5 (got '$quiet')" >&2; exit 1; } ;;
    --kcache) kcache=${2:?--kcache needs a directory or off}; shift
      [ "$kcache" = auto ] && { echo "--kcache auto was removed (#927: the cold cache is not the hang): pass DIR or off" >&2; exit 1; } ;;
    -h|--help) usage; exit 0 ;;
    --) shift; pass+=("$@"); break ;;
    *) pass+=("$1") ;;
  esac
  shift
done

# ---------------------------------------------------------------- 0. the device process identity (#902)
# dev_setup HOME: sets DROP (the setpriv argv, empty when no drop), DEV_PROC (the label), DEV_ERR (why the drop cannot
# run; never a fallback to root), DEV_SUDO ("sudo -n" if the device user may sudo, for ub0_repro.py's ipmitool/lspci)
dev_setup() {
  local cwd=$1 euid u uid g gid glist n bad=() f d id cdir
  DROP=(); DEV_ERR=""; DEV_SUDO=""; DEV_LSUDO=""; DEV_CWD=$cwd; DEV_HOME=""; DEV_PROC="refused"
  euid=${TRAY_REPRO_TEST_EUID:-$(id -u)}
  if [ "$euid" != 0 ]; then DEV_PROC="uid:$(id -u)($(id -un)) gid:$(id -g) caller"; return 0; fi
  if [ "${TRAY_REPRO_DEVICE_AS_ROOT:-0}" = 1 ]; then
    DEV_PROC="uid:0(root) gid:0 capeff=$(awk '/^CapEff/ {print $2}' /proc/self/status) as-root(TRAY_REPRO_DEVICE_AS_ROOT=1)"; return 0
  fi
  u=${SUDO_USER:-}; [ "$u" = root ] && u=""
  u=${u:-${TRAY_REPRO_DEVICE_USER:-$(stat -c %U "$HERE" 2>/dev/null)}}
  uid=$(getent passwd "$u" | cut -d: -f3)
  if [ -z "$u" ] || [ -z "$uid" ] || [ "$uid" = 0 ]; then
    DEV_ERR="no non-root user for the device process (SUDO_USER, TRAY_REPRO_DEVICE_USER and the tree owner give '${u:-none}'); set TRAY_REPRO_DEVICE_USER, or TRAY_REPRO_DEVICE_AS_ROOT=1 for an A/B root run"; return 1
  fi
  local nodes=() dr=${TRAY_REPRO_DEV_ROOT:-/dev/tenstorrent}
  for n in "$dr"/[0-9]*; do [ -e "$n" ] && nodes+=("$n"); done
  [ ${#nodes[@]} -gt 0 ] || { DEV_ERR="no $dr/<n> device node to read the device group from"; return 1; }
  g=$(stat -c %G "${nodes[0]}"); gid=$(stat -c %g "${nodes[0]}")
  for n in "${nodes[@]}"; do [ "$(stat -c %g "$n")" = "$gid" ] || { DEV_ERR="$dr nodes have mixed groups ($n is $(stat -c %G "$n"), ${nodes[0]} is $g)"; return 1; }; done
  glist=$(id -G "$u" | tr ' ' '\n' | grep -vx "$gid" | tr '\n' , ); glist=$gid${glist:+,${glist%,}}
  DROP=(setpriv --reuid="$uid" --regid="$gid" --groups "$glist" --inh-caps=-all)
  DEV_HOME=$(getent passwd "$u" | cut -d: -f6)
  [ -d "$DEV_HOME" ] || { DEV_ERR="$u's home '$DEV_HOME' is no directory (the device process uses its warm JIT cache there)"; return 1; }
  install -d -o "$uid" -g "$gid" -m 0750 "$cwd" || { DEV_ERR="cannot make the device process cwd $cwd for $u"; return 1; }
  # the cold-state file (#946): ub0_repro.py rewrites it as the device user (a tmp file in its dir, then a rename over
  # it), so the user must own it (a rename over another user's file in sticky /var/tmp fails) and write its dir. An
  # existing file owned by someone else is lent to the user (chown uid:gid, u+rw) and given back, owner and mode, by
  # state_restore; a missing one is made by the device process itself. --device-exec never touches it.
  f=$(state_path)
  if [ -n "$f" ] && [ "$mode" != exec ]; then
    if [ -e "$f" ] && [ ! -L "$f" ] && [ "$(stat -c %u "$f")" != "$uid" ]; then
      STATE_OWN=$(stat -c %u:%g "$f"); STATE_MODE=$(stat -c %a "$f")
      chown "$uid:$gid" "$f" && STATE_F=$f && chmod u+rw "$f" && echo "dev_setup: lent the cold-state file $f ($STATE_OWN mode $STATE_MODE) to $u:$g; given back at exit" >&2
    fi
    if [ -L "$f" ]; then bad+=("$f (cold-state file is a symlink)")
    elif [ -e "$f" ]; then
      { (dev_exec test -r "$f" -a -w "$f") && [ "$(stat -c %u "$f")" = "$uid" ]; } || bad+=("$f (cold-state file, owner $(stat -c %U "$f"))")
    fi
    d=$(dirname "$f"); (dev_exec test -w "$d" -a -x "$d") || bad+=("$d (cold-state file's dir)")
  fi
  # the checks run AS the device user: access(2) only (test -r/-w), nothing is opened
  for n in "${nodes[@]}"; do (dev_exec test -r "$n" -a -w "$n") || bad+=("$n"); done
  for d in ${TRAY_REPRO_HUGE_DIRS:-/dev/hugepages-1G}; do
    [ -d "$d" ] || continue
    (dev_exec test -w "$d" -a -x "$d") || bad+=("$d (hugepages)")
    for f in "$d"/*; do [ -e "$f" ] || continue; (dev_exec test -r "$f" -a -w "$f") || bad+=("$f"); done
  done
  d=${TRAY_REPRO_SHM:-/dev/shm}; (dev_exec test -w "$d" -a -x "$d") || bad+=("$d")
  for f in "$d"/*CHIP_IN_USE*; do [ -e "$f" ] || continue; (dev_exec test -r "$f" -a -w "$f") || bad+=("$f"); done
  (dev_exec test -w "$cwd") || bad+=("$cwd")
  cdir=${TT_METAL_CACHE:-$DEV_HOME/.cache/tt-metal-cache}                 # tt-metal's default: $HOME/.cache/tt-metal-cache
  if [ -e "$cdir" ]; then (dev_exec test -w "$cdir") || bad+=("$cdir (JIT cache)"); else (dev_exec test -w "$DEV_HOME") || bad+=("$DEV_HOME (home)"); fi
  if [ ${#bad[@]} -gt 0 ]; then
    DEV_ERR="device user $u (uid $uid, group $g) cannot read/write: ${bad[*]}"; return 1
  fi
  # the label is read from the dropped process itself; real, effective, saved and fs uid/gid must all be the user's
  id=$(dev_exec sh -c 'echo "uid:$(id -u)($(id -un)) gid:$(id -g) capeff=$(awk "/^CapEff/ {print \$2}" /proc/self/status)"; awk "/^[UG]id:/ {print \$2, \$3, \$4, \$5}" /proc/self/status' 2>&1) || { DEV_ERR="setpriv to $u failed: $id"; return 1; }
  DEV_PROC=$(head -1 <<< "$id" | sed -E 's/capeff=0+$/capeff=0/')
  case $DEV_PROC in "uid:$uid("*" gid:$gid capeff=0") DEV_PROC+=" jit_cache=$cdir($([ -n "$(ls -A "$cdir" 2>/dev/null)" ] && echo warm || echo cold))" ;; *) DEV_ERR="the device process is not $u/$g with no capabilities: $DEV_PROC"; return 1 ;; esac
  [ "$(sed -n 2,3p <<< "$id" | tr '\n' ' ')" = "$uid $uid $uid $uid $gid $gid $gid $gid " ] || { DEV_ERR="the device process keeps another uid/gid (saved or fs): $(sed -n 2,3p <<< "$id" | tr '\n' ' ')"; return 1; }
  DEV_SUDO=$( (dev_exec sudo -n -l "${IPMITOOL:-ipmitool}" sel elist last 40 < /dev/null) > /dev/null 2>&1 && echo "sudo -n")   # ub0_repro.py's SEL read; never sudo -n true
  DEV_LSUDO=$( (dev_exec sudo -n -l lspci -vvv -d 1e52: < /dev/null) > /dev/null 2>&1 && echo "sudo -n")
  return 0
}
# state_path: the cold-state file ub0_repro.py will use (its --state-file, else $UB0_STATE, else the default; '' = none)
state_path() {
  local i f=${UB0_STATE-/var/tmp/ub0_repro.state.json}
  for ((i = 0; i < ${#pass[@]}; i++)); do
    case ${pass[i]} in --state-file) f=${pass[i+1]-} ;; --state-file=*) f=${pass[i]#--state-file=} ;; esac
  done
  printf '%s' "$f"
}
# state_restore: gives a lent cold-state file back (owner and mode; the device process may have replaced the inode).
# Runs at exit; a SIGKILL skips it, and the file then stays the device user's until the next sudo run lends it again.
# chown -h and no chmod unless it is still a regular file: a device user that swapped the path for a symlink (a
# --state-file outside sticky /var/tmp) cannot make root chown or chmod the link's target.
STATE_F=""
state_restore() {
  [ -n "$STATE_F" ] || return 0
  if [ -L "$STATE_F" ] || { [ -e "$STATE_F" ] && [ ! -f "$STATE_F" ]; }; then
    chown -h "$STATE_OWN" "$STATE_F" 2>/dev/null
    echo "WARNING the cold-state file $STATE_F is no longer a regular file (symlink or other): owner given back with chown -h, mode $STATE_MODE not restored" >&2
  elif [ -e "$STATE_F" ] && ! { chown -h "$STATE_OWN" "$STATE_F" && chmod "$STATE_MODE" "$STATE_F"; }; then
    echo "WARNING could not give the cold-state file $STATE_F back to $STATE_OWN mode $STATE_MODE" >&2
  fi
  STATE_F=""
}
trap state_restore EXIT
# dev_exec CMD...: exec CMD as the device process. Call it only in a subshell (`( dev_exec ...)`, `$(dev_exec ...)`,
# `dev_exec ... &`): it replaces the environment and execs. A no-op wrapper when DROP is empty.
dev_exec() {
  [ ${#DROP[@]} = 0 ] && exec "$@"
  local v keep u=${DROP[1]#--reuid=}
  for v in $(compgen -e); do
    case $v in HOME|USER|LOGNAME) unset "$v" ;;
      PATH|LANG|LC_*|TZ|TT_*|UB0_*|PYTHON*|VIRTUAL_ENV|LD_LIBRARY_PATH|ARCH_NAME|IPMI_PASSWORD) ;;
      *) unset "$v" ;; esac
  done
  u=$(getent passwd "$u" | cut -d: -f1)
  export HOME=$DEV_HOME USER=$u LOGNAME=$u                              # TT_METAL_CACHE only if the caller or --kcache set it
  cd "$DEV_CWD" || exit 1
  exec "${DROP[@]}" "$@"
}
if [ $mode != run ]; then
  dev_setup "${TRAY_REPRO_DEV_CWD:-$out/devhome}" || { state_restore; echo "DEVICE PROCESS REFUSED: $DEV_ERR" >&2; exit 2; }
  state_restore                                                         # before the label: load_test.sh reads the last line
  [ $mode = ident ] && { echo "dev_proc=$DEV_PROC"; exit 0; }
  dev_exec "${pass[@]}"
fi

host=$(hostname -s); stamp=$(date -u +%Y%m%dT%H%M%SZ); name=ub0_${host}_${stamp}; ev=$out/$name
mkdir -p "$ev" || { echo "VERDICT: ERROR (cannot create $ev)"; exit 1; }
log() { echo "REPRO|$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)|$*" | tee -a "$ev/wrapper.log"; }
log "run_repro.sh host=$host out=$ev dry_run=$dry args=${pass[*]:-}"

# ---------------------------------------------------------------- 1. environment
envbad=()
if { [ -z "${TT_METAL_HOME:-}" ] || [ ! -d "${TT_METAL_HOME:-}" ]; } && [ -n "${UB0_PYTHON:-}" ] && pipv=$(timeout 30 "$UB0_PYTHON" -c "import importlib.metadata as m; print(m.version('ttnn'))" 2>/dev/null); then
  # this package: ttnn installed from pip (setup.sh), no tt-metal checkout
  unset TT_METAL_HOME; log "tt-metal: none; ttnn $pipv from pip ($UB0_PYTHON). Not the validated ltx-rt ${EXPECT_COMMIT:0:11} build"
elif [ -z "${TT_METAL_HOME:-}" ] || [ ! -d "${TT_METAL_HOME:-}" ]; then
  envbad+=("neither a tt-metal checkout (TT_METAL_HOME) nor a pip ttnn (UB0_PYTHON, see setup.sh)")
else
  commit=$(timeout 10 git -C "$TT_METAL_HOME" rev-parse HEAD 2>/dev/null || true)
  if [ "$commit" = "$EXPECT_COMMIT" ]; then log "tt-metal $commit (the validated ltx-rt commit)"
  else log "WARNING tt-metal commit ${commit:-unknown}, not the validated ltx-rt ${EXPECT_COMMIT:0:11}: a recent tt-metal very likely reproduces, but it is untested"; fi
  export PYTHONPATH=${PYTHONPATH:-$TT_METAL_HOME}
fi
py=${UB0_PYTHON:-}
[ -z "$py" ] && [ -x "${TT_METAL_HOME:-/nonexistent}/python_env/bin/python" ] && py=$TT_METAL_HOME/python_env/bin/python
[ -z "$py" ] && py=$(command -v python3 || true)
if [ -z "$py" ]; then
  envbad+=("no python found (set UB0_PYTHON or create tt-metal's python_env)")
else
  mods="ttnn torch"; case " ${pass[*]:-} " in *" --fmax"*) mods="$mods tt_umd" ;; esac
  # find_spec only: importing ttnn is slow, and the dry run must not touch the device
  miss=$(timeout 60 "$py" -c "import importlib.util, sys
print(' '.join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))" $mods 2>&1) || miss="python failed: $miss"
  if [ -n "$miss" ]; then envbad+=("$py cannot find: $miss (activate tt-metal's python_env)"); else log "python $py has $mods"; fi
fi
for e in "${envbad[@]}"; do log "ENV ERROR: $e"; done

# ---------------------------------------------------------------- 1b. shared JIT cache (--kcache, #906; opt-in, start-up time only)
kc_tool=${TT_METAL_HOME:-/nonexistent}/build_Release/tools/kernel_prewarm
kc_home=$HOME; [ -n "${SUDO_USER:-}" ] && kc_home=$(getent passwd "$SUDO_USER" | cut -d: -f6)
kc_seed=${UB0_KCACHE_SEED:-${kc_home:-/nonexistent}/.cache/tt-metal-cache}
kc_chown() {  # files a root run made in the cache go back to the cache dir's owner (non-root runs must write there too)
  local o; [ "$(id -u)" = 0 ] || return 0; o=$(stat -c %u:%g "$kcache" 2>/dev/null) || return 0
  [ "${o%%:*}" = 0 ] || chown -R --from=0 "$o" "$kcache" 2>/dev/null || log "WARNING kcache: chown back to $o failed"
}
kc_prewarm() {  # off-device: copies files and runs kernel_prewarm, which compiles on the host and opens no device
  local k n=0 tmp rc
  mkdir -p "$kcache" || { log "WARNING kcache: cannot create $kcache, the run compiles cold"; return 0; }
  [ "$(id -u)" = 0 ] && [ -n "${SUDO_USER:-}" ] && [ "$(stat -c %u "$kcache")" = 0 ] && chown "$SUDO_USER:" "$kcache"
  kc_chown
  for k in "$kc_seed"/[0-9]*/; do
    [ -d "$k" ] || continue; k=$(basename "$k")
    [ -e "$kcache/tt-metal-cache$k" ] && continue
    tmp=$kcache/.seed_$k.$$                               # copied whole, then renamed: a cut copy is never a cache
    if cp -a "$kc_seed/$k" "$tmp" && mv -T "$tmp" "$kcache/tt-metal-cache$k"; then n=$((n + 1))
    else rm -rf "$tmp"; log "WARNING kcache: seeding key $k from $kc_seed failed"; fi
  done
  kc_chown
  log "kcache: seeded $n build key(s) from $kc_seed; $(ls -d "$kcache"/tt-metal-cache[0-9]* 2>/dev/null | wc -l) in $kcache"
  if [ -x "$kc_tool" ]; then
    (cd "$TT_METAL_HOME" && timeout 900 "$kc_tool" "$kcache") > "$ev/kernel_prewarm.log" 2>&1; rc=$?
    kc_chown
    log "kcache: kernel_prewarm rc=$rc: $(grep -oE 'kernel prewarm \(offline\): [0-9]+ targets built.*|no manifest.*' "$ev/kernel_prewarm.log" | tail -1 | cut -c1-160)"
  else
    log "WARNING kcache: no $kc_tool, kernel prewarm skipped (seeded keys only)"
  fi
}
if [ "$kcache" = off ]; then
  log "kcache: off, tt-metal's own JIT cache (TT_METAL_CACHE=${TT_METAL_CACHE:-unset}, HOME=${HOME:-unset})"
else
  export TT_METAL_CACHE=$kcache
  log "kcache: export TT_METAL_CACHE=$kcache"
  if [ $dry = 1 ]; then
    log "kcache: prewarm (off-device; not run in a dry run): seed missing build keys from $kc_seed into $kcache/tt-metal-cache<key>/, then (cd $TT_METAL_HOME && $kc_tool $kcache)"
  elif [ ${#envbad[@]} = 0 ]; then
    kc_prewarm
  fi
fi

# exact sudo -n -l probes, never sudo -n true: ipmitool by its CPLD read, lspci and dmesg on their own
. "$(dirname "$(readlink -f "$0")")/lib/sudo_l.sh"
SUDO=$(sudo_pfx "${IPMITOOL:-ipmitool}" raw 0x06 0x52 0x11 0x80 0x01 0x1E); LSUDO=$(sudo_pfx lspci -vvv -d 1e52:); DSUDO=$(sudo_pfx dmesg -T)
ipmi_ok=0
IPMITOOL=${IPMITOOL:-ipmitool}                                       # the offline tests point this at a stub
if command -v "$IPMITOOL" > /dev/null; then
  if [ "$(id -u)" = 0 ] || [ -n "$SUDO" ] || [ -n "${BMC_HOST:-}" ]; then ipmi_ok=1; fi
fi
IB="timeout 15 $SUDO $IPMITOOL"                                        # in-band: CPLD raw reads
if [ -n "${BMC_HOST:-}" ]; then
  export IPMI_PASSWORD="${BMC_PASS:-}"                                # ipmitool -E reads it from the env, never argv (ps)
  OOB=(timeout 60 "$IPMITOOL" -C 17 -I lanplus -H "$BMC_HOST" -U "${BMC_USER:-root}" -E)
  OOB_SHOWN="ipmitool -C 17 -I lanplus -H $BMC_HOST -U ${BMC_USER:-root} -E (IPMI_PASSWORD from BMC_PASS)"
else
  # shellcheck disable=SC2206  # $SUDO is empty or "sudo -n"
  OOB=(timeout 60 $SUDO "$IPMITOOL"); OOB_SHOWN="sudo ipmitool"
fi
log "ipmitool: $([ $ipmi_ok = 1 ] && echo "usable (${OOB_SHOWN}; CPLD via in-band ${SUDO:+sudo }ipmitool)" || echo "not usable (missing, or no root/sudo and no BMC_HOST): capture commands will be printed")"

# the device process (#902): root drops to a non-root user for ub0_repro.py only; the captures below stay root
if dev_setup "${TRAY_REPRO_DEV_CWD:-$out/devhome}"; then
  log "dev_proc=$DEV_PROC${DROP[*]:+ (${DROP[*]}; HOME=$DEV_HOME, cwd=$DEV_CWD)}"
  if [ ${#DROP[@]} -gt 0 ]; then
    chown "${DROP[1]#--reuid=}:${DROP[2]#--regid=}" "$ev" || envbad+=("cannot chown $ev to the device user")
    [ -z "$DEV_SUDO" ] && log "WARNING the device user has no passwordless sudo: ub0_repro.py gets no in-band CPLD/SEL reads, so its loss class lacks CPLD evidence (the wrapper's captures still run as root)"
  fi
else
  envbad+=("device process: $DEV_ERR")
fi

# ---------------------------------------------------------------- helpers
SDR_RE='Power_UBB[0-3] |PSU|P12V|54V|TEMP_UBB[0-3]_RT0|TMAX_UBB'
TRAY_NIBBLE=([1]=0 [2]=4 [3]=c [4]=8)                               # PCI bus high nibble per tray
BUSES="0x0D 0x0F 0x11 0x13"                                           # tray CPLDs at 0x80; PDB CPLD bus 0x09 at 0x62
reg() { $IB raw 0x06 0x52 "$1" "$2" 0x01 "$3" 2>/dev/null | tr -d ' \n' || true; }
cpld_dump() {  # the plain registers of trayfault.py's full dump (SysEng's BH_s7t_cpld_dump reads); live, so first. Reads only
               # the debug registers need a CPLD index write first, so they are named, not read
  local b r d line
  echo "# $(date -u +%FT%T.%3NZ) tray CPLDs (bus 0x0D/0x0F/0x11/0x13 addr 0x80; tray 2 = 0x13 confirmed) and PDB (0x09/0x62)"
  echo "# tray reg0x1E alive (ff healthy, 00 dead), reg0x2D therm_trip, reg0x19 VR PG; debug registers (tray dbg0x58 0x5F 0x59 0xD0 0xD1 0xD6 0x5D 0xD5 0xD2 0xD3, PDB dbg0xD1 0xD2 0xD4 0xD5 0x58 0x5F) not read: needs a CPLD index write (not allowed)"
  for b in $BUSES; do
    line="tray-bus $b:"
    for r in 0x1E 0x2D 0x19; do line+=" reg$r=$(reg "$b" 0x80 $r)"; done
    echo "$(date -u +%T.%3N) $line"
  done
  line="PDB:"
  for r in 0x05 0x06 0x07 0x08 0x09 0x0A; do line+=" reg$r=$(reg 0x09 0x62 $r)"; done
  echo "$(date -u +%T.%3N) $line"
}
lspci_trays() {  # which chips are on the bus, per tray (tt-smi BH_UBB_BUS_IDS: tray1 0x0X, 2 0x4X, 3 0xCX, 4 0x8X)
  local all; all=$(timeout 30 lspci -D -d 1e52: 2>&1)
  echo "$all"; echo
  local t nib
  for t in 1 2 3 4; do
    nib=${TRAY_NIBBLE[$t]}
    echo "tray $t: $(echo "$all" | grep -ciE "^[0-9a-f]{4}:$nib[0-9a-f]:") chips on the bus"
  done
}
write_commands() {  # what to run by hand when ipmitool is not usable here
  cat > "$ev/capture_commands.txt" <<EOF
# Run these RIGHT NOW, before any reset or power cycle. I = $OOB_SHOWN
# 1. CPLD reads (live, reads only), per tray bus B in 0x0D 0x0F 0x11 0x13:
#      sudo ipmitool raw 0x06 0x52 B 0x80 0x01 0x1E      # alive byte (ff healthy, 00 dead)
#      sudo ipmitool raw 0x06 0x52 B 0x80 0x01 0x19      # VR power-good
#    PDB: sudo ipmitool raw 0x06 0x52 0x09 0x62 0x01 0x09 ; ... 0x01 0x0A
#    Debug registers (dbg 0x58, 0x5F, PDB 0xD2): not read: needs a CPLD index write (not allowed)
\$I sdr | grep -E '$SDR_RE'                      # 2+3. UBB power and retimer temps (live)
\$I sel elist > sel_\$(date -u +%m%dT%H%M).txt   # 4. SEL with timestamps
\$I mc info                                      # 5. BMC firmware
lspci -D -d 1e52:                                # 6. host: chips still on the bus
sudo dmesg -T | tail -n 400
EOF
}

# ---------------------------------------------------------------- 2. the run
lim_idle=60; lim_burst=460; prev=""
for x in "${pass[@]}"; do  # a hard outer limit, far past ub0_repro.py's own phase limits (it should never fire)
  case $prev in --idle-s) lim_idle=$x ;; --burst-s) lim_burst=$x ;; esac; prev=$x
done
limit=$(( 3600 + lim_idle + lim_burst ))
limit=$(( limit + ${TRAY_QUIET_CLK_CAP_S:-300} ))   # the open idle's clock check: at most 5 min
[ -n "$quiet" ] && limit=$(( limit + $(awk -v q="$quiet" 'BEGIN { s = q * 120; printf "%d", s == int(s) ? s : int(s) + 1 }') + 2 * ${TRAY_QUIET_CLK_CAP_S:-300} ))
                    # --wait-quiet-min: at most 2 clock checks (5 min each) and 2 counts of N min (one restart); N may be
                    # fractional (0.5), so awk, not $(( ))
args=(--telem "$ev/telem.csv" --result "$ev/result.json" ${quiet:+--wait-quiet-min "$quiet"} "${pass[@]}")
[ $dry = 1 ] && args+=(--dry-run)
# loss evidence for ub0_repro.py's loss class (CHIP_HANG / DROP): read-only CPLD and SEL reads, root port lspci
if [ ${#DROP[@]} -gt 0 ]; then  # a dropped device process reaches in-band ipmitool and full lspci only via sudo -n
  if [ $ipmi_ok = 1 ]; then
    [ -n "$DEV_SUDO" ] && args+=(--ipmitool "timeout 15 $DEV_SUDO $IPMITOOL")
    if [ -n "${BMC_HOST:-}" ]; then args+=(--sel-ipmitool "${OOB[*]}")
    elif [ -n "$DEV_SUDO" ]; then args+=(--sel-ipmitool "timeout 60 $DEV_SUDO $IPMITOOL"); fi
  fi
  args+=(--lspci "timeout 10 $DEV_LSUDO lspci")
else
  if [ $ipmi_ok = 1 ]; then
    if [ "$(id -u)" = 0 ] || [ -n "$SUDO" ]; then args+=(--ipmitool "$IB"); fi
    args+=(--sel-ipmitool "${OOB[*]}")                                  # lanplus: IPMI_PASSWORD stays in the env (-E)
  fi
  args+=(--lspci "timeout 10 $LSUDO lspci")
fi
if [ $dry = 0 ] && [ ${#envbad[@]} -gt 0 ]; then
  echo "VERDICT: ERROR (environment: ${envbad[0]})"; exit 1
fi
[ $dry = 1 ] || { lspci_trays > "$ev/lspci_before.txt" 2>&1; [ $ipmi_ok = 1 ] && "${OOB[@]}" mc info > "$ev/bmc_mc_info.txt" 2>&1; }
log "running: ${DROP[*]:+${DROP[*]} }${py:-python3} $HERE/ub0_repro.py ${args[*]}  (hard limit ${limit}s; dev_proc=$DEV_PROC)"
child=""; stopping=0
on_stop() {  # one TERM to ub0_repro.py (timeout forwards it): it stops the burst, closes the mesh, exits 1
  [ $stopping = 1 ] && return; stopping=1
  log "$1: stopping ub0_repro.py (pid ${child:-none})"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null
}
trap 'on_stop SIGINT' INT; trap 'on_stop SIGTERM' TERM; trap 'on_stop SIGHUP' HUP
# timeout puts the run in its own process group, so a Ctrl-C reaches only this script, which forwards one TERM
# dev_exec: as the device user when root drops (#902), else a plain exec; either way $child is timeout's pid
dev_exec timeout -k 90 "$limit" "${py:-python3}" -u "$HERE/ub0_repro.py" "${args[@]}" > "$ev/ub0.log" 2>&1 & child=$!
tail -n +1 -f --pid="$child" "$ev/ub0.log" & tailp=$!
while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
wait "$child" 2>/dev/null; r2=$?; [ $r2 != 127 ] && rc=$r2
wait "$tailp" 2>/dev/null
rc=${rc:-1}
[ "$kcache" = off ] || kc_chown                                          # what the run compiled, back to the cache owner
[ -n "$STATE_F" ] && { log "cold-state file $STATE_F given back to $STATE_OWN mode $STATE_MODE"; state_restore; }
jrc=$("${py:-python3}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["exit_code"])' "$ev/result.json" 2>/dev/null)
[ -n "$jrc" ] && rc=$jrc                                                # the result JSON is the authority
[ "$rc" = 124 ] || [ "$rc" = 137 ] && rc=1                              # the outer limit fired

# ---------------------------------------------------------------- 3. captures, live signals first
if [ $dry = 0 ]; then
  if [ $ipmi_ok = 1 ]; then
    log "capture: CPLD registers (live)"; cpld_dump > "$ev/cpld_dump.txt" 2>&1
    log "capture: BMC SDR (tray power, PSU, retimer temps; live)"
    "${OOB[@]}" sdr list > "$ev/bmc_sdr_full.txt" 2>&1; grep -E "$SDR_RE" "$ev/bmc_sdr_full.txt" > "$ev/bmc_sdr.txt"
    log "capture: BMC SEL"; "${OOB[@]}" sel elist > "$ev/bmc_sel.txt" 2>&1
  else
    write_commands; log "ipmitool not usable here: run these now, before any reset (also in capture_commands.txt):"
    sed 's/^/    /' "$ev/capture_commands.txt"
  fi
  lspci_trays > "$ev/lspci_after.txt" 2>&1
  { timeout 20 dmesg -T 2>/dev/null || timeout 20 $DSUDO dmesg -T 2>&1; } | tail -n 400 > "$ev/dmesg_tail.txt"
  grep -iE 'tenstorrent|pcie|aer|link' "$ev/dmesg_tail.txt" > "$ev/dmesg_excerpt.txt"
  [ $ipmi_ok = 1 ] && [ $rc = 3 ] && "${OOB[@]}" sdr list 2>&1 | grep -E "$SDR_RE" > "$ev/bmc_sdr_plus10s.txt" &
  [ $ipmi_ok = 1 ] && [ $rc = 3 ] && { sleep 10; wait; }
fi
{ echo "host $host"; uname -a; echo "TT_METAL_HOME ${TT_METAL_HOME:-unset} commit ${commit:-unknown}"; echo "python ${py:-none}"
  echo "kmd $(cat /sys/module/tenstorrent/version 2>/dev/null)"; for f in bios_version bios_date board_name product_name; do
  echo "$f $(cat /sys/class/dmi/id/$f 2>/dev/null)"; done; } > "$ev/versions.txt" 2>&1

# ---------------------------------------------------------------- 4. tarball and verdict
tar -czf "$out/$name.tar.gz" -C "$out" "$name" 2>/dev/null && log "evidence: $out/$name.tar.gz" || log "WARNING tarball failed; evidence left in $ev"
verdict=$("${py:-python3}" - "$ev/result.json" "$rc" "$dry" <<'EOF' 2>/dev/null
import json, sys
path, rc, dry = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
try:
    r = json.load(open(path))
except (OSError, ValueError):
    r = {}
v = r.get("verdict")
cold = (r.get("cold_state") or {}).get("status")
note = "" if cold in (None, "cold", "likely_cold") else f" (cold state {cold}: a NO TRIP says little)"
td = r.get("teardown") if isinstance(r.get("teardown"), dict) else {}
td = f", TEARDOWN TIMEOUT after {td.get('max_s')}s" if td.get("status") == "TIMEOUT" else ""  # clean exit, hung teardown
if v == "TRIP" or rc == 3:
    f = r.get("first_loss") or {}
    chip = min(f.get("chips") or r.get("lost_chips") or ["?"])
    trays = f.get("trays") or r.get("lost_trays") or ["?"]
    s = f.get("s_after_burst_begin")
    at = f"at +{s}s" if s is not None else f"in phase {f.get('phase', '?')}"
    tr = r.get("transient_chips") or []
    print(f"TRIP chip {chip} tray {'/'.join(str(t) for t in trays)} {at}"
          f" ({len(r.get('lost_chips') or [])}/32 chips lost) loss={r.get('loss_class') or 'UNKNOWN'}"
          + (f" transient={','.join(map(str, tr))}" if tr else ""))
elif v == "CCL_HANG" or rc == 4:  # --load allgather: a CCL op hung with every chip on PCIe; not a drop
    h = r.get("ccl_hang") or {}
    if h.get("line"):               # names the variant and mesh axis, or says no op was in flight
        print("CCL HANG: " + h["line"].split(": ", 1)[-1])
    else:
        print(f"CCL HANG: chip {h.get('chips', 'unknown')} ({h.get('chips_present', '?')}/32 on PCIe), op {h.get('op', '?')}"
              f" did not finish in {h.get('s_in_op', '?')}s")
elif v == "PREFLIGHT_REFUSED" or rc == 2:
    print(f"PREFLIGHT REFUSED ({'; '.join(r.get('preflight_problems') or ['see the log'])})")
elif v == "DRY_RUN_OK":
    print(f"DRY RUN OK: preflight passed, a real run would start{note}")
elif v == "NO_TRIP" and rc == 0:
    post = ""
    if r.get("long_idle_chip_missing"):  # lost after the load (GO_LONG_IDLE window): not a load trip, not clean either
        li = r.get("long_idle") if isinstance(r.get("long_idle"), dict) else {}
        gone = li.get("missing_chips") or []
        post = (f", POST-RUN CHIP LOSS chip {min(gone) if gone else '?'} tray "
                f"{'/'.join(str(t) for t in li.get('missing_trays') or ['?'])} after GO_LONG_IDLE"
                f" ({len(gone)}/32 chips gone after the load; not a load trip)")
    print(f"NO TRIP{post}{note}{td}")
else:
    print(f"ERROR ({r.get('error') or 'no result JSON; see ub0.log'}){td}")
EOF
)
[ -n "$verdict" ] || verdict="ERROR (exit $rc; see $ev/ub0.log)"
if [ $dry = 1 ]; then
  if [ ${#envbad[@]} -gt 0 ] && [ "$rc" = 0 ]; then verdict="DRY RUN: environment NOT ready (${envbad[0]})"; rc=1
  else verdict="DRY RUN: ${verdict#DRY RUN OK: }"; fi
fi
echo "VERDICT: $verdict"
exit "$rc"
