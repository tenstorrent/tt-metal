#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# load_test.sh: Blackhole Galaxy tray-drop load test, one command, no other services needed.
#
#   load_test.sh [--load ub0|matmul|allgather] [--screen | --once [--quiet Q]] [--fmax MHz] [--throttle N] [--stagger T:V]
#                [--env NAME=VALUE] [--out DIR] [--dry-run] [-- extra ub0_repro.py args]
#
# Up to 8 runs: quiet 1 min, then 3 (every chip <= 800 MHz in a row, watched in sysfs), 4 tries at each (user, 10-15).
# Each run is run_repro.sh, with the run's quiet Q split in half as in run.sh: wait until every chip has
# been <= 800 MHz for Q/2 in a row (cold quiet), open all 32 chips, upload, idle Q/2 with the device open, then a 30 s
# bf16 matmul burst on every chip at stock clock, then capture evidence (CPLD, BMC SEL/SDR, lspci, dmesg) into its own
# folder. An --idle-s after -- replaces the open-idle half.
# The first drop ends the test: DROP (tray/chip, quiet, try). PASS only if all 8 runs are clean.
# A line of statistics after every run; all runs in <out>/load_test.csv.
# --once: a single run (the ladder's first step, 1 min quiet, or --quiet Q min) with the same checks, evidence, stats
#   and verdict; exit 0 means that one run was clean, not PASS. --quiet without --once is refused (exit 1): the
#   ladder's quiets are always 1 and 3 min.
# --screen: the step-down screen, 3 runs at 1 min of quiet; PASS (screen) only if all 3 are clean.
#
# --load ub0 (default): the 4x8 ring fabric (chip-to-chip network) is up, no traffic on it, and each chip runs its
#   own matmul burst. --load matmul: the same burst with no fabric (for a box whose inter-chip cabling is broken).
#   --load allgather (opt-in): the same ring fabric, a 30 s burst of ttnn.all_gather CCL ops (both mesh
#   axes and the full 32-way gather, 2 GiB per chip per gather) instead of matmuls; a gather that hangs with every
#   chip on PCIe is CCL_HANG (exit 7), not a drop.
# --dry-run: the checks below plus run_repro.sh --dry-run (environment, plan; no device open), then exit.
# Mitigations: all OFF by default (the aggressive SysEng repro unless asked; throttling is options only).
#   With none given every run is exactly as before them: ub0_repro.py gets `--load L --fmax 0` and no TT_MM_* is set.
#   --fmax MHz        AICLK cap for every run (200-1300, e.g. 800; 0 = stock): sent to ub0_repro.py in place of --fmax 0, which
#                     sets it at each run's start and restores stock at its end.
#   --throttle N      TT_MM_THROTTLE_PERF=N (1-5).
#   --stagger T:V     TT_MM_STAGGER_TYPE=T and TT_MM_STAGGER_VALUE=V (whole numbers).
#   --env NAME=VALUE  (repeatable; TT_MM_* only) any other TT_MM_* setting.
#   They are exported to every run. Give them as options under sudo: sudo resets the environment, so
#   `TT_MM_THROTTLE_PERF=5 sudo load_test.sh` loses it. Each run's ub0_repro.py logs the TT_MM_* values it sees
#   ("mitigation env seen by ..."); the CSV's tt_mm_env column has them.
# Config label: cfg=stock, or e.g. cfg=cap800+stagger2/1000+thr5 (any other TT_MM_* appended as NAME=VALUE), from the
#   cap and the TT_MM_* actually in force; on the start line, every CSV row (cfg column), every stats line and the VERDICT.
# AICLK cap: with no --fmax every run passes --fmax 0 (stock clock), so a cap set outside is cleared at each run start.
#   `-- --fmax 800` (the form before --fmax existed) still works: ub0_repro.py's argparse keeps the LAST --fmax, so
#   `--fmax 0 --fmax 800` runs capped at 800. Giving both --fmax and `-- --fmax` is refused (exit 1).
#   The CSV's fmax_mhz column, the start line and the stats line carry the effective value (0 = stock, no cap).
# Cap check (only with a cap): each run's ub0_repro.py reads AICLK_ARB_MAX back on every chip after setting the cap
#   ("arb_max per chip (N/32 read): 0=800 1=800 ..."). The row's cap_check is ok only if the cap went to 32/32 chips
#   and all 32 read back the requested value, else cap-unverified (and the per-chip line is logged). A clean run that
#   is cap-unverified ends the test: VERDICT: CAP-UNVERIFIED, exit 4, not a PASS. On exit after a capped run, once no
#   other process holds the device (the holder check below), stock fmax is restored on every chip that answers and read
#   back per chip, so no cap is left behind; if the device is held it is not touched and a WARNING says so.
#
# A loss is DROP (a tray drop), CHIP_HANG (one chip fell off PCIe at an MMIO timeout, AER up on its root port, every
#   tray CPLD alive) or UNKNOWN (not proven either way, or no class given: never read as DROP), all ub0_repro.py's
#   loss_class. All three end the test with exit 3 (the chip needs a reset); they are counted apart in the stats and the CSV (result, loss_class).
# Early verdict: while a run goes, its log is read every 2 s. As soon as it has a FIRST LOSS line, the
#   verdict (one chip at mesh_open = CHIP_HANG, else DROP) goes to <out>/VERDICT, its CSV row (rc column `early`) and
#   stdout, so a job runner's kill -9 (the broker's chip-loss kill) cannot lose it. It is written once: a later signal
#   keeps it and writes no row; a run that still ends by itself with its own DROP/CHIP_HANG replaces it (same row).
#
# Before every run it checks, and refuses to start otherwise:
#   - all 32 chips answer on PCIe (a tray that dropped earlier reads all-ones: power cycle the system first);
#   - no other process has the device open (lsof of /dev/tenstorrent/* and of the /dev/shm CHIP_IN_USE lock files)
#     or owns a chip's CHIP_IN_USE lock (UMD's mutex in that file; ps of each owner); waits up to 10 min for it to be
#     released. Run as root or with passwordless sudo so other users' processes can be seen. tt-telemetry is ignored.
#   - no device-broker job that this test did not start (another user or project, e.g. an LTX job; the calling user's
#     own ended jobs do not count) ran since the previous run ended or runs now (broker_collision.py reads
#     `tt-device-mcp status`, read-only). Such a job is a COLLISION: a row goes to <out>/collisions.jsonl (job, owner,
#     status, start/end, kill reason, chips if logged, local time), the test waits until the job has ended and the
#     broker reads healthy, unheld and 0 off-bus for 5 min, then the run's quiet starts from zero. It never cancels or kills a job. No broker on the box: the gate is off.
# Chips left at the busy clock by an earlier program are sent to idle, so the quiet wait can start.
# A drop leaves the tray off: power cycle the system before testing again.
# Under sudo (or as root) only the device processes drop root (a root device process triggers the
#   chip-0 mesh_open CHIP_HANG): each run's ub0_repro.py (run_repro.sh does it) and this script's own tt_umd steps (the
#   GO_LONG_IDLE send and the stock-fmax restore, via run_repro.sh --device-exec) run as $SUDO_USER (else
#   $TRAY_REPRO_DEVICE_USER, else the owner of this tree) with the group of /dev/tenstorrent/<n> (stat -c %G at run
#   time), no capabilities, cwd <out>/devhome and that user's own HOME, so its own JIT cache (~/.cache/tt-metal-cache)
#   is used (not a hang fix: a cold JIT cache was ruled out as the hang's cause). The holder scans (lsof, the CHIP_IN_USE
#   owners, ps), the captures and the BMC reads stay root. If that user cannot reach the devices, hugepages or
#   /dev/shm locks: CANNOT START, exit 2 (never a silent root fallback). TRAY_REPRO_DEVICE_AS_ROOT=1 keeps the old
#   root device process, for A/B runs only. Every CSV row ends with dev_proc (e.g. uid:3011(<user>) gid:998
#   capeff=0 jit_cache=/home/<user>/.cache/tt-metal-cache(warm); `... caller` when not run as root, which changes
#   nothing). See run_repro.sh's header.
# env: TT_METAL_HOME (default: the tt-metal next to this script), plus run_repro.sh's BMC_HOST/BMC_USER/BMC_PASS.
# Never on cs04 (any host name ending in cs04, e.g. f07cs04): refused, exit 2.
# The tt-projects host: a tray drop there reboots the box and stops them until it is back. The test prints a
#   warning there but runs (the allow-file guard was removed 10-15 by user decision: SysEng tests that box too).
# Last line, always: `DROPPED TRAY: none`, or one clause per tray that left the PCIe bus, e.g.
#   `DROPPED TRAY: tray 1 (chips 8-15, pci 0000:41:00.0-0000:48:00.0)`; `DROPPED TRAY: UNKNOWN (chips ..., pci ...)` when a
#   lost chip maps to no known tray or its config space could not be read. Printed once, on every exit path.
# exit: 0 PASS (with --once: the run was clean), 3 DROP, CHIP_HANG or UNKNOWN, 2 cannot start (chips missing, device in use, refused host),
#   4 CAP-UNVERIFIED (a clean run whose cap was not read back on every chip), 1 error (a bad option, or one step failed
#   3 times without a verdict), 7 CCL_HANG (--load allgather: a gather hung, no chip lost).
# LOAD_TEST_SYS, LOAD_TEST_SHM and LOAD_TEST_PROC (default /sys, /dev/shm, /proc) are for the tests only.
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
RR=$HERE/run_repro.sh
BC=$HERE/broker_collision.py; SINCE=$(date +%s); NCOL=0; n=0
LOAD=ub0; OUT=$PWD/load_test_$(hostname -s)_$(date -u +%Y%m%dT%H%M%SZ); DRY=0; PASS=(); ONCE=0; ONCE_Q=1; QSET=0
SCREEN=0; FMAX_OPT=""
SYS=${LOAD_TEST_SYS:-/sys}; SHM=${LOAD_TEST_SHM:-/dev/shm}; PROC=${LOAD_TEST_PROC:-/proc}
HB_S=${TRAY_REPRO_HEARTBEAT_S:-20}; HB_PID=""
hb_start() {  # $1 what runs: one line every HB_S s until hb_stop (the EXIT traps stop it too), so a job runner that reaps
              # a silent job as hung (the device broker: no output for about a minute) leaves the waits below alone.
              # The loop also ends by itself within 1 s once this shell is gone.
  hb_stop; local main=$$ what=$1
  ( trap - EXIT; s0=$(date +%s); k=0
    while kill -0 "$main" 2>/dev/null; do
      sleep 1 > /dev/null 2>&1; k=$((k + 1)); [ "$k" -ge "$HB_S" ] || continue; k=0; say "$what: running, $(( $(date +%s) - s0 )) s"
    done ) &
  HB_PID=$!
}
hb_stop() { [ -n "$HB_PID" ] && { kill "$HB_PID" 2>/dev/null; wait "$HB_PID" 2>/dev/null; }; HB_PID=""; }

# --- dropped-tray verdict -------------------------------------------------------------------------------------------
# Every exit path ends with ONE line naming the tray(s) that left the PCIe bus, so a bare run says what dropped:
#   DROPPED TRAY: none
#   DROPPED TRAY: tray 1 (chips 8-15, pci 0000:41:00.0-0000:48:00.0)        (one clause per tray, comma-separated)
#   DROPPED TRAY: UNKNOWN (chips 8+9, pci 0000:41:00.0+0000:42:00.0)        (lost chips in no known tray, or unread)
# Several chips in a clause are joined with '+', consecutive chip indexes compressed to a-b; a single chip prints bare.
# Tray from the endpoint's PCI bus (tt-smi BH_UBB_BUS_IDS, re-read on a box 2026-10-16: tray 0 =
# bus 01-08 = chips 0-7, tray 1 = 41-48 = chips 8-15, tray 2 = c1-c8 = chips 16-23, tray 3 = 81-88 = chips 24-31),
# the chip index from <bdf>/tenstorrent/tenstorrent!N while tt-kmd's node is there. A chip that still answers config
# space but hung on the NOC is NOT a tray drop: a CHIP_HANG prints "none" (its own VERDICT line says CHIP_HANG).
# A config read that times out or cannot be read is "unread", never clean (chip0 rule: post-loss sysfs reads can block
# forever under AER recovery), so it goes to UNKNOWN and the whole scan is bounded by LOAD_TEST_DT_DEADLINE seconds.
DT_DONE=0; DT_BUSMAP=${LOAD_TEST_TRAY_BUSMAP:-"0:01-08 1:41-48 2:c1-c8 3:81-88"}
dt_probe() {  # one "<bdf> <chip|-> ok|lost|unread" line per Blackhole endpoint
  local d bdf c nn w rc dl
  dl=$(( $(date +%s) + ${LOAD_TEST_DT_DEADLINE:-20} ))
  for d in "$SYS"/bus/pci/devices/*; do
    [ "$(cat "$d/vendor" 2>/dev/null)" = 0x1e52 ] || continue
    bdf=${d##*/}; c=-
    for nn in "$d"/tenstorrent/tenstorrent!*; do [ -e "$nn" ] && c=${nn##*!}; done
    if [ "$(date +%s)" -ge "$dl" ]; then echo "$bdf $c unread"; continue; fi
    w=$(timeout "${LOAD_TEST_DT_READ_S:-2}" od -An -tx2 -N2 "$d/config" 2>/dev/null); rc=$?
    w=${w//[[:space:]]/}
    if [ $rc -ne 0 ] || [ -z "$w" ]; then echo "$bdf $c unread"
    elif [ "$w" = 1e52 ]; then echo "$bdf $c ok"
    else echo "$bdf $c lost"; fi
  done
}
dropped_tray_line() {  # write-once, the very last line of the run
  [ "${DT_DONE:-0}" = 1 ] && return 0
  DT_DONE=1
  dt_probe | awk -v map="$DT_BUSMAP" '
    function hex(s,  i,v,d) { s = tolower(s); v = 0
      for (i = 1; i <= length(s); i++) { d = index("0123456789abcdef", substr(s, i, 1)) - 1; if (d < 0) return -1; v = v * 16 + d }
      return v }
    function trayof(bdf,  i, f, bus) { split(bdf, f, ":"); bus = hex(f[2])
      for (i = 1; i <= nm; i++) if (bus >= LO[i] && bus <= HI[i]) return TN[i]
      return "" }
    function ckey(x) { return (x ~ /^[0-9]+$/) ? x + 0 : 1e9 }
    function chiplist(s,  i, j, k, v, a, out, st, pv) {   # " 8 9 10 12" -> "8-10+12" (non-numeric kept as given, last)
      k = split(s, a, " "); out = ""; st = ""; pv = ""
      for (i = 2; i <= k; i++) { v = a[i]; for (j = i - 1; j >= 1 && ckey(a[j]) > ckey(v); j--) a[j + 1] = a[j]; a[j + 1] = v }
      for (i = 1; i <= k; i++) {
        if (a[i] !~ /^[0-9]+$/) { out = out (out == "" ? "" : "+") a[i]; continue }
        if (st == "") { st = a[i]; pv = a[i]; continue }
        if (a[i] + 0 == pv + 0 + 1) { pv = a[i]; continue }
        out = out (out == "" ? "" : "+") (st == pv ? st : st "-" pv); st = a[i]; pv = a[i] }
      if (st != "") out = out (out == "" ? "" : "+") (st == pv ? st : st "-" pv)
      return out == "" ? "?" : out }
    BEGIN { nm = split(map, M, " ")
      for (i = 1; i <= nm; i++) { split(M[i], a, ":"); split(a[2], b, "-"); TN[i] = a[1]; LO[i] = hex(b[1]); HI[i] = hex(b[2]) } }
    $3 == "ok" { next }
    { t = ($3 == "unread") ? "UNKNOWN" : trayof($1); if (t == "") t = "UNKNOWN"
      if (!(t in SEEN)) { SEEN[t] = 1; ORD[++no] = t }
      CH[t] = CH[t] " " ($2 == "-" ? "?" : $2)
      PCI[t] = PCI[t] (PCI[t] == "" ? "" : " ") $1 }
    END {
      if (no == 0) { print "DROPPED TRAY: none"; exit }
      for (i = 1; i <= no; i++) for (j = i + 1; j <= no; j++) {   # trays in numeric order, UNKNOWN last
        ai = (ORD[i] == "UNKNOWN") ? 1e9 : ORD[i] + 0; aj = (ORD[j] == "UNKNOWN") ? 1e9 : ORD[j] + 0
        if (aj < ai) { tmp = ORD[i]; ORD[i] = ORD[j]; ORD[j] = tmp } }
      out = ""
      for (i = 1; i <= no; i++) { t = ORD[i]; k = split(PCI[t], p, " ")
        if (t == "UNKNOWN") { pl = p[1]; for (j = 2; j <= k; j++) pl = pl "+" p[j]
          c = "UNKNOWN (chips " chiplist(CH[t]) ", pci " pl ")" }
        else c = "tray " t " (chips " chiplist(CH[t]) ", pci " p[1] (k > 1 ? "-" p[k] : "") ")"
        out = out (out == "" ? "" : ", ") c }
      print "DROPPED TRAY: " out }'
}
trap 'hb_stop; dropped_tray_line' EXIT
# --- end dropped-tray verdict ----------------------------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case $1 in
    --load) LOAD=${2:?--load needs ub0, matmul or allgather}; shift ;;
    --out) OUT=${2:?--out needs a directory}; shift ;;
    --dry-run) DRY=1 ;;
    --once) ONCE=1 ;;
    --screen) SCREEN=1 ;;
    --quiet) ONCE_Q=${2:?--quiet needs minutes}; QSET=1; shift ;;
    --env) [[ ${2:-} =~ ^TT_MM_[A-Z0-9_]+=[^[:space:],]*$ ]] || { echo "--env needs TT_MM_NAME=VALUE, no spaces or commas (got '${2:-}')" >&2; exit 1; }
           export "${2?}"; shift ;;
    --fmax) [[ ${2:-} =~ ^(0|[1-9][0-9]{2,3})$ ]] && { [ "$2" = 0 ] || [ "$2" -ge 200 -a "$2" -le 1300 ]; } ||
              { echo "--fmax needs 0 (stock) or a cap of 200 to 1300 MHz (got '${2:-}')" >&2; exit 1; }
            FMAX_OPT=$2; shift ;;
    --throttle) [[ ${2:-} =~ ^[1-5]$ ]] || { echo "--throttle needs 1 to 5 (got '${2:-}')" >&2; exit 1; }
                export TT_MM_THROTTLE_PERF=$2; shift ;;
    --stagger) [[ ${2:-} =~ ^([0-9]+):([0-9]+)$ ]] || { echo "--stagger needs TYPE:VALUE, whole numbers (got '${2:-}')" >&2; exit 1; }
               export TT_MM_STAGGER_TYPE=${BASH_REMATCH[1]} TT_MM_STAGGER_VALUE=${BASH_REMATCH[2]}; shift ;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    --) shift; PASS=("$@"); break ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 1 ;;
  esac
  shift
done
FMAX_ARG=${FMAX_OPT:-0}                       # what ub0_repro.py gets after --load: 0 (stock) unless --fmax
FMAX=$FMAX_ARG; i=0                           # effective AICLK cap of every run: the last --fmax passed through
while [ $i -lt ${#PASS[@]} ]; do
  case ${PASS[$i]} in --fmax) FMAX=${PASS[$((i + 1))]:-0}; i=$((i + 1)) ;; --fmax=*) FMAX=${PASS[$i]#--fmax=} ;; esac
  i=$((i + 1))
done
[ -n "$FMAX_OPT" ] && [[ " ${PASS[*]} " =~ \ --fmax[=\ ] ]] && { echo "give the cap once: --fmax MHz or -- --fmax MHz, not both" >&2; exit 1; }
[ -n "${TT_METAL_HOME:-}" ] || { [ -d "$HERE/tt-metal" ] && export TT_METAL_HOME=$HERE/tt-metal; }
# this package: a pip venv next to this script (made by setup.sh) is the python when UB0_PYTHON is unset (sudo drops env)
[ -z "${UB0_PYTHON:-}" ] && [ -z "${TT_METAL_HOME:-}" ] && [ -x "$HERE/venv/bin/python" ] && export UB0_PYTHON=$HERE/venv/bin/python
[ -x "$RR" ] || { echo "missing $RR" >&2; exit 1; }
H=$(hostname -s)
case $H in *cs04) echo "$H: no tray test runs here (cs04 is never used)" >&2; exit 2 ;; esac
[ $QSET = 1 ] && [ $ONCE = 0 ] && { echo "--quiet only goes with --once (the ladder's quiets are always 1 and 3 min)" >&2; exit 1; }
[ $SCREEN = 1 ] && [ $ONCE = 1 ] && { echo "--screen and --once do not go together" >&2; exit 1; }
[[ $ONCE_Q =~ ^[1-9][0-9]*$ ]] || { echo "--quiet needs whole minutes (got '$ONCE_Q')" >&2; exit 1; }
if [ $ONCE = 1 ]; then QUIETS=$ONCE_Q; TRIES=1; elif [ $SCREEN = 1 ]; then QUIETS=1; TRIES="1 2 3"; else QUIETS="1 3"; TRIES="1 2 3 4"; fi
PY=${UB0_PYTHON:-${TT_METAL_HOME:-}/python_env/bin/python}; [ -x "$PY" ] || PY=python3
if [ "$(id -u)" = 0 ]; then SUDO=""; elif sudo -n true 2>/dev/null; then SUDO="sudo -n"; else SUDO=none; fi

say() { echo "$(date '+%F %T %Z') $H: $*"; }
half_min() { [ $(( $1 % 2 )) = 0 ] && echo $(( $1 / 2 )) || echo "$(( $1 / 2 )).5"; }   # Q/2 minutes: 1 -> 0.5, 3 -> 1.5
chips() { ls "$SYS"/class/tenstorrent 2>/dev/null | wc -l; }
answering() {  # Blackhole PCIe functions whose config space answers (a powered-off tray reads ffff)
  local n=0 d
  for d in "$SYS"/bus/pci/devices/*; do
    [ "$(cat "$d/vendor" 2>/dev/null)" = 0x1e52 ] || continue
    [ "$(od -An -tx2 -N2 "$d/config" 2>/dev/null | tr -d ' ')" = 1e52 ] && n=$((n + 1))
  done
  echo $n
}
lockowners() {  # pids that own a chip's UMD lock, $SHM/*CHIP_IN_USE*: UMD holds it with a robust pthread mutex at
  # offset 0 of the file, not an flock, so lsof can miss the holder. Read as the broker's repro_job.sh mutexown does:
  # glibc int __lock @0 (owner TID | OWNER_DIED 0x40000000), int __owner @8 (owner TID; 0x7fffffff/0x7ffffffe: use
  # __lock's TID); the TID's Tgid in $PROC/<tid>/status is the process. No such TID, or OWNER_DIED with no owner: stale.
  local f w lk tid tg
  for f in "$SHM"/*CHIP_IN_USE*; do [ -f "$f" ] || continue
    w=($(od -An -v -t u4 -N 12 "$f" 2>/dev/null)); [ ${#w[@]} -ge 3 ] || continue
    lk=${w[0]}; tid=$(( w[2] & 0x3fffffff ))
    case ${w[2]} in 2147483647|2147483646) tid=$(( lk & 0x3fffffff )) ;; esac
    [ $tid = 0 ] && continue
    [ $(( lk >> 30 & 1 )) = 1 ] && [ $(( lk & 0x3fffffff )) = 0 ] && continue
    tg=$(awk '$1 == "Tgid:" { print $2; exit }' "$PROC/$tid/status" 2>/dev/null); [ -n "$tg" ] && echo "$tg"
  done | sort -u
}
holders() {  # other processes holding the device: "pid user command" per line (tt-telemetry ignored): lsof of
             # /dev/tenstorrent/* and the CHIP_IN_USE files, and ps of every CHIP_IN_USE lock owner
  local L f p fs=(/dev/tenstorrent/*)
  for f in "$SHM"/*CHIP_IN_USE*; do [ -e "$f" ] && fs+=("$f"); done
  if [ "$SUDO" = none ]; then L=$(lsof -w "${fs[@]}" 2>/dev/null); else L=$($SUDO lsof -w "${fs[@]}" 2>/dev/null); fi
  { echo "$L" | awk 'NR > 1 { print $2, $3, $1 }'
    for p in $(lockowners); do ps -o pid=,user=,comm= -p "$p" 2>/dev/null; done | awk '{ print $1, $2, $3 }'
  } | awk 'NF == 3 && $3 !~ /^tt_telem/' | sort -u
}
nreport() {  # the numeric lines over every run so far, before the verdict; matmul loads only
  [ "$LOAD" = allgather ] || [ -z "${OUT:-}" ] || [ ! -d "$OUT" ] || python3 "$HERE/numeric_report.py" "$OUT"
}
FINAL=""; FINAL_RC=0
final() { FINAL_RC=$1; FINAL=$2; }   # the decided final verdict, set before it is printed: a later signal (e.g. during the
                                     # EXIT trap's fmax_stock, up to 300 s) keeps it in <out>/VERDICT
cannot() { say "CANNOT START: $*"; final 2 "VERDICT: CANNOT START ($*)"; echo "$FINAL"; [ -f "${CSV:-}" ] && stats; exit 2; }
precheck() {  # $1 = runs done so far
  local a h i
  a=$(answering)
  if [ "$a" != 32 ]; then
    [ "$1" -gt 0 ] && { say "only $a/32 chips answer after run $1"; final 3 "VERDICT: DROP (chips lost after run $1: $a/32 answer) | $OUT"; echo "$FINAL"; stats; exit 3; }
    cannot "$a/32 chips answer on PCIe; a tray is off: power cycle the system"
  fi
  [ "$SUDO" = none ] && say "WARNING: no root or passwordless sudo, so other users' processes using the device cannot be seen"
  for i in $(seq 30); do   # a line every 20 s (a job runner reaps a silent job as hung)
    h=$(holders); [ -z "$h" ] && return 0
    say "device in use, waiting up to 10 min for it to be released ($((i * 20 - 20)) s so far): $(echo "$h" | tr '\n' ';')"
    sleep 20
  done
  cannot "device in use by: $(echo "$h" | tr '\n' ';')"
}
collision_gate() {  # a broker job not ours, running now or ended since $SINCE: record it, wait it out, settle
  local rc
  [ -n "${BROKER_STATUS_CMD:-}" ] || command -v tt-device-mcp > /dev/null || return 0
  hb_start "collision gate (other broker jobs)"
  python3 "$BC" "$1" --since "$SINCE" --rows "$OUT/collisions.jsonl" --seen "$OUT/collisions.seen" --key "$H-run$((n + 1))" \
    --settle "${LOAD_TEST_SETTLE:-300}" --poll "${LOAD_TEST_POLL:-30}" --max-wait 3600 > /dev/null; rc=$?; hb_stop
  case $rc in
    0) ;;
    10) NCOL=$((NCOL + 1)); say "COLLISION: a broker job this test did not start used the box (rows: $OUT/collisions.jsonl);" \
          "$([ "$1" = wait ] && echo "it has ended and the broker settled: the quiet starts from zero" || echo "dry run: no wait")" ;;
    11) say "COLLISION: a broker job this test did not start is running now (dry run: no wait; rows: $OUT/collisions.jsonl)" ;;
    12) cannot "a broker job this test did not start still ran, or the broker had not settled, after 1 h (see $OUT/collisions.jsonl)" ;;
    *) say "WARNING: the broker's job list could not be read (rc $rc): collision gate skipped for this run" ;;
  esac
}
idle_clocks() {  # a program that exited without its teardown leaves the chips at the busy clock (1350 MHz), and
                 # the quiet wait (<= 800 MHz) would never pass: send AICLK_GO_LONG_IDLE (ARC 0x54), as tt-metal's close does
  local hi; hi=$(cat "$SYS"/class/tenstorrent/*/tt_aiclk 2>/dev/null | awk '$1 > 800' | wc -l)
  [ "$hi" = 0 ] && return 0
  say "$hi chip(s) above 800 MHz with nothing using the device: sending GO_LONG_IDLE to every chip"
  "${DEVX[@]}" "$PY" - "$(chips)" <<'EOF'
import sys, tt_umd
for c in range(int(sys.argv[1])):
    d = tt_umd.TTDevice.create(c); d.init_tt_device(); d.arc_msg(0x54, True, [], 1000); del d
EOF
  sleep 2; say "clocks now: $(cat "$SYS"/class/tenstorrent/*/tt_aiclk | sort -n | uniq -c | tr -s ' \n' ' ')(count MHz)"
}
stats() {
  awk -F, -v h="$H" -v f="$FMAX" -v c="$CFG" 'NR > 1 { n++; r[$8]++; if ($8 == "CLEAN") q[$2]++; if ($15 == "cap-unverified") u++ }
    END { printf "%s stats: %d run(s): %d clean, %d drop, %d error, %d CHIP_HANG, %d UNKNOWN | clean at 1/3 min: %d/%d | fmax %s | cfg=%s%s\n",
          h, n, r["CLEAN"], r["DROP"], r["ERROR"], r["CHIP_HANG"], r["UNKNOWN"], q[1], q[3], (f == 0 ? "0 (stock)" : f " MHz"), c,
          (f == 0 ? "" : " | " u + 0 " cap-unverified") }' "$CSV"
}
cap_check() {  # $1 = a capped run's log; sets CC=ok if ub0_repro.py sent the cap to 32/32 chips and every chip read it
  local s p     # back, else CC=cap-unverified; CAPLINE = the run's per-chip arb_max line
  s=$(grep -aoE 'fmax [0-9a-z]+ sent to [0-9]+/[0-9]+ chips' "$1" | tail -1)
  p=$(grep -aoE 'arb_max per chip \([0-9]+/[0-9]+ read\):( [0-9]+=[0-9-]+)*' "$1" | tail -1); CAPLINE=${p:-"no arb_max per chip line in $1"}
  CC=cap-unverified
  [ "$s" = "fmax $FMAX sent to 32/32 chips" ] && [ -n "$p" ] || return 0
  CC=$(tr ' ' '\n' <<< "${p#*: }" | awk -F= -v f="$FMAX" 'NF == 2 { n++; if ($2 != f) bad++ } END { print (n == 32 && !bad ? "ok" : "cap-unverified") }')
}
fmax_stock() {  # on exit after a capped run: stock fmax (ARC 0x23, restore-default bit, as ub0_repro.py's set_fmax(0)) on
                # every chip that answers, read back per chip; only once no other process holds the device
  local h o rc
  h=$(holders)
  [ -n "$h" ] && { say "WARNING: stock fmax NOT restored, the device is held by: $(echo "$h" | tr '\n' ';') - check arb_max before LTX or the next test"; return; }
  hb_start "end: stock fmax restore"
  o=$(timeout 300 "${DEVX[@]}" "$PY" - "$SYS" 2>&1 <<'EOF'
import glob, os, sys, time
import tt_umd
ids = sorted(int(os.path.basename(p)) for p in glob.glob("/dev/tenstorrent/*") if os.path.basename(p).isdigit())
def up(c):
    try:
        return int(open(f"{sys.argv[1]}/class/tenstorrent/tenstorrent!{c}/tt_aiclk").read()) < 100000
    except (OSError, ValueError):
        return False
live = [c for c in ids if up(c)]
for c in live:
    d = tt_umd.TTDevice.create(c); d.init_tt_device(); d.arc_msg(0x23, True, [0, 1], 1000); del d
time.sleep(2)  # telemetry lags the message by 1-2 s
out = []
for c in ids:
    v = "-"          # not live: nothing to restore
    if c in live:
        v = "?"      # live but arb_max not readable: not confirmed
        d = tt_umd.TTDevice.create(c); d.init_tt_device(); t = d.get_arc_telemetry_reader()
        if t.is_entry_available(tt_umd.TelemetryTag.AICLK_ARB_MAX):
            v = t.read_entry(tt_umd.TelemetryTag.AICLK_ARB_MAX) & 0xFFFF
        del t, d
    out.append(f"{c}={v}")
print("arb_max per chip after the stock restore:", " ".join(out))
EOF
); rc=$?; hb_stop
  if [ $rc = 0 ] && [[ $o =~ arb_max\ per\ chip\ after ]] &&
     tr ' ' '\n' <<< "${o##*: }" | awk -F= -v f="$FMAX" 'NF == 2 && $2 != "-" && ($2 == "?" || $2 + 0 <= f + 0) { bad = 1 } END { exit bad }'; then
    say "end: stock fmax restored (no cap left behind): ${o##*$'\n'}"
  else say "WARNING: stock fmax restore not confirmed, a chip may still be capped: check arb_max before LTX or the next test: $(tail -1 <<< "$o")"; fi
}
MM=$(env | grep -E '^TT_MM_[A-Z0-9_]+=' | sort | paste -sd' ')
CFG=""; [ "$FMAX" != 0 ] && CFG+="+cap$FMAX"
[ -n "${TT_MM_STAGGER_TYPE+x}${TT_MM_STAGGER_VALUE+x}" ] && CFG+="+stagger${TT_MM_STAGGER_TYPE-}/${TT_MM_STAGGER_VALUE-}"
[ -n "${TT_MM_THROTTLE_PERF+x}" ] && CFG+="+thr$TT_MM_THROTTLE_PERF"
for v in $MM; do case $v in TT_MM_STAGGER_TYPE=*|TT_MM_STAGGER_VALUE=*|TT_MM_THROTTLE_PERF=*) ;; *) CFG+="+$v" ;; esac; done
CFG=${CFG#+}; CFG=${CFG:-stock}; CFG=${CFG//,/;}
say "mitigation env: ${MM:-none}; AICLK cap (fmax): $([ "$FMAX" = 0 ] && echo "0 (stock; a cap set outside is cleared, pass --fmax 800 to cap)" || echo "$FMAX MHz"); cfg=$CFG"
[ -z "$MM" ] && [ -n "${SUDO_USER:-}" ] && say "note: run under sudo with no TT_MM_* set; sudo drops the caller's environment, pass them as options (--throttle, --stagger, --env)"

# tt-metal writes under <TT_METAL_LOGS_PATH or the cwd>/generated; a dir left by another user (a sudo run) fails every
# run late, so check it first. Check only: nothing is chowned or chmodded.
# Applies to --dry-run too (it reports the same result and exits 2 on a bad dir).
GEN=${TT_METAL_LOGS_PATH:-$PWD}/generated; BADP=()
if [ -e "$GEN" ]; then
  for d in "$GEN" "$GEN"/{watcher,dprint,inspector,fabric,profiler,noc_data}; do
    [ -e "$d" ] || continue; { [ -w "$d" ] && [ -x "$d" ]; } || BADP+=("$d")
  done
else
  [ -w "$(dirname "$GEN")" ] || BADP+=("$(dirname "$GEN")")
fi
if [ ${#BADP[@]} -gt 0 ]; then
  for d in "${BADP[@]}"; do say "output dir not writable: $(stat -c '%n %U:%G %a' "$d")"; done
  cannot "tt-metal output dir not writable by $(id -un) (uid $(id -u)): ${BADP[*]}; fix its owner, this script never chowns"
fi
say "output dir check: $GEN is writable by $(id -un)"

# the device process identity: run_repro.sh decides and checks it; root drops to a non-root user for every
# device open (DEVX wraps this script's own tt_umd steps), a non-root caller is unchanged (DEVX empty)
export TRAY_REPRO_DEV_CWD=${TRAY_REPRO_DEV_CWD:-$OUT/devhome}
DEVP=$("$RR" --device-ident 2>&1) || cannot "device process: $(tail -1 <<< "$DEVP" | sed 's/^DEVICE PROCESS REFUSED: //')"
[ "$(wc -l <<< "$DEVP")" -gt 1 ] && say "$(head -n -1 <<< "$DEVP" | tr '\n' ' ')"
DEVP=$(tail -1 <<< "$DEVP"); DEVP=${DEVP#dev_proc=}; DEVX=()
case $DEVP in *" caller"|*"as-root("*) ;; *) DEVX=("$RR" --device-exec --) ;; esac
say "device process: $DEVP$([ ${#DEVX[@]} -gt 0 ] && echo " (root drops for every device open; cwd $TRAY_REPRO_DEV_CWD)")"
[[ $DEVP == *"as-root("* ]] && say "WARNING: TRAY_REPRO_DEVICE_AS_ROOT=1: the device process runs as root (A/B only; root triggers the chip-0 mesh_open hang)"

if [ $DRY = 1 ]; then
  mkdir -p "$OUT"; collision_gate check
  precheck 0; say "checks passed: 32/32 chips answer, no other process holds the device"
  "$RR" --dry-run --out "$OUT/dry_run" -- --load "$LOAD" --fmax "$FMAX_ARG" "${PASS[@]}"; exit $?
fi
mkdir -p "$OUT"; CSV=$OUT/load_test.csv
echo "n,quiet_min,try,attempt,start_utc,end_utc,rc,result,verdict,evidence,loss_class,tt_mm_env,fmax_mhz,cfg,cap_check,dev_proc" > "$CSV"
if [ $ONCE = 1 ]; then say "load test: load=$LOAD, cfg=$CFG, one run at quiet $ONCE_Q min (--once); out $OUT"
elif [ $SCREEN = 1 ]; then say "load test: load=$LOAD, cfg=$CFG, screen: 3 runs at quiet 1 min (--screen), first drop ends it; out $OUT"
else say "load test: load=$LOAD, cfg=$CFG, quiet 1/3 min x 4 tries, first drop ends it; out $OUT"; fi
n=0
RUNNING=0; rp=""; EARLY=""; ECLS=""; SIGQ=""
fl_class() {  # $1 = a FIRST LOSS line: CHIP_HANG if it is one chip at mesh_open, DROP if a whole tray (8+ chips) was
  # lost, else UNKNOWN (no power evidence here: never DROP)
  local ch nc; ch=$(echo "$1" | sed -n 's/.*chip(s) \[\([^]]*\)\].*/\1/p'); nc=$(echo "$ch" | tr ',' '\n' | grep -c '[0-9]')
  if [[ $1 == *"phase mesh_open"* ]] && [ "$nc" = 1 ]; then echo CHIP_HANG
  elif [ "$nc" -ge 8 ] || [[ $1 == *"..."* ]]; then echo DROP; else echo UNKNOWN; fi
}
term_run() {  # the run's whole process group (setsid: pgid = $rp). The pid alone is TERMed only while bash's job table
  # still lists the run as running (the instant before setsid makes the group): a reaped run's pid may be reused
  # A group still there 5 s after the TERM gets KILL: the broker KILLs us 10 s after its TERM, and a setsid group it
  # never signals would outlive us with the chips open
  local i
  [ -n "$rp" ] || return 0
  if kill -TERM -- -"$rp" 2>/dev/null; then
    for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 -- -"$rp" 2>/dev/null || return 0; sleep 0.5; done
    kill -KILL -- -"$rp" 2>/dev/null
  elif [[ $'\n'$(jobs -rp)$'\n' == *$'\n'"$rp"$'\n'* ]]; then kill -TERM "$rp" 2>/dev/null; fi
  return 0
}
sig_traps() { trap 'on_signal SIGINT' INT; trap 'on_signal SIGTERM' TERM; trap 'on_signal SIGHUP' HUP; }
defer_sigs() { SIGQ=""; trap 'SIGQ=SIGINT' INT; trap 'SIGQ=SIGTERM' TERM; trap 'SIGQ=SIGHUP' HUP; }   # while a verdict
undefer_sigs() { sig_traps; [ -z "$SIGQ" ] || on_signal "$SIGQ"; }   # and its row are written: a signal waits for both
early_verdict() {  # the run log has a FIRST LOSS line: write the verdict now (<out>/VERDICT, the CSV row, stdout),
  # while the run still goes, so the broker's kill -9 after a chip loss (no handler runs) cannot lose it. Write-once.
  local fl v
  [ -z "$FINAL" ] || return 0
  fl=$(grep -a 'FIRST LOSS' "$d.log" 2>/dev/null | tail -1 | cut -d'|' -f3-); [ -n "$fl" ] || return 0
  defer_sigs
  ECLS=$(fl_class "$fl"); EARLY=$n
  v="VERDICT: $ECLS at quiet $q min, try $t (run $n): written early, the run still going: $fl | evidence $d | cfg=$CFG"
  final 3 "$v"; echo "$v" > "$OUT/VERDICT"
  echo "$n,$q,$t,$att,$t0,$(date -u +%FT%TZ),early,$ECLS,\"${fl//\"/}\",$d,$ECLS,${MM:-none},$FMAX,$CFG,-,$DEVP" >> "$CSV"
  echo "$v" 2>/dev/null
  undefer_sigs
  say "run $n: verdict written early (FIRST LOSS in the run log), also in $OUT/VERDICT; the run goes on" 2>/dev/null
  stats 2>/dev/null
}
on_signal() {  # a job runner can end us mid-run: the device broker kills a job whose chip left the PCIe bus (its
  # device recovery), so say what the run saw before we go, in <out>/VERDICT (also on stdout, if still connected)
  local sig=$1 fl cls v
  trap - INT TERM HUP
  if [ -n "$FINAL" ]; then   # the verdict is decided and its CSV row written: keep it, and let the exit (and its stock
    echo "$FINAL" > "$OUT/VERDICT"; say "$sig after the final verdict: kept in $OUT/VERDICT" 2>/dev/null   # restore) finish
    [ "$RUNNING" = 1 ] && { term_run; exit "$FINAL_RC"; }   # an early verdict: no second row; end the run
    return
  fi
  if [ "$RUNNING" != 1 ]; then v="VERDICT: INTERRUPTED by $sig between runs | $OUT"
  else
    term_run
    fl=$(grep -a 'FIRST LOSS' "$d.log" 2>/dev/null | tail -1 | cut -d'|' -f3-)
    if [ -n "$fl" ]; then
      cls=$(fl_class "$fl")
      v="VERDICT: $cls at quiet $q min, try $t (run $n): the job was ended by $sig after: $fl | evidence $d | cfg=$CFG"
      echo "$n,$q,$t,$att,$t0,$(date -u +%FT%TZ),$sig,$cls,\"${fl//\"/}\",$d,$cls,${MM:-none},$FMAX,$CFG,-,$DEVP" >> "$CSV"
    else v="VERDICT: INTERRUPTED by $sig during run $n (no chip loss in its log yet) | evidence $d | cfg=$CFG"; fi
  fi
  echo "$v" > "$OUT/VERDICT"; echo "$v" 2>/dev/null; say "verdict also in $OUT/VERDICT" 2>/dev/null
  exit 3
}
sig_traps
[ "$FMAX" != 0 ] && trap 'hb_stop; [ $n -gt 0 ] && fmax_stock; dropped_tray_line' EXIT   # a capped test leaves no cap behind (the cap is for the test only)
for q in $QUIETS; do
  for t in $TRIES; do
    for att in 1 2 3; do                       # an ERROR (not a drop) is retried, at most 3 attempts per run
      collision_gate wait
      precheck $n
      idle_clocks
      n=$((n + 1)); d=$OUT/run$(printf %02d $n)_q${q}_t${t}
      say "run $n: quiet $q min, try $t (attempt $att): $(( q * 30 )) s cold quiet, then $(( q * 30 )) s open idle, then the burst"
      t0=$(date -u +%FT%TZ)
      # setsid: the run is its own process group (pgid = $rp), so on_signal's TERM reaches all of it; an async child
      # of this script ignores SIGINT, so a SIGINT to the job's leftovers would not stop it
      # the quiet split in half: Q/2 cold before the mesh opens, Q/2 open idle; a user --idle-s in PASS wins
      setsid -w "$RR" --out "$d" --wait-quiet-min "$(half_min "$q")" -- --load "$LOAD" --fmax "$FMAX_ARG" --idle-s $(( q * 30 )) \
        "${PASS[@]}" > "$d.log" 2>&1 &
      rp=$!; s0=$(date +%s); RUNNING=1; k=0
      while kill -0 "$rp" 2>/dev/null; do   # the run log read every 2 s for a FIRST LOSS (early verdict), and a
        sleep 2 & wait $!                     # progress line every 10 reads (about 20 s): a job runner (e.g. the device broker)
        kill -0 "$rp" 2>/dev/null || break    # reaps a job that prints nothing for about a minute as hung (wait: a signal is
        early_verdict                         # handled at once)
        k=$((k + 1)); [ $((k % 10)) = 0 ] || continue
        last=$(grep -a -E '^(STRESS|REPRO)[|]' "$d.log" 2>/dev/null | tail -1 | cut -d'|' -f3- | cut -c1-150)
        say "run $n: in progress, $(( $(date +%s) - s0 )) s: ${last:-starting}"
      done
      wait "$rp"; rc=$?; RUNNING=0; SINCE=$(date +%s); v=$(grep -a '^VERDICT:' "$d.log" | tail -1); v=${v:-"VERDICT: ERROR (no verdict, rc $rc)"}
      case $rc in
        3) case $v in *" loss=CHIP_HANG"*) res=CHIP_HANG ;; *" loss=DROP"*) res=DROP ;; *) res=UNKNOWN ;; esac ;;
        0) case $v in *"CHIP LOSS"*) res=DROP ;; *) res=CLEAN ;; esac ;;
        2) [ "$(answering)" = 32 ] && res=ERROR || res=DROP ;;   # refused: quiet never came, or chips went missing
        4) res=CCL_HANG ;;                                         # --load allgather: a CCL op hung, every chip present
        *) res=ERROR ;;
      esac
      if [ $res = ERROR ] || { [ $res = DROP ] && [ $rc = 2 ]; }; then   # the run died before its own verdict (e.g. no
        fl=$(grep -a 'FIRST LOSS' "$d.log" 2>/dev/null | tail -1 | cut -d'|' -f3-)   # result JSON after a hung MMIO read):
        if [ -n "$fl" ]; then                                                       # classify from its FIRST LOSS line
          ch=$(echo "$fl" | sed -n 's/.*chip(s) \[\([^]]*\)\].*/\1/p')
          res=$(fl_class "$fl")
          v="VERDICT: $res from the run log (${v#VERDICT: }): $fl loss=$res"
        fi
      fi
      if [ "$EARLY" = $n ]; then   # the verdict was written early: only the run's own DROP/CHIP_HANG replaces it
        case $res in DROP|CHIP_HANG|UNKNOWN) say "run $n: ended, rc $rc, with its own $res verdict: it replaces the early $ECLS one (same row)" ;;
          *) say "run $n: ended, rc $rc: ${v#VERDICT: }; the early verdict stands"; stats; nreport; echo "$FINAL"; exit "$FINAL_RC" ;; esac
      fi
      lc=""; [[ $v =~ \ loss=([A-Z_]+) ]] && lc=${BASH_REMATCH[1]}; { [ $res = DROP ] || [ $res = UNKNOWN ]; } && [ -z "$lc" ] && lc=$res
      CC=-; [ "$FMAX" != 0 ] && { cap_check "$d.log"; [ $CC = ok ] || say "run $n: cap-unverified (requested $FMAX MHz): $CAPLINE"; }
      row="$n,$q,$t,$att,$t0,$(date -u +%FT%TZ),$rc,$res,\"${v//\"/}\",$d,$lc,${MM:-none},$FMAX,$CFG,$CC,$DEVP"
      [ "$EARLY" = $n ] && defer_sigs   # in place of the early row (the last one), still one row per run
      if [ "$EARLY" = $n ]; then { head -n -1 "$CSV"; echo "$row"; } > "$CSV.tmp" && mv -f "$CSV.tmp" "$CSV"; else echo "$row" >> "$CSV"; fi
      FINAL=""
      case $res in
        DROP) final 3 "VERDICT: DROP at quiet $q min, try $t (run $n): ${v#VERDICT: } | evidence $d | cfg=$CFG" ;;
        UNKNOWN) final 3 "VERDICT: UNKNOWN at quiet $q min, try $t (run $n): ${v#VERDICT: } | evidence $d | cfg=$CFG" ;;
        CHIP_HANG) final 3 "VERDICT: CHIP_HANG at quiet $q min, try $t (run $n): ${v#VERDICT: } | evidence $d | cfg=$CFG" ;;
        CCL_HANG) final 7 "VERDICT: CCL_HANG at quiet $q min, try $t (run $n): ${v#VERDICT: } | evidence $d | cfg=$CFG" ;;
        CLEAN) [ $CC = cap-unverified ] && final 4 "VERDICT: CAP-UNVERIFIED at quiet $q min, try $t (run $n): the $FMAX MHz cap was not read back on every chip; not a PASS | evidence $d | cfg=$CFG" ;;
        *) [ $att = 3 ] && final 1 "VERDICT: ERROR (quiet $q min, try $t failed 3 times; see $d.log) | cfg=$CFG" ;;
      esac
      [ "$EARLY" = $n ] && { echo "$FINAL" > "$OUT/VERDICT"; undefer_sigs; }
      say "run $n: $res - ${v#VERDICT: }"
      stats
      [ -n "$FINAL" ] && { nreport; echo "$FINAL"; exit $FINAL_RC; }
      [ $res = CLEAN ] && break
    done
  done
done
if [ $ONCE = 1 ]; then final 0 "VERDICT: CLEAN (one run at quiet $ONCE_Q min, --once; not a PASS) | $CSV | cfg=$CFG"
elif [ $SCREEN = 1 ]; then final 0 "VERDICT: PASS (screen: 3/3 clean runs at quiet 1 min) | $CSV | cfg=$CFG"
else final 0 "VERDICT: PASS (8/8 clean runs at quiet 1 and 3 min) | $CSV | cfg=$CFG"; fi
[ $NCOL = 0 ] || say "$NCOL collision(s) with other broker jobs, each waited out before its run: $OUT/collisions.jsonl"
nreport; echo "$FINAL"
exit 0
