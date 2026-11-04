#!/bin/bash
# Boot-loop screen for the GDDR6 ACT-word alias: reset the Galaxy, run test_dram_act_patterns.py steps, report per target.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
TEST=models/demos/deepseek_v3_d_p/tests/test_dram_act_patterns.py
SOCK=/run/ttop-ipmi/reset

usage() {
  cat <<'EOF'
Usage: dram_act_screen.sh [options]

Boot loop that screens a Blackhole Galaxy for silent wrong-address GDDR6 reads. Each boot resets the board, runs the
steps below on all 32 chips, and prints one line per target. The fault shows on some boots only, so screen several.

Run the full default loop for 5 boots:  dram_act_screen.sh -n 5

Options
  -n, --boots N         number of boots (default 1)
  -c, --ctrl N          DRAM controller for cafull and random (default 3)
  -s, --seconds S       seconds per target (default 120)
  -p, --steps LIST      comma list of all8,cafull,random (default all three)
      --random-always   run random on every boot (default: only on a boot where an earlier step hit)
      --no-reset        skip the reset before the first boot (board was just reset)
  -o, --out DIR         output directory (default <repo>/repro_logs/act_screen/<start time>)
      --dry-run         print the plan and the reset method, touch no device
  -h, --help            this text

Steps (about 2.3 h per boot, plus 50 min when random runs)
  all8    all 8 controllers, loads sdpa and sdpa16 on row 0x200. ~7 min.
  cafull  one controller, 21 targets x {sdpa, sdpa16} = 42 targets. ~2 h.
          orig        0x4003880: row 0x200, bank 1, the first address that read wrong data
          orig_bank3  same row in bank 3; same failing banks, but a stronger load (hits ~20x more often)
          caN_up/dn   19 rows where row bit R(N+4) drives ACT pin CA N against opposite-toggling neighbours
  random  one controller, 15 random read/write/barrier mixes, seed printed so a hit can be replayed. ~50 min.

How a target works
  Write   each chip's DRAM buffer is filled with a per-address hash pattern (address, controller and launch seed
          mixed together; the seed changes every launch, so data left by the previous launch is detectable).
  Load    240 data-movement RISCs (12x10 grid) read 256 B entries from a short list of per-bank addresses in a loop,
          while a matmul runs on the same cores to draw power. Address decode: col = addr[12:6],
          bank = addr[16:13], row = addr[31:17].
            sdpa    3 groups 1:1:1: a 153-read stream from row 0x200, ping-pong reads between two rows of the target
                    bank with a barrier after each (one row activation per read), and 16-read write bursts to other
                    rows of that bank.
            sdpa16  same, but the stream reads row 0x200 in all 16 banks.
  Check   every read is compared word by word with the hash it should hold. A wrong read is classified as stale
          (previous launch's data), alias of bit k (the bytes equal the data at addr ^ (1 << k), read back from
          DRAM; the known fault is bit 26 = 64 MB), or other. Results per chip, controller, lane and bank are logged.

Output
  [PASS] target clean on every chip       [HIT ] wrong data read: chip and count shown
  [FAIL] step crashed or timed out; the rest of the boot is skipped and the next boot starts with a reset
  [STOP] a reset ran and failed; the loop ends because the device state is unknown
  [WARN] no reset method on this host; boots run without a reset
  <out>/<step>_<boot>.log has the full pytest output, <out>/<step>_<boot>/ the per-target JSON.

Reset
  TTOP container (socket /run/ttop-ipmi/reset): IPMI reset via that socket. tt-smi -glx_reset_auto wedges all chips
  in this container, so it is never used there. Bare metal (/dev/ipmi0): tt-smi -glx_reset_auto. Neither: [WARN]
  and no reset; every boot then runs on the same power-up, so it repeats the test instead of sampling a new boot.
  After the reset the script checks that 32 chips enumerate; enumeration alone does not prove the chips are usable,
  so a bad reset can still show up as [FAIL] on the first step.

Examples
  dram_act_screen.sh --dry-run -n 5
  dram_act_screen.sh -n 5 2>&1 | tee screen.out
  dram_act_screen.sh --no-reset -p cafull -s 60
EOF
}

NB=1 CTRL=3 SECS=120 STEPS=all8,cafull,random RANDOM_ALWAYS=0 RESET_FIRST=1 OUT="" DRY=0
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    -n|--boots) NB=$2; shift 2 ;;
    -c|--ctrl) CTRL=$2; shift 2 ;;
    -s|--seconds) SECS=$2; shift 2 ;;
    -p|--steps) STEPS=$2; shift 2 ;;
    --random-always) RANDOM_ALWAYS=1; shift ;;
    --no-reset) RESET_FIRST=0; shift ;;
    -o|--out) OUT=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument $1 (see --help)" >&2; exit 2 ;;
  esac
done
[[ $NB =~ ^[1-9][0-9]*$ ]] || { echo "-n needs a positive boot count, got $NB" >&2; exit 2; }
OUT=${OUT:-$REPO/repro_logs/act_screen/$(date +%Y%m%d_%H%M%S)}
for s in ${STEPS//,/ }; do
  [[ $s =~ ^(all8|cafull|random)$ ]] || { echo "unknown step $s (see --help)" >&2; exit 2; }
done
has_step() { [[ ,$STEPS, == *,$1,* ]]; }

if [[ -t 1 ]]; then
  G=$'\e[32m' R=$'\e[31m' Y=$'\e[33m' B=$'\e[1m' N=$'\e[0m'
else
  G="" R="" Y="" B="" N=""
fi
say() { echo "$(date +%H:%M:%S) $*"; }

if [[ -S $SOCK ]]; then
  RESET_MODE=ttop_ipmi
elif [[ -e /dev/ipmi0 ]]; then
  RESET_MODE=glx_reset_auto
else
  RESET_MODE=none
fi

board_reset() {
  case $RESET_MODE in
    ttop_ipmi)
      python3 - "$SOCK" <<'PY'
import socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sys.argv[1])
buf = b""
while d := s.recv(4096):
    buf += d
    sys.stdout.write(d.decode(errors="replace"))
    sys.stdout.flush()
codes = [int(l.split(":")[1]) for l in buf.decode(errors="replace").splitlines() if l.startswith("__EXIT__:")]
sys.exit(codes[-1] if codes else 1)
PY
      ;;
    # The venv and PYTHONPATH (repo root) break the host tt-smi import.
    glx_reset_auto) env -u PYTHONPATH -u VIRTUAL_ENV tt-smi -glx_reset_auto ;;
  esac
}

do_reset() {  # do_reset <boot>
  local log=$OUT/reset_$1.log r n
  if [[ $RESET_MODE == none ]]; then
    say "${Y}[WARN]${N} no $SOCK socket and no /dev/ipmi0: no reset, boot $1 continues on the current power-up"
    return
  fi
  say "${B}reset${N} ($RESET_MODE) ..."
  board_reset > "$log" 2>&1
  r=$?
  n=$(ls /dev/tenstorrent 2>/dev/null | wc -l)
  if [[ $r -ne 0 || $n -ne 32 ]]; then
    say "${R}[STOP]${N} reset failed: exit $r, $n/32 chips. Last lines of $log:"
    tail -5 "$log" | sed 's/^/         /'
    say "${R}[STOP]${N} loop ended at boot $1; check the board before restarting"
    summary
    exit 1
  fi
  say "${G}reset ok${N}, 32 chips"
}

PID=""
trap 'say "${Y}interrupted${N}"; [[ -n $PID ]] && kill $PID 2>/dev/null; summary; exit 130' INT TERM

STEP_RESULT=""
run_step() {  # run_step <step> <boot> <ctrls> <patterns> <timeout_s>; sets STEP_RESULT to pass|hit|fail
  local name=$1_$2 log=$OUT/$1_$2.log r hits clean
  say "${B}$1${N} start (timeout $(($5 / 60)) min), log $log"
  ( cd "$REPO" && STRESS_ARMS=on ACT_ITERS=4000 ACT_CTRLS=$3 ACT_PATTERNS=$4 ACT_SECONDS=$SECS \
      STRESS_OUT_DIR=$OUT/$name exec timeout "$5" python3 -m pytest -svq "$TEST" -p no:randomly ) > "$log" 2>&1 &
  PID=$!
  tail -n +1 -F --pid=$PID "$log" 2>/dev/null | grep --line-buffered " arm on: " | while read -r l; do
    [[ $l =~ \ -\ (.+)\ arm\ on:\ [0-9]+\ launches,\ ([0-9.e+]+)\ reads,.*bad\ reads\ by\ chip\ (.*)$ ]] || continue
    if [[ ${BASH_REMATCH[3]} == none ]]; then
      say "  ${G}[PASS]${N} ${BASH_REMATCH[1]}  (${BASH_REMATCH[2]} reads)"
    else
      say "  ${R}[HIT ]${N} ${BASH_REMATCH[1]}  bad reads by chip ${BASH_REMATCH[3]}"
    fi
  done
  wait $PID
  r=$?
  PID=""
  hits=$(grep -c "bad reads by chip {" "$log")
  clean=$(grep -c "bad reads by chip none" "$log")
  if [[ $r -eq 124 ]]; then
    say "${R}[FAIL]${N} $1 timed out after $(($5 / 60)) min ($clean clean, $hits hit before that)"
    STEP_RESULT=fail
  elif [[ $hits -gt 0 ]]; then
    say "${R}[HIT ]${N} $1 done: $hits of $((hits + clean)) targets read wrong data"
    STEP_RESULT=hit
  elif [[ $r -ne 0 ]]; then
    say "${R}[FAIL]${N} $1 exit $r after $clean clean targets:"
    grep -E "Error|error:|FAILED|Timeout" "$log" | tail -3 | cut -c1-200 | sed 's/^/         /'
    STEP_RESULT=fail
  else
    say "${G}[PASS]${N} $1 done: $clean targets clean"
    STEP_RESULT=pass
  fi
}

SUMMARY=()
summary() {
  [[ ${#SUMMARY[@]} -eq 0 ]] && return
  echo "${B}summary${N} ($OUT)"
  printf '  %s\n' "${SUMMARY[@]}"
}

echo "${B}DRAM ACT screen${N}: $NB boot(s), steps $STEPS, ctrl $CTRL, ${SECS}s per target, reset $RESET_MODE"
echo "  repo $REPO"
echo "  out  $OUT"
if [[ $DRY -eq 1 ]]; then
  [[ $RESET_MODE == none ]] && echo "${Y}[WARN]${N} no reset method on this host; boots would run without a reset"
  echo "dry run: nothing started"
  exit 0
fi
[[ -f $REPO/$TEST ]] || { echo "${R}missing $REPO/$TEST${N}" >&2; exit 2; }
mkdir -p "$OUT"
cd "$REPO" && source python_env/bin/activate
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO TT_METAL_OPERATION_TIMEOUT_SECONDS=300

for ((i = 1; i <= NB; i++)); do
  echo
  say "${B}=== boot $i/$NB ===${N}"
  if [[ $i -gt 1 || $RESET_FIRST -eq 1 ]]; then do_reset "$i"; fi
  res="" bad=0
  for step in all8 cafull; do
    has_step $step || continue
    if [[ $step == all8 ]]; then run_step all8 "$i" all sdpa_long,sdpa16_long 1800
    else run_step cafull "$i" "$CTRL" ca_full_long 14400; fi
    res+=" $step:$STEP_RESULT"
    [[ $STEP_RESULT == hit ]] && bad=1
    [[ $STEP_RESULT == fail ]] && break
  done
  if [[ $STEP_RESULT != fail ]] && has_step random && [[ $bad -eq 1 || $RANDOM_ALWAYS -eq 1 || $STEPS == random ]]; then
    run_step random "$i" "$CTRL" random_long 5400
    res+=" random:$STEP_RESULT"
    [[ $STEP_RESULT == hit ]] && bad=1
  fi
  if [[ $STEP_RESULT == fail ]]; then verdict="${R}FAIL${N}"; elif [[ $bad -eq 1 ]]; then verdict="${R}BAD${N}"; else verdict="${G}clean${N}"; fi
  SUMMARY+=("boot $i: $verdict  $res")
  say "boot $i: $verdict ($res )"
done
echo
summary
