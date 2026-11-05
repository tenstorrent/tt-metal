#!/bin/bash
# Short ACT screen, about 15 min per boot: dram_act_screen.sh limited to the loads that hit, then the GDDR EDC
# counters of every chip and instance. A reset clears the counters, so each boot's read covers that boot only.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
EDC_PROBE=${EDC_PROBE:-$REPO/ognjen/probes/18_edc_regs.py}

usage() {
  cat <<'EOF'
Usage: dram_act_quick.sh [-n BOOTS] [-o DIR] [--no-reset] [other dram_act_screen.sh options]

Per boot: reset, all8 (sdpa and sdpa16 on all 8 controllers), three random mixes on controller 3 (seeds
761442-761444 hit chip 5 of bh-glx-110-c10u08 7, 30 and 11 times; the first also ramps the matmul load),
then an EDC counter read of every chip.
Targets run ACT_SECONDS=90 in launches of ACT_LONG_LAUNCH_S=60; override either, or ACT_RANDOM_SEED and
ACT_RANDOM_DRAWS, through the environment. EDC_PROBE points at the counter reader (skipped if missing).
--no-reset applies to the first boot only. Exit 0 only if every boot is clean and EDC is zero.
EOF
}

NB=1 OUT=$REPO/repro_logs/act_quick/$(date +%Y%m%d_%H%M%S) NO_RESET=0 DRY=0 PASS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    -n|--boots) NB=$2; shift 2 ;;
    -o|--out) OUT=$2; shift 2 ;;
    --no-reset) NO_RESET=1; shift ;;
    --dry-run) DRY=1; PASS+=("$1"); shift ;;
    *) PASS+=("$1"); shift ;;
  esac
done
[[ $NB =~ ^[1-9][0-9]*$ ]] || { echo "-n needs a positive boot count, got $NB" >&2; exit 2; }

export ACT_RANDOM_SEED=${ACT_RANDOM_SEED:-761442} ACT_RANDOM_DRAWS=${ACT_RANDOM_DRAWS:-3}
export ACT_LONG_LAUNCH_S=${ACT_LONG_LAUNCH_S:-60}
chips=$(ls /dev/tenstorrent 2>/dev/null | grep -c '^[0-9]')
rc=0
for ((i = 1; i <= NB; i++)); do
  b=$OUT/boot$i
  mkdir -p "$b"
  echo "=== quick boot $i/$NB ==="
  extra=()
  [[ $i -eq 1 && $NO_RESET -eq 1 ]] && extra=(--no-reset)
  "$HERE/dram_act_screen.sh" -n 1 -p all8,random --random-always -s "${ACT_SECONDS:-90}" -o "$b" \
    --chips "$chips" "${extra[@]}" "${PASS[@]}" 2>&1 | tee -i "$b/screen.out"
  r=${PIPESTATUS[0]}
  grep -qE "^ *boot 1: .*(BAD|FAIL)" "$b/screen.out" && rc=1
  if [[ $r -ne 0 ]]; then
    echo "screen exit $r (reset failed or interrupted): stopping"
    exit $r
  fi
  [[ $DRY -eq 1 ]] && continue
  if [[ -f $EDC_PROBE ]]; then
    # Runs between boots only: the probe and an open test process would share the chips' NoC.
    (cd "$REPO" && source python_env/bin/activate && timeout 300 python3 "$EDC_PROBE" --all $(seq 0 $((chips - 1)))) \
      > "$b/edc.log" 2>&1
    grep -E "\*\*\* EDC|checked|read failed" "$b/edc.log" | sed 's/^/  edc: /'
    dirty=$(grep -oE "[0-9]+ with EDC errors" "$b/edc.log" | grep -oE "^[0-9]+")
    [[ ${dirty:-x} == 0 ]] || rc=1
    # The probe skips an instance whose read fails instead of counting it, so a failed read is not clean.
    grep -q "read failed" "$b/edc.log" && rc=1
  else
    echo "  edc: no probe at $EDC_PROBE, skipped"
  fi
done
exit $rc
