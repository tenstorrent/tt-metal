#!/usr/bin/env bash
# apply_pin15.sh — pin-15 conf ceremony, mechanized (lane DZ prep).
#
# Applies tools/pin15/pin15-conf.patch (PIN HISTORY #15 + CURRENT PIN
# prose, KNOBS/KNOB_MODES on-plus legs for list-schedule / lreg-alloc /
# milp, HEADLINE_ROWS targets, baseline header entry 15) and substitutes
# the ceremony's real values for the three placeholders:
#
#   __CC1PLUS15__  — sha256 of the INSTALLED pin-15 cc1plus (64-hex)
#   __DRIVER15__   — sha256 of the INSTALLED pin-15 driver (64-hex)
#   __SFPIGCC15__  — the sfpi-gcc pin-15 union commit
#
# then re-runs conf-lint and the harness selftests.  The ON set is
# UNCHANGED at pin 15 by design — this script cannot promote a flag.
#
# Usage (after pin-install-fast has installed the union binaries):
#   tools/pin15/apply_pin15.sh \
#       --cc1plus-sha $(sha256sum "$(tests/sfpi/compiler/bin/riscv-tt-elf-g++ -print-prog-name=cc1plus)" | cut -d' ' -f1) \
#       --driver-sha  $(sha256sum tests/sfpi/compiler/bin/riscv-tt-elf-g++ | cut -d' ' -f1) \
#       --sfpi-gcc-commit <union tip sha>
#
# Remaining ceremony steps it PRINTS but does not do (they need review /
# the device): commit (conf prose + values in the SAME commit — the pin
# rule), REVIEW_RECORD-<cc1plus12>.md in BOTH locations, witness_preflight
# on the INSTALLED binary, then MEASUREMENT-PLAN-PIN15.md.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)          # .../corpus/tools/pin15
CORPUS=$(cd "$HERE/../.." && pwd)            # .../corpus
PATCH="$HERE/pin15-conf.patch"

CC1='' DRV='' GCC='' DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --cc1plus-sha) CC1=$2; shift 2 ;;
    --driver-sha) DRV=$2; shift 2 ;;
    --sfpi-gcc-commit) GCC=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "apply_pin15: unknown arg $1" >&2; exit 2 ;;
  esac
done
[[ "$CC1" =~ ^[0-9a-f]{64}$ ]] || { echo "apply_pin15: --cc1plus-sha must be a full 64-hex lowercase sha256 (got '$CC1')" >&2; exit 2; }
[[ "$DRV" =~ ^[0-9a-f]{64}$ ]] || { echo "apply_pin15: --driver-sha must be a full 64-hex lowercase sha256 (got '$DRV')" >&2; exit 2; }
[[ "$GCC" =~ ^[0-9a-f]{7,40}$ ]] || { echo "apply_pin15: --sfpi-gcc-commit must be a 7-40 hex commit (got '$GCC')" >&2; exit 2; }
[ "$CC1" != "$DRV" ] || { echo "apply_pin15: cc1plus and driver shas are identical — one of them is wrong" >&2; exit 2; }
[ -f "$PATCH" ] || { echo "apply_pin15: missing $PATCH" >&2; exit 2; }

cd "$CORPUS/../../../.."   # repo root (patch paths are repo-relative)
FILES=(
  tt_metal/tt-llk/tests/corpus/sweep_2x2.conf
  tt_metal/tt-llk/tests/corpus/sweep_2x2.py
  tt_metal/tt-llk/tests/corpus/sfpu_device_baseline_p150_v1.tsv
)

if grep -q "^_REVIEWED_CC1PLUS_SHA256=$CC1\$" "${FILES[0]}"; then
  echo "apply_pin15: pin 15 already applied AND substituted with these shas — re-running the gates only"
  SKIP_SUBST=1
elif grep -q '__CC1PLUS15__' "${FILES[0]}"; then
  echo "apply_pin15: patch already applied (placeholders present) — substituting only"
else
  git apply --check "$PATCH" || { echo "apply_pin15: patch no longer applies — the conf moved since lane DZ drafted it; regenerate (base: agent/pin15-prep)" >&2; exit 2; }
  if [ "$DRY" = 1 ]; then echo "apply_pin15: DRY RUN — patch applies cleanly; stopping before any write"; exit 0; fi
  git apply "$PATCH"
  echo "apply_pin15: patch applied"
fi
[ "$DRY" = 1 ] && { echo "apply_pin15: DRY RUN — stopping before substitution"; exit 0; }

if [ "${SKIP_SUBST:-0}" != 1 ]; then
  sed -i -e "s/__CC1PLUS15__/$CC1/g" -e "s/__DRIVER15__/$DRV/g" -e "s/__SFPIGCC15__/$GCC/g" "${FILES[@]}"
  for ph in __CC1PLUS15__ __DRIVER15__ __SFPIGCC15__; do
    if grep -rq "$ph" "${FILES[@]}"; then
      echo "apply_pin15: FATAL — placeholder $ph survived substitution" >&2; exit 2
    fi
  done
  echo "apply_pin15: placeholders substituted (cc1plus ${CC1:0:12}…, driver ${DRV:0:12}…, sfpi-gcc $GCC)"
fi

echo "apply_pin15: running conf-lint + harness selftests on the substituted tree"
bash "$CORPUS/conf_lint.sh"
python3 "$CORPUS/selftest_dst_layout_32b.py" >/dev/null && echo "apply_pin15: selftest_dst_layout_32b GREEN (knob-integration checks armed)"
python3 "$CORPUS/selftest_knob_legs_semleg.py" >/dev/null && echo "apply_pin15: selftest_knob_legs_semleg GREEN"
python3 -c "
import importlib.util
spec = importlib.util.spec_from_file_location('sweep_2x2', '$CORPUS/sweep_2x2.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for k in ('list-schedule', 'lreg-alloc', 'milp'):
    assert m.knob_mode(k) == 'on-plus', k
    dict(m.knob_legs(k))
print('apply_pin15: KNOBS/KNOB_MODES import-time validation GREEN')
"

cat <<EOF
apply_pin15: DONE.  Remaining ceremony steps (in order):
  1. git add ${FILES[*]} && commit (pin prose + values SAME commit; cite the union gate evidence)
  2. REVIEW_RECORD-${CC1:0:12}.md in corpus/review_records/ AND the sweep evidence parent (template: REVIEW_RECORD_TEMPLATE.md; needs '## Reviewed'/'## Gates' headings + the full 64-hex sha)
  3. python3 $CORPUS/witness_preflight.py --work /tmp/witness-pin15 (all 25 reviewed rows, INSTALLED binary)
  4. launch the measurement pass: $CORPUS/MEASUREMENT-PLAN-PIN15.md
EOF
