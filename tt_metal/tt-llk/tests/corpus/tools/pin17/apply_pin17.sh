#!/usr/bin/env bash
# apply_pin17.sh — pin-16 conf ceremony, mechanized (lane EN).
#
# Applies tools/pin17/pin17-conf.patch (PIN HISTORY #17 + CURRENT PIN
# prose, KNOBS/KNOB_MODES on-plus legs for delivery-shape / record-hoist /
# prera / round-interleave / store-fold / int-not [reassoc rides the merged
# lane-EJ licensed leg], baseline header entry 16) and substitutes the
# ceremony's real values for the three placeholders:
#
#   __CC1PLUS16__  — sha256 of the INSTALLED pin-16 cc1plus (64-hex)
#   __DRIVER16__   — sha256 of the INSTALLED pin-16 driver (64-hex)
#   __SFPIGCC16__  — the sfpi-gcc pin-16 union commit
#
# then re-runs conf-lint and the harness selftests.  The ON set is
# UNCHANGED at pin 16 by design — this script cannot promote a flag.
#
# Usage (after pin-install-fast has installed the union binaries):
#   tools/pin17/apply_pin17.sh \
#       --cc1plus-sha <64hex> --driver-sha <64hex> --sfpi-gcc-commit <union tip>
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)          # .../corpus/tools/pin17
CORPUS=$(cd "$HERE/../.." && pwd)            # .../corpus
PATCH="$HERE/pin17-conf.patch"

CC1='' DRV='' GCC='' DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --cc1plus-sha) CC1=$2; shift 2 ;;
    --driver-sha) DRV=$2; shift 2 ;;
    --sfpi-gcc-commit) GCC=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "apply_pin17: unknown arg $1" >&2; exit 2 ;;
  esac
done
[[ "$CC1" =~ ^[0-9a-f]{64}$ ]] || { echo "apply_pin17: --cc1plus-sha must be a full 64-hex lowercase sha256 (got '$CC1')" >&2; exit 2; }
[[ "$DRV" =~ ^[0-9a-f]{64}$ ]] || { echo "apply_pin17: --driver-sha must be a full 64-hex lowercase sha256 (got '$DRV')" >&2; exit 2; }
[[ "$GCC" =~ ^[0-9a-f]{7,40}$ ]] || { echo "apply_pin17: --sfpi-gcc-commit must be a 7-40 hex commit (got '$GCC')" >&2; exit 2; }
[ "$CC1" != "$DRV" ] || { echo "apply_pin17: cc1plus and driver shas are identical — one of them is wrong" >&2; exit 2; }
[ -f "$PATCH" ] || { echo "apply_pin17: missing $PATCH" >&2; exit 2; }

cd "$CORPUS/../../../.."   # repo root (patch paths are repo-relative)
FILES=(
  tt_metal/tt-llk/tests/corpus/sweep_2x2.conf
  tt_metal/tt-llk/tests/corpus/sweep_2x2.py
  tt_metal/tt-llk/tests/corpus/sfpu_device_baseline_p150_v1.tsv
)

if grep -q "^_REVIEWED_CC1PLUS_SHA256=$CC1\$" "${FILES[0]}"; then
  echo "apply_pin17: pin 16 already applied AND substituted with these shas — re-running the gates only"
  SKIP_SUBST=1
elif grep -q '__CC1PLUS16__' "${FILES[0]}"; then
  echo "apply_pin17: patch already applied (placeholders present) — substituting only"
else
  git apply --check "$PATCH" || { echo "apply_pin17: patch no longer applies — the conf moved since lane EN drafted it; regenerate" >&2; exit 2; }
  if [ "$DRY" = 1 ]; then echo "apply_pin17: DRY RUN — patch applies cleanly; stopping before any write"; exit 0; fi
  git apply "$PATCH"
  echo "apply_pin17: patch applied"
fi
[ "$DRY" = 1 ] && { echo "apply_pin17: DRY RUN — stopping before substitution"; exit 0; }

if [ "${SKIP_SUBST:-0}" != 1 ]; then
  sed -i -e "s/__CC1PLUS16__/$CC1/g" -e "s/__DRIVER16__/$DRV/g" -e "s/__SFPIGCC16__/$GCC/g" "${FILES[@]}"
  for ph in __CC1PLUS16__ __DRIVER16__ __SFPIGCC16__; do
    if grep -rq "$ph" "${FILES[@]}"; then
      echo "apply_pin17: FATAL — placeholder $ph survived substitution" >&2; exit 2
    fi
  done
  echo "apply_pin17: placeholders substituted (cc1plus ${CC1:0:12}…, driver ${DRV:0:12}…, sfpi-gcc $GCC)"
fi

echo "apply_pin17: running conf-lint + harness selftests on the substituted tree"
bash "$CORPUS/conf_lint.sh"
python3 "$CORPUS/selftest_dst_layout_32b.py" >/dev/null && echo "apply_pin17: selftest_dst_layout_32b GREEN"
python3 "$CORPUS/selftest_knob_legs_semleg.py" >/dev/null && echo "apply_pin17: selftest_knob_legs_semleg GREEN"
python3 "$CORPUS/selftest_reassoc_license.py" >/dev/null && echo "apply_pin17: selftest_reassoc_license GREEN"
python3 - <<PY
import importlib.util
spec = importlib.util.spec_from_file_location('sweep_2x2', '$CORPUS/sweep_2x2.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for k in ('delivery-shape', 'record-hoist', 'prera', 'round-interleave',
          'reassoc', 'store-fold', 'int-not'):
    assert m.knob_mode(k) == 'on-plus', k
    dict(m.knob_legs(k))
assert 'reassoc' in m.LICENSED_KNOBS
print('apply_pin17: KNOBS/KNOB_MODES import-time validation GREEN')
PY

cat <<DONE
apply_pin17: DONE.  Remaining ceremony steps (in order):
  1. git add ${FILES[*]} && commit (pin prose + values SAME commit; cite the union gate evidence)
  2. REVIEW_RECORD-${CC1:0:12}.md in corpus/review_records/ AND ~/sfpi-uplift/sweep-2x2/ ('## Reviewed'/'## Gates' + full 64-hex sha)
  3. python3 $CORPUS/witness_preflight.py --work /tmp/witness-pin17 (all 25 reviewed rows, INSTALLED binary)
  4. reviewed baseline refresh for the pin-15 YELLOW stale rows (cite headline-pin15-20260820 ROW-VERDICTs)
DONE
