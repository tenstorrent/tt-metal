#!/usr/bin/env bash
# selftest_sweep_wrapper_lib.sh — proves the evidence-root collision guard
# refuses foreign-pin/unknown roots and passes same-pin resumes, and that the
# --prev-run clean chain skips contaminated/quarantined roots.  Synthetic
# roots only; no conf, no device, no toolchain.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=sweep_wrapper_lib.sh
source "$HERE/sweep_wrapper_lib.sh" || { echo "FATAL: cannot source sweep_wrapper_lib.sh"; exit 2; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/selftest-wrapper-lib.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
PIN_A="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
PIN_B="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
FAILS=0
say() { echo "  $1"; }
check() { # check NAME EXPECTED_RC ACTUAL_RC
  if [ "$2" = "$3" ]; then say "PASS: $1"; else say "FAIL: $1 (want rc=$2 got rc=$3)"; FAILS=$((FAILS+1)); fi
}

echo "== evidence_root_guard =="

# T1: fresh (absent) root -> proceed + PIN_STAMP written
EV="$TMP/weekly-20990101"
evidence_root_guard "$EV" "$PIN_A" "selftest" >/dev/null 2>&1; rc=$?
check "fresh root proceeds" 0 "$rc"
if [ "$(head -n1 "$EV/PIN_STAMP" 2>/dev/null)" = "$PIN_A" ]; then say "PASS: PIN_STAMP recorded"; else say "FAIL: PIN_STAMP missing/wrong"; FAILS=$((FAILS+1)); fi

# T2: same root, same pin -> proceed (idempotent resume)
touch "$EV/some-artifact.txt"
evidence_root_guard "$EV" "$PIN_A" "selftest" >/dev/null 2>&1; rc=$?
check "same-pin root resumes" 0 "$rc"

# T3: same root, FOREIGN pin -> refuse rc 3, message names both pins + suggests a SWEEP_DATE
msg=$(evidence_root_guard "$EV" "$PIN_B" "selftest" 2>&1); rc=$?
check "foreign-pin root refused" 3 "$rc"
case "$msg" in
  *COLLISION*"$PIN_A"*"$PIN_B"*SWEEP_DATE=20990101b*) say "PASS: refusal message is loud + suggests free suffixed root" ;;
  *) say "FAIL: refusal message incomplete: $msg"; FAILS=$((FAILS+1)) ;;
esac

# T4: suggestion skips existing suffixed roots
mkdir -p "$TMP/weekly-20990101b"
msg=$(evidence_root_guard "$EV" "$PIN_B" "selftest" 2>&1)
case "$msg" in
  *SWEEP_DATE=20990101c*) say "PASS: suggestion skips taken suffix (b -> c)" ;;
  *) say "FAIL: suggestion did not advance past taken suffix: $msg"; FAILS=$((FAILS+1)) ;;
esac

# T5: root with only sweep_2x2.py's preflight.json (no PIN_STAMP) — both directions
EV2="$TMP/nightly-20990102"
mkdir -p "$EV2"
printf '{"cc1plus_sha256": "%s", "other": 1}\n' "$PIN_A" > "$EV2/preflight.json"
evidence_root_guard "$EV2" "$PIN_A" "selftest" >/dev/null 2>&1; rc=$?
check "preflight.json same-pin resumes" 0 "$rc"
if [ -f "$EV2/PIN_STAMP" ]; then say "PASS: stamp backfilled from preflight.json"; else say "FAIL: stamp not backfilled"; FAILS=$((FAILS+1)); fi
EV3="$TMP/nightly-20990103"
mkdir -p "$EV3"
printf '{"cc1plus_sha256": "%s"}\n' "$PIN_B" > "$EV3/preflight.json"
evidence_root_guard "$EV3" "$PIN_A" "selftest" >/dev/null 2>&1; rc=$?
check "preflight.json foreign-pin refused" 3 "$rc"

# T6: non-empty root with NO pin record -> fail closed
EV4="$TMP/weekly-20990104"
mkdir -p "$EV4"
touch "$EV4/orphan.csv"
msg=$(evidence_root_guard "$EV4" "$PIN_A" "selftest" 2>&1); rc=$?
check "unknown-provenance root refused (fail closed)" 3 "$rc"
case "$msg" in
  *"NO pin record"*) say "PASS: refusal names the missing provenance" ;;
  *) say "FAIL: unknown-provenance message wrong: $msg"; FAILS=$((FAILS+1)) ;;
esac

# T7: empty pin (conf not sourced) -> refuse
evidence_root_guard "$TMP/weekly-20990105" "" "selftest" >/dev/null 2>&1; rc=$?
check "empty pin refused" 3 "$rc"

echo "== newest_clean_runs =="
ROOT="$TMP/ev"
mkdir -p "$ROOT/weekly-20330201" "$ROOT/weekly-20330202" \
         "$ROOT/weekly-20330203-CONTAMINATED-pinX" "$ROOT/nightly-20330204" \
         "$ROOT/weekly-20330205" "$ROOT/weekly-20330206"
mkdir -p "$ROOT/weekly-20330205"; touch "$ROOT/weekly-20330205/CONTAMINATION-NOTE.md"
# mtime order, oldest -> newest: 20330201, 20330202, nightly-20330204, 20330206(current)
# Fixture dates are deliberately PRE-2038: the original 2099 mtimes clamped
# on a y2038-limited filesystem (this box's xfs/tmpfs), scrambling the
# ordering and FAILing 3 cases at every wrapper preflight (ledger 18 smell).
touch -t 203302010000 "$ROOT/weekly-20330201"
touch -t 203302020000 "$ROOT/weekly-20330202"
touch -t 203302030000 "$ROOT/weekly-20330203-CONTAMINATED-pinX"
touch -t 203302040000 "$ROOT/nightly-20330204"
touch -t 203302050000 "$ROOT/weekly-20330205"
touch -t 203302060000 "$ROOT/weekly-20330206"
CUR="$ROOT/weekly-20330206"

got=$(newest_clean_runs "$ROOT" "$CUR" 3 weekly nightly headline)
want="$ROOT/nightly-20330204,$ROOT/weekly-20330202,$ROOT/weekly-20330201"
if [ "$got" = "$want" ]; then say "PASS: chain = newest-first, skips current + name-contaminated + note-contaminated"; else say "FAIL: chain wrong: got '$got' want '$want'"; FAILS=$((FAILS+1)); fi

got=$(newest_clean_runs "$ROOT" "$CUR" 1 weekly nightly headline)
if [ "$got" = "$ROOT/nightly-20330204" ]; then say "PASS: N=1 single path, no comma"; else say "FAIL: N=1 wrong: '$got'"; FAILS=$((FAILS+1)); fi

got=$(newest_clean_runs "$ROOT" "$CUR" 2 weekly nightly headline)
if [ "$got" = "$ROOT/nightly-20330204,$ROOT/weekly-20330202" ]; then say "PASS: N caps the chain"; else say "FAIL: N=2 wrong: '$got'"; FAILS=$((FAILS+1)); fi

# QUARANTINED marker file
touch "$ROOT/nightly-20330204/QUARANTINED"
got=$(newest_clean_runs "$ROOT" "$CUR" 3 weekly nightly headline)
if [ "$got" = "$ROOT/weekly-20330202,$ROOT/weekly-20330201" ]; then say "PASS: QUARANTINED marker skipped"; else say "FAIL: QUARANTINED not skipped: '$got'"; FAILS=$((FAILS+1)); fi

# empty candidate set
got=$(newest_clean_runs "$TMP/empty-root" "$CUR" 3 weekly)
if [ -z "$got" ]; then say "PASS: no candidates -> empty output"; else say "FAIL: expected empty, got '$got'"; FAILS=$((FAILS+1)); fi

echo ""
if [ "$FAILS" -eq 0 ]; then
  echo "sweep_wrapper_lib self-test: ALL PASS"
  exit 0
fi
echo "sweep_wrapper_lib self-test: $FAILS FAILURE(S)"
exit 1
