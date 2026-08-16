#!/usr/bin/env bash
# DejaGnu byte-parity gate for the scheduled sweeps (extracted from
# weekly_bh_sweep.sh so the counting/gating logic is directly self-testable:
# selftest_dejagnu_gate.sh runs this script against synthetic clean/failing
# .sum fixtures and asserts clean->GREEN(rc 0) and failing->RED(rc 1)).
#
# Usage: dejagnu_gate.sh <evidence-dir> [--force]
# Env (from sweep_2x2.conf or the caller):
#   DEJAGNU_BUILD_TREE  built sfpi-gcc tree containing gcc/xg++
#   SFPI_GCC_SRC        gcc source dir (gcc/testsuite)
#   DEJAGNU_SUITES      space list of rvtt.exp suite patterns
#
# Defect D2 (PULL_ANALYSIS-20260817 §2c) lived here in its previous inline
# form: `FAIL=$(grep -c '^FAIL' g++.sum || echo 0)` — grep -c PRINTS 0 on a
# clean suite AND exits 1, so the `|| echo 0` appended a second line, the
# numeric test errored, and a CLEAN suite gated RED while the intended
# zero-FAIL enforcement never functioned.  grep -c never needs the fallback:
# capture stdout alone and treat an empty result (missing .sum) as its own
# RED, never as zero FAILs.
# Sweep-hardening round 2 (adversarial review, 2026-08-16) closed four more
# holes here:
#   * UNRESOLVED/XPASS/UNTESTED/UNSUPPORTED/ERROR lines were invisible — a
#     mid-suite compiler ICE (UNRESOLVED) or a tcl abort (ERROR) gated GREEN
#     with FAIL:0.  Any non-PASS outcome class (XFAIL excepted: it is the
#     EXPECTED outcome of an xfail-encoded test; the un-expected direction,
#     XPASS, is counted) now gates RED.
#   * g++.sum was never deleted between suites, so a runtest that wrote
#     nothing was counted from the PREVIOUS suite's leftover .sum — the
#     NO_SUM RED path was unreachable after the first successful suite.
#   * the resume branch echoed 'resume' and exited 0 without re-deriving the
#     verdict: any rerun of the same SWEEP_DATE converted a RED into GREEN.
#     Resume now re-derives RC from the stored summary (and REDs on a
#     partial summary — fewer suite lines than patterns).
#   * $DEJAGNU_SUITES was expanded unquoted with globbing enabled: a file
#     matching 'loadmacro*' in the caller's cwd silently rewrote the suite
#     list.  set -f keeps the patterns literal for runtest.
set -uo pipefail
set -f  # never pathname-expand the suite patterns in the caller's cwd
EV="${1:?usage: dejagnu_gate.sh <evidence-dir> [--force]}"
FORCE="${2:-}"
: "${DEJAGNU_BUILD_TREE:?}" "${SFPI_GCC_SRC:?}" "${DEJAGNU_SUITES:?}"

DEJ="$EV/dejagnu"
mkdir -p "$DEJ"
RC=0

# Non-PASS outcome classes that gate RED (XFAIL deliberately absent).
BAD_CLASSES="FAIL XPASS UNRESOLVED UNTESTED UNSUPPORTED ERROR"

if [ -x "$DEJAGNU_BUILD_TREE/gcc/xg++" ] && [ -d "$SFPI_GCC_SRC/gcc/testsuite" ] \
    && command -v runtest >/dev/null 2>&1; then
  if [ -s "$DEJ/summary.txt" ] && [ "$FORCE" != "--force" ]; then
    # Resume MUST re-derive the verdict from the stored summary — a resumed
    # RED that exits 0 silently un-triages the failure.
    echo "dejagnu: resume — re-deriving verdict from $DEJ/summary.txt"
    EXPECTED=0
    for SUITE in $DEJAGNU_SUITES; do EXPECTED=$((EXPECTED + 1)); done
    RECORDED=$(grep -c . "$DEJ/summary.txt")
    if grep -q "SKIP_NO_BUILD_TREE" "$DEJ/summary.txt"; then
      echo "RED: dejagnu resume — stored summary is a SKIP but the build tree exists now (rerun with --force)"
      RC=1
    elif [ "$RECORDED" -ne "$EXPECTED" ]; then
      echo "RED: dejagnu resume — summary has $RECORDED suite line(s), expected $EXPECTED (partial run; rerun with --force)"
      RC=1
    fi
    if grep -q "NO_SUM" "$DEJ/summary.txt"; then
      echo "RED: dejagnu resume — stored summary records a NO_SUM suite"
      RC=1
    fi
    while IFS= read -r LINE; do
      [ -n "$LINE" ] || continue
      case "$LINE" in *NO_SUM*|*SKIP_NO_BUILD_TREE*) continue ;; esac
      for CLASS in $BAD_CLASSES; do
        if printf '%s\n' "$LINE" | grep -qE "(^|[[:space:]])$CLASS:[1-9]"; then
          echo "RED: dejagnu resume — stored summary carries failures: $LINE"
          RC=1
        fi
      done
      if ! printf '%s\n' "$LINE" | grep -qE "(^|[[:space:]])PASS:[1-9]"; then
        echo "RED: dejagnu resume — stored summary has a zero-PASS suite: $LINE"
        RC=1
      fi
    done < "$DEJ/summary.txt"
  else
    # Never clobber a build tree's evidentiary .sum: run from a scratch site dir.
    cp "$DEJAGNU_BUILD_TREE/gcc/testsuite/g++/site.exp" "$DEJ/site.exp"
    echo "set GXX_UNDER_TEST \"$DEJAGNU_BUILD_TREE/gcc/xg++ -B$DEJAGNU_BUILD_TREE/gcc/\"" >> "$DEJ/site.exp"
    : > "$DEJ/summary.txt"
    for SUITE in $DEJAGNU_SUITES; do
      # A leftover .sum from the previous suite must NEVER be counted for
      # this one: delete first, so a runtest that writes nothing is NO_SUM.
      rm -f "$DEJ/g++.sum" "$DEJ/g++.log"
      ( cd "$DEJ" && runtest --tool g++ --srcdir "$SFPI_GCC_SRC/gcc/testsuite" \
          "rvtt.exp=$SUITE" > "run-$SUITE.log" 2>&1 )
      if [ ! -s "$DEJ/g++.sum" ]; then
        # A missing/empty .sum is a broken run, not a clean one.
        echo -e "$SUITE\tNO_SUM\tRED" >> "$DEJ/summary.txt"
        echo "RED: dejagnu $SUITE produced no g++.sum"
        RC=1
        continue
      fi
      PASS=$(grep -c '^PASS' "$DEJ/g++.sum" 2>/dev/null); PASS=${PASS:-0}
      LINE="$SUITE\tPASS:$PASS"
      SUITE_BAD=0
      for CLASS in $BAD_CLASSES; do
        N=$(grep -c "^$CLASS" "$DEJ/g++.sum" 2>/dev/null); N=${N:-0}
        # XPASS must not double-count into PASS, nor XFAIL into FAIL: the
        # anchored ^CLASS greps are disjoint because ^PASS cannot match an
        # XPASS line, but ^FAIL vs XFAIL needs no care either (XFAIL lines
        # start with X).  ERROR lines can also land only in the run log
        # (tcl aborts before the .sum record): count both.
        if [ "$CLASS" = "ERROR" ]; then
          NLOG=$(grep -c '^ERROR' "$DEJ/run-$SUITE.log" 2>/dev/null); NLOG=${NLOG:-0}
          [ "$NLOG" -gt "$N" ] && N=$NLOG
        fi
        LINE="$LINE\t$CLASS:$N"
        [ "$N" -ne 0 ] && SUITE_BAD=1
      done
      cp "$DEJ/g++.sum" "$DEJ/g++-$SUITE.sum" 2>/dev/null
      echo -e "$LINE" >> "$DEJ/summary.txt"
      # Byte-parity suites must have ZERO non-PASS outcomes (loadmacro*/
      # macro-planner* carry the minmax 19+6 parities and the refusal
      # oracles) — and must actually have run: zero PASS lines means the
      # pattern matched nothing, which can never prove parity.
      if [ "$SUITE_BAD" -ne 0 ]; then
        echo "RED: dejagnu $SUITE has non-PASS outcomes ($(echo -e "$LINE" | cut -f3-))"
        RC=1
      elif [ "$PASS" -eq 0 ]; then
        echo "RED: dejagnu $SUITE ran zero PASS lines (pattern matched nothing?)"
        RC=1
      fi
    done
  fi
else
  echo -e "dejagnu\tSKIP_NO_BUILD_TREE\t$DEJAGNU_BUILD_TREE" | tee "$DEJ/summary.txt"
fi
exit $RC
