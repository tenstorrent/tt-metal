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
set -uo pipefail
EV="${1:?usage: dejagnu_gate.sh <evidence-dir> [--force]}"
FORCE="${2:-}"
: "${DEJAGNU_BUILD_TREE:?}" "${SFPI_GCC_SRC:?}" "${DEJAGNU_SUITES:?}"

DEJ="$EV/dejagnu"
mkdir -p "$DEJ"
RC=0

if [ -x "$DEJAGNU_BUILD_TREE/gcc/xg++" ] && [ -d "$SFPI_GCC_SRC/gcc/testsuite" ] \
    && command -v runtest >/dev/null 2>&1; then
  if [ -s "$DEJ/summary.txt" ] && [ "$FORCE" != "--force" ]; then
    echo "dejagnu: resume — $DEJ/summary.txt exists"
  else
    # Never clobber a build tree's evidentiary .sum: run from a scratch site dir.
    cp "$DEJAGNU_BUILD_TREE/gcc/testsuite/g++/site.exp" "$DEJ/site.exp"
    echo "set GXX_UNDER_TEST \"$DEJAGNU_BUILD_TREE/gcc/xg++ -B$DEJAGNU_BUILD_TREE/gcc/\"" >> "$DEJ/site.exp"
    : > "$DEJ/summary.txt"
    for SUITE in $DEJAGNU_SUITES; do
      ( cd "$DEJ" && runtest --tool g++ --srcdir "$SFPI_GCC_SRC/gcc/testsuite" \
          "rvtt.exp=$SUITE" > "run-$SUITE.log" 2>&1 )
      if [ ! -s "$DEJ/g++.sum" ]; then
        # A missing/empty .sum is a broken run, not a clean one.
        echo -e "$SUITE\tNO_SUM\tRED" >> "$DEJ/summary.txt"
        echo "RED: dejagnu $SUITE produced no g++.sum"
        RC=1
        continue
      fi
      PASS=$(grep -c '^PASS' "$DEJ/g++.sum" 2>/dev/null)
      FAIL=$(grep -c '^FAIL' "$DEJ/g++.sum" 2>/dev/null)
      PASS=${PASS:-0}; FAIL=${FAIL:-0}
      cp "$DEJ/g++.sum" "$DEJ/g++-$SUITE.sum" 2>/dev/null
      echo -e "$SUITE\tPASS:$PASS\tFAIL:$FAIL" >> "$DEJ/summary.txt"
      # Byte-parity suites must be zero-FAIL (loadmacro*/macro-planner* carry
      # the minmax 19+6 parities and the refusal oracles) — and must actually
      # have run: zero PASS lines means the pattern matched nothing, which can
      # never prove parity and is RED, not silently GREEN.
      if [ "$FAIL" -ne 0 ]; then
        echo "RED: dejagnu $SUITE has $FAIL FAILs"
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
