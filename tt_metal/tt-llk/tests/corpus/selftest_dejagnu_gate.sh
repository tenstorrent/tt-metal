#!/usr/bin/env bash
# Self-test for corpus/dejagnu_gate.sh (defect D2, PULL_ANALYSIS-20260817 §2c).
#
# Proves, against the REAL gate script (not a re-implementation):
#   1. a CLEAN suite (PASSes, zero FAILs) gates GREEN  (rc 0)  — the pre-fix
#      code returned RED here (grep -c prints 0 AND exits 1, "0\n0" fallout);
#   2. a FAILING suite gates RED (rc 1);
#   3. a run that produces no g++.sum gates RED (rc 1), never GREEN.
# Sweep-hardening round 2 regression cases (adversarial review 2026-08-16):
#   4. UNRESOLVED/ERROR/XPASS outcomes gate RED even with FAIL:0;
#   5. a second suite whose runtest writes nothing must be NO_SUM RED — the
#      previous suite's leftover g++.sum must never be counted for it;
#   6. resuming a RED summary stays RED (rc 1), never converts to GREEN;
#   7. resuming a GREEN summary stays GREEN (rc 0);
#   8. suite patterns are NEVER pathname-expanded in the caller's cwd (a
#      file matching 'fixturesuite*' must not rewrite the suite list);
#   9. resuming a PARTIAL summary (fewer suite lines than patterns) is RED.
#
# Uses a stub `runtest` on PATH that writes a fixture g++.sum, so no compiler
# build or DejaGnu installation is needed.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
GATE="$HERE/dejagnu_gate.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Stub toolchain layout the gate script requires.
export DEJAGNU_BUILD_TREE="$TMP/build"
export SFPI_GCC_SRC="$TMP/src"
export DEJAGNU_SUITES="fixturesuite*"
mkdir -p "$DEJAGNU_BUILD_TREE/gcc/testsuite/g++" "$SFPI_GCC_SRC/gcc/testsuite"
echo "# stub site.exp" > "$DEJAGNU_BUILD_TREE/gcc/testsuite/g++/site.exp"
printf '#!/bin/sh\nexit 0\n' > "$DEJAGNU_BUILD_TREE/gcc/xg++"
chmod +x "$DEJAGNU_BUILD_TREE/gcc/xg++"

# Stub runtest: writes $FIXTURE_SUM into ./g++.sum (or nothing when unset).
# FIXTURE_WRITE_FOR (comma list of suite patterns, default 'all') limits
# which suites get a .sum — case 5 uses it to model a runtest that crashes
# before writing anything for the second suite.
mkdir -p "$TMP/bin"
cat > "$TMP/bin/runtest" <<'EOS'
#!/bin/sh
pat=""
for a in "$@"; do case "$a" in rvtt.exp=*) pat=${a#rvtt.exp=} ;; esac; done
case ",${FIXTURE_WRITE_FOR:-all}," in
  *,all,*|*,"$pat",*) ;;
  *) exit 0 ;;
esac
if [ -n "${FIXTURE_SUM:-}" ]; then printf '%b\n' "$FIXTURE_SUM" > g++.sum; fi
exit 0
EOS
chmod +x "$TMP/bin/runtest"
export PATH="$TMP/bin:$PATH"

overall=0
check() { # check <name> <expected-rc> <actual-rc>
  if [ "$2" -eq "$3" ]; then
    echo "SELFTEST PASS: $1 (rc=$3 as expected)"
  else
    echo "SELFTEST FAIL: $1 (expected rc=$2, got rc=$3)"
    overall=1
  fi
}

# 1. clean suite -> GREEN
FIXTURE_SUM='PASS: rvtt/fixture-a.C\nPASS: rvtt/fixture-b.C' \
  "$GATE" "$TMP/ev-clean" --force > "$TMP/out-clean.log" 2>&1
check "clean suite gates GREEN" 0 $?
grep -q "FAIL:0" "$TMP/ev-clean/dejagnu/summary.txt" || { echo "SELFTEST FAIL: clean summary lacks FAIL:0"; overall=1; }

# 2. failing suite -> RED
FIXTURE_SUM='PASS: rvtt/fixture-a.C\nFAIL: rvtt/fixture-b.C' \
  "$GATE" "$TMP/ev-fail" --force > "$TMP/out-fail.log" 2>&1
check "failing suite gates RED" 1 $?
grep -q "FAIL:1" "$TMP/ev-fail/dejagnu/summary.txt" || { echo "SELFTEST FAIL: failing summary lacks FAIL:1"; overall=1; }

# 3. no .sum produced -> RED
FIXTURE_SUM='' "$GATE" "$TMP/ev-nosum" --force > "$TMP/out-nosum.log" 2>&1
check "missing g++.sum gates RED" 1 $?

# 4. UNRESOLVED/ERROR/XPASS with FAIL:0 -> RED (defect: only ^PASS/^FAIL
#    were counted; a mid-suite compiler ICE gated GREEN).
FIXTURE_SUM='PASS: rvtt/fixture-a.C\nUNRESOLVED: rvtt/fixture-b.C compilation failed\nERROR: tcl error sourcing rvtt.exp\nXPASS: rvtt/fixture-c.C' \
  "$GATE" "$TMP/ev-badclass" --force > "$TMP/out-badclass.log" 2>&1
check "UNRESOLVED/ERROR/XPASS gate RED" 1 $?
grep -q "UNRESOLVED:1" "$TMP/ev-badclass/dejagnu/summary.txt" || { echo "SELFTEST FAIL: badclass summary lacks UNRESOLVED:1"; overall=1; }
grep -q "XPASS:1" "$TMP/ev-badclass/dejagnu/summary.txt" || { echo "SELFTEST FAIL: badclass summary lacks XPASS:1"; overall=1; }

# 5. two suites, second writes no .sum -> NO_SUM RED for the SECOND suite
#    (defect: the first suite's leftover g++.sum was counted for it).
DEJAGNU_SUITES="suiteA suiteB" FIXTURE_WRITE_FOR="suiteA" \
  FIXTURE_SUM='PASS: rvtt/fixture-a.C' \
  "$GATE" "$TMP/ev-leftover" --force > "$TMP/out-leftover.log" 2>&1
check "leftover .sum never counted for a suite that wrote nothing" 1 $?
grep -Pq "^suiteB\tNO_SUM" "$TMP/ev-leftover/dejagnu/summary.txt" || { echo "SELFTEST FAIL: suiteB not recorded NO_SUM"; overall=1; }
grep -Pq "^suiteA\tPASS:1" "$TMP/ev-leftover/dejagnu/summary.txt" || { echo "SELFTEST FAIL: suiteA not recorded PASS:1"; overall=1; }

# 6. resume of a RED summary stays RED (defect: the resume branch echoed
#    'resume' and exited 0, converting weekly REDs into GREEN on rerun).
FIXTURE_SUM='PASS: rvtt/fixture-a.C\nFAIL: rvtt/fixture-b.C' \
  "$GATE" "$TMP/ev-resume-red" --force > /dev/null 2>&1
FIXTURE_SUM='PASS: rvtt/fixture-a.C\nFAIL: rvtt/fixture-b.C' \
  "$GATE" "$TMP/ev-resume-red" > "$TMP/out-resume-red.log" 2>&1
check "resume of a RED summary stays RED" 1 $?

# 7. resume of a GREEN summary stays GREEN.
FIXTURE_SUM='PASS: rvtt/fixture-a.C' \
  "$GATE" "$TMP/ev-resume-green" --force > /dev/null 2>&1
FIXTURE_SUM='PASS: rvtt/fixture-a.C' \
  "$GATE" "$TMP/ev-resume-green" > "$TMP/out-resume-green.log" 2>&1
check "resume of a GREEN summary stays GREEN" 0 $?

# 8. suite patterns must reach runtest LITERALLY: a file matching the
#    pattern in the caller's cwd must not rewrite the suite list (defect:
#    unquoted $DEJAGNU_SUITES with globbing enabled).
mkdir -p "$TMP/globtrap" && : > "$TMP/globtrap/fixturesuiteEVIL"
( cd "$TMP/globtrap" && FIXTURE_SUM='PASS: rvtt/fixture-a.C' \
    "$GATE" "$TMP/ev-glob" --force > "$TMP/out-glob.log" 2>&1 )
check "glob-trap run still GREEN" 0 $?
grep -Fq 'fixturesuite*' "$TMP/ev-glob/dejagnu/summary.txt" || { echo "SELFTEST FAIL: summary lost the literal pattern (glob expansion in caller cwd)"; overall=1; }
grep -q "fixturesuiteEVIL" "$TMP/ev-glob/dejagnu/summary.txt" && { echo "SELFTEST FAIL: glob-trap filename leaked into the suite list"; overall=1; }

# 9. resume of a PARTIAL summary (fewer suite lines than patterns) is RED.
DEJAGNU_SUITES="suiteA" FIXTURE_SUM='PASS: rvtt/fixture-a.C' \
  "$GATE" "$TMP/ev-partial" --force > /dev/null 2>&1
DEJAGNU_SUITES="suiteA suiteB" FIXTURE_SUM='PASS: rvtt/fixture-a.C' \
  "$GATE" "$TMP/ev-partial" > "$TMP/out-partial.log" 2>&1
check "resume of a partial summary is RED" 1 $?

if [ "$overall" -eq 0 ]; then
  echo "dejagnu-gate self-test: ALL GREEN (clean->GREEN, failing->RED, no-sum->RED, badclass->RED, leftover-sum->RED, resume-red->RED, resume-green->GREEN, glob-literal, partial-resume->RED)"
else
  echo "dejagnu-gate self-test: FAILED"
fi
exit $overall
