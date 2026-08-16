#!/usr/bin/env bash
# Self-test for corpus/dejagnu_gate.sh (defect D2, PULL_ANALYSIS-20260817 §2c).
#
# Proves, against the REAL gate script (not a re-implementation):
#   1. a CLEAN suite (PASSes, zero FAILs) gates GREEN  (rc 0)  — the pre-fix
#      code returned RED here (grep -c prints 0 AND exits 1, "0\n0" fallout);
#   2. a FAILING suite gates RED (rc 1);
#   3. a run that produces no g++.sum gates RED (rc 1), never GREEN.
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
mkdir -p "$TMP/bin"
cat > "$TMP/bin/runtest" <<'EOS'
#!/bin/sh
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

if [ "$overall" -eq 0 ]; then
  echo "dejagnu-gate self-test: ALL GREEN (clean->GREEN, failing->RED, no-sum->RED)"
else
  echo "dejagnu-gate self-test: FAILED"
fi
exit $overall
