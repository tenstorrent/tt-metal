#!/usr/bin/env bash
# Quick smoke-test for post_bash_hook.py.
# Creates a temp LOG_DIR with a minimal run.json, feeds payloads via stdin,
# then asserts that counters and artifact files land correctly.
#
# Usage:  bash codegen/hooks/test_post_bash_hook.sh

set -euo pipefail

HOOK="$(cd "$(dirname "$0")" && pwd)/post_bash_hook.py"
PASS=0
FAIL=0

# ── helpers ────────────────────────────────────────────────────────────────

setup() {
    LOG_DIR=$(mktemp -d)
    cat > "$LOG_DIR/run.json" <<'JSON'
{
  "compile_failures": 0,
  "compile_successes": 0,
  "run_successes": 0,
  "run_failures": 0
}
JSON
    # Write state file so the hook can find LOG_DIR
    cat > /tmp/codegen_run_state.sh <<EOF
export LOG_DIR="$LOG_DIR"
EOF
}

teardown() {
    rm -rf "$LOG_DIR"
    rm -f /tmp/codegen_run_state.sh
}

# Feed a JSON payload to the hook; capture stdout
run_hook() {
    python3 "$HOOK" <<< "$1"
}

# Read a field from run.json
field() {
    python3 -c "import json,sys; d=json.load(open('$LOG_DIR/run.json')); print(d.get('$1', 'MISSING'))"
}

ok() { echo "  PASS  $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }

assert_eq() {
    local label="$1" got="$2" want="$3"
    if [ "$got" = "$want" ]; then ok "$label"; else fail "$label (got=$got want=$want)"; fi
}

assert_file() {
    local label="$1" pattern="$2"
    if ls "$LOG_DIR"/$pattern 2>/dev/null | grep -q .; then ok "$label"; else fail "$label (no match for $pattern)"; fi
}

assert_no_file() {
    local label="$1" pattern="$2"
    if ls "$LOG_DIR"/$pattern 2>/dev/null | grep -q .; then fail "$label (unexpected match for $pattern)"; else ok "$label"; fi
}

# Wrap a command+output into the PostToolUse JSON the hook expects.
# $1 = command, $2 = output (may include "Exit code: N")
make_payload() {
    python3 -c "
import json, sys
print(json.dumps({
    'tool_name': 'Bash',
    'tool_input': {'command': sys.argv[1]},
    'tool_response': sys.argv[2],
}))" "$1" "$2"
}

# ── test cases ─────────────────────────────────────────────────────────────

echo "=== compiler.py — success ==="
setup
run_hook "$(make_payload \
    "CHIP_ARCH=quasar python scripts/compiler.py foo.cpp -t x" \
    "Environment OK\ncompilation successful")"
assert_eq "compile_successes +1"  "$(field compile_successes)" 1
assert_eq "compile_failures  =0"  "$(field compile_failures)"  0
assert_file "compile_success_1.txt created" "compile_success_1.txt"
assert_no_file "no compile_failure artifact" "compile_failure_*.txt"
teardown

echo ""
echo "=== compiler.py — failure ==="
setup
run_hook "$(make_payload \
    "CHIP_ARCH=quasar python scripts/compiler.py foo.cpp" \
    "foo.cpp:12: error: 'vFloat' was not declared\nExit code: 1")"
assert_eq "compile_failures  +1"  "$(field compile_failures)"  1
assert_eq "compile_successes =0"  "$(field compile_successes)" 0
assert_file "compile_failure_1.txt created" "compile_failure_1.txt"
assert_no_file "no compile_success artifact" "compile_success_*.txt"
teardown

echo ""
echo "=== run_llk_tests.sh compile — success ==="
setup
run_hook "$(make_payload \
    "bash run_llk_tests.sh compile --worktree /tmp/x --arch quasar --test test_foo.py" \
    "collected 12 items\n12 passed")"
assert_eq "compile_successes +1"  "$(field compile_successes)" 1
assert_eq "run_successes     =0"  "$(field run_successes)"     0
assert_file "compile_success_1.txt created" "compile_success_1.txt"
assert_no_file "no run_success dir" "run_success_*"
teardown

echo ""
echo "=== run_llk_tests.sh simulate — success ==="
setup
run_hook "$(make_payload \
    "bash run_llk_tests.sh simulate --worktree /tmp/x --arch quasar --test test_foo.py" \
    "156 passed")"
assert_eq "run_successes     +1"  "$(field run_successes)"     1
assert_eq "compile_successes =0"  "$(field compile_successes)" 0
assert_file "run_success_1 dir created" "run_success_1"
assert_no_file "no compile_success artifact" "compile_success_*.txt"
teardown

echo ""
echo "=== run_llk_tests.sh run — success (both counters) ==="
setup
run_hook "$(make_payload \
    "bash run_llk_tests.sh run --worktree /tmp/x --arch quasar --test test_foo.py" \
    "156 passed")"
assert_eq "compile_successes +1"  "$(field compile_successes)" 1
assert_eq "run_successes     +1"  "$(field run_successes)"     1
assert_file "compile_success_1.txt created" "compile_success_1.txt"
assert_file "run_success_1 dir created"     "run_success_1"
teardown

echo ""
echo "=== run_llk_tests.sh count — success (no counters incremented) ==="
setup
run_hook "$(make_payload \
    "bash run_llk_tests.sh count --worktree /tmp/x --arch quasar --test test_foo.py" \
    "156")"
assert_eq "compile_successes =0"  "$(field compile_successes)" 0
assert_eq "run_successes     =0"  "$(field run_successes)"     0
assert_no_file "no compile_success artifact" "compile_success_*.txt"
assert_no_file "no run_success dir"          "run_success_*"
teardown

echo ""
echo "=== run_llk_tests.sh — compile error (exit 2) ==="
setup
run_hook "$(make_payload \
    "bash run_llk_tests.sh run --worktree /tmp/x --arch quasar --test test_foo.py" \
    "ERROR: compile step failed\nExit code: 2")"
assert_eq "compile_failures  +1"  "$(field compile_failures)"  1
assert_eq "run_failures      =0"  "$(field run_failures)"      0
assert_file "compile_failure_1.txt created" "compile_failure_1.txt"
teardown

echo ""
echo "=== run_llk_tests.sh — test failure (exit 1) ==="
setup
run_hook "$(make_payload \
    "bash run_llk_tests.sh simulate --worktree /tmp/x --arch quasar --test test_foo.py" \
    "FAILED test_foo.py::test_bar\nExit code: 1")"
assert_eq "run_failures      +1"  "$(field run_failures)"      1
assert_eq "run_successes     =0"  "$(field run_successes)"     0
assert_file "failed_attempt_1 dir created" "failed_attempt_1"
teardown

echo ""
echo "=== run_llk_tests.sh — timeout (exit 3) ==="
setup
run_hook "$(make_payload \
    "bash run_llk_tests.sh simulate --worktree /tmp/x --arch quasar --test test_foo.py" \
    "Could not acquire simulator lock\nExit code: 3")"
assert_eq "run_failures      +1"  "$(field run_failures)"      1
assert_file "failed_attempt_1 dir created" "failed_attempt_1"
teardown

echo ""
echo "=== non-Bash tool is ignored ==="
setup
python3 "$HOOK" <<< '{"tool_name": "Read", "tool_input": {"file_path": "foo.py"}, "tool_response": ""}'
assert_eq "no counters changed" "$(field compile_failures)" 0
teardown

echo ""
echo "=== no state file — hook exits silently ==="
rm -f /tmp/codegen_run_state.sh
LOG_DIR=$(mktemp -d)
# Should exit 0 and produce no output
out=$(run_hook "$(make_payload "python scripts/compiler.py foo.cpp" "Exit code: 1")")
if [ -z "$out" ]; then ok "silent when no state file"; else fail "unexpected output: $out"; fi
rm -rf "$LOG_DIR"

echo ""
echo "=== accumulation — two failures then a success ==="
setup
run_hook "$(make_payload "python scripts/compiler.py foo.cpp" "error: foo\nExit code: 1")" > /dev/null
run_hook "$(make_payload "python scripts/compiler.py foo.cpp" "error: bar\nExit code: 1")" > /dev/null
run_hook "$(make_payload "python scripts/compiler.py foo.cpp" "Environment OK")" > /dev/null
assert_eq "compile_failures  =2"  "$(field compile_failures)"  2
assert_eq "compile_successes =1"  "$(field compile_successes)" 1
assert_file "compile_failure_1.txt" "compile_failure_1.txt"
assert_file "compile_failure_2.txt" "compile_failure_2.txt"
assert_file "compile_success_1.txt" "compile_success_1.txt"
teardown

# ── summary ────────────────────────────────────────────────────────────────
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
