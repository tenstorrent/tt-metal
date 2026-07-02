#!/bin/bash
# ACE-Step v1.5 bring-up progress benchmark.
# Primary metric = modules_passing (count of PCC tests passing). Higher is better.
# Runs the acestep PCC suite, parses pytest output, emits METRIC lines.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

# Activate the worktree venv if present.
if [ -f python_env/bin/activate ]; then
  # shellcheck disable=SC1091
  source python_env/bin/activate
fi

PCC_DIR="models/experimental/acestep/tests/pcc"

# Fast pre-check: python can import the acestep package (syntax errors caught <1s).
python -c "import ast,glob,sys
bad=0
for f in glob.glob('models/experimental/acestep/**/*.py', recursive=True):
    try: ast.parse(open(f).read())
    except SyntaxError as e:
        print(f'SYNTAX ERROR {f}: {e}'); bad=1
sys.exit(bad)" || { echo 'METRIC modules_passing=0'; echo 'PRECHECK_FAILED syntax'; exit 0; }

# If no PCC tests exist yet, baseline is zero (bring-up not started).
if ! ls "$PCC_DIR"/test_*.py >/dev/null 2>&1; then
  echo "No PCC tests yet — baseline."
  echo "METRIC modules_passing=0"
  echo "METRIC tests_total=0"
  echo "METRIC tests_failing=0"
  echo "METRIC min_pcc=0"
  echo "METRIC avg_pcc=0"
  echo "METRIC suite_seconds=0"
  exit 0
fi

START=$(date +%s.%N)
LOG=$(mktemp)
# -p no:randomly for determinism; short tracebacks; report reasons.
pytest "$PCC_DIR" -q -rA --no-header -p no:cacheprovider 2>&1 | tee "$LOG"
END=$(date +%s.%N)
SUITE_SECONDS=$(echo "$END - $START" | bc)

# Parse pass/fail counts from pytest summary line, robust to formats.
PASSED=$(grep -oE '[0-9]+ passed' "$LOG" | tail -1 | grep -oE '[0-9]+' || echo 0)
FAILED=$(grep -oE '[0-9]+ failed' "$LOG" | tail -1 | grep -oE '[0-9]+' || echo 0)
ERRORS=$(grep -oE '[0-9]+ error' "$LOG" | tail -1 | grep -oE '[0-9]+' || echo 0)
PASSED=${PASSED:-0}; FAILED=${FAILED:-0}; ERRORS=${ERRORS:-0}
TOTAL=$((PASSED + FAILED + ERRORS))
FAILING=$((FAILED + ERRORS))

# Extract PCC values printed by comp_pcc (format: "PCC: 0.9993..." or "pcc=0.999").
PCC_VALS=$(grep -oiE 'pcc[:= ]+0\.[0-9]+' "$LOG" | grep -oE '0\.[0-9]+' || true)
if [ -n "$PCC_VALS" ]; then
  MIN_PCC=$(echo "$PCC_VALS" | sort -g | head -1)
  AVG_PCC=$(echo "$PCC_VALS" | awk '{s+=$1; n++} END{if(n>0) printf "%.6f", s/n; else print 0}')
else
  MIN_PCC=0; AVG_PCC=0
fi

# E2E pipeline PCC (STRICT gate >=0.95). Tests print "E2E_PCC: 0.xxxx"; take the minimum seen.
E2E_VALS=$(grep -oiE 'e2e_pcc[:= ]+0\.[0-9]+' "$LOG" | grep -oE '0\.[0-9]+' || true)
if [ -n "$E2E_VALS" ]; then
  E2E_PCC=$(echo "$E2E_VALS" | sort -g | head -1)
else
  E2E_PCC=0
fi

rm -f "$LOG"

echo "METRIC modules_passing=${PASSED}"
echo "METRIC tests_total=${TOTAL}"
echo "METRIC tests_failing=${FAILING}"
echo "METRIC min_pcc=${MIN_PCC}"
echo "METRIC avg_pcc=${AVG_PCC}"
echo "METRIC e2e_pcc=${E2E_PCC}"
echo "METRIC suite_seconds=${SUITE_SECONDS}"
