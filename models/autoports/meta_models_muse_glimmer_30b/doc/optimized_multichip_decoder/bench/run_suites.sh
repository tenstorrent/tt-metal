#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The acceptance gate for the optimized multichip decoder: the multichip stage's
# own two correctness modules, unchanged, run against the optimized default path.
# The correctness floor does not move when the layer gets faster ($optimize:
# "your initial functional test suite remains the correctness floor").
#
# Two pytest invocations, for the reason the multichip stage documented:
# test_multichip_decoder.py holds a session-scoped 1x4 mesh, and
# test_multichip_vs_single_chip.py opens a 1x1 mesh first; run together the 1x1
# open finds the four dies still owned and times out on an Ethernet core.
set -uo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/optimized_multichip_decoder
T=models/autoports/meta_models_muse_glimmer_30b/tests
mkdir -p "$D/logs"
# Delete the previous run's XML first.  A session that aborts at teardown -- which
# this mesh does, three times on record (work_log section 10.2) -- would otherwise
# leave a passing XML behind and the gate would read it as this run's.
rm -f "$D/test_results.xml" "$D/test_results_vs_single_chip.xml"

# Read the JUnit XML pytest already writes, not its console output.  Three
# separate defects in this stage came from grepping ANSI-coloured pytest text
# with an anchored pattern (work_log.md section 10.1); the XML has no such trap,
# and it also catches a run that never collected anything.
junit_failures () {  # $1 = junit xml path
  python - "$1" <<'PYEOF'
import sys, xml.etree.ElementTree as ET
try:
    root = ET.parse(sys.argv[1]).getroot()
except Exception as exc:  # missing or truncated => treat as failure
    print(f"unreadable:{exc}"); raise SystemExit(0)
suites = [root] if root.tag == "testsuite" else list(root)
total = sum(int(s.get("tests", 0)) for s in suites)
bad = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
print(bad if total else "collected-nothing")
PYEOF
}

# ``|| true`` keeps the second module and the figure check running when the first
# module fails, but the exit codes are captured and this script exits non-zero on
# any of them -- it is the acceptance gate, so it has to be able to fail.
python -m pytest "$T/test_multichip_decoder.py" -q --no-header -p no:randomly \
  --junitxml="$D/test_results.xml" > "$D/logs/full_test_run.log" 2>&1 || true
main_exit=$(junit_failures "$D/test_results.xml")
grep -aoE "[0-9]+ (passed|failed)[^|]*" "$D/logs/full_test_run.log" | tail -1

python -m pytest "$T/test_multichip_vs_single_chip.py" -q --no-header -p no:randomly \
  --junitxml="$D/test_results_vs_single_chip.xml" > "$D/logs/vs_single_chip_run.log" 2>&1 || true
vs_exit=$(junit_failures "$D/test_results_vs_single_chip.xml")
grep -aoE "[0-9]+ (passed|failed)[^|]*" "$D/logs/vs_single_chip_run.log" | tail -1

# The multichip-vs-single-chip worst values are the ones that can see a
# parallelisation or scheduling fault; print them, not just the pass count.
grep -haoE "worst\[[^]]*\]: [0-9.]+ on [a-z0-9]+" "$D/logs/vs_single_chip_run.log" || true

# Every mechanically-sourced figure in README.md, work_log.md and
# context_contract.json, re-derived from the committed CSVs and logs.  Three
# rounds of $stage-review found the same class of defect -- a number from a
# superseded run -- so it is now a gate rather than a habit.
python "$D/bench/check_reported_figures.py"
figures_exit=$?

if [ "$main_exit" != "0" ] || [ "$vs_exit" != "0" ] || [ "$figures_exit" != "0" ]; then
  echo "GATE FAILED: ${main_exit} failing test(s) in the main module, ${vs_exit} in the comparison module, figure check exit ${figures_exit}"
  exit 1
fi
echo "GATE PASSED"
