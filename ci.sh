#!/bin/bash
# ci.sh — local, repeatable CI battery for the top-k campaign (single device).
#
# Runs the CI-relevant pytest suites serially under the device lock, one
# tee'd log per suite, one manifest line per suite with the real pytest
# summary, and automatic device recovery on a suite timeout. Mirrors what
# the GitHub tiers would run for this branch's touched surfaces; see the
# tier map below. The branch is local-only: this script is how "CI ran"
# until a push is permitted.
#
# Usage:
#   ./ci.sh                 # all tiers (overnight-class)
#   ./ci.sh --quick         # tier 1 only (top-k core proof, ~10 min)
#   ./ci.sh --tiers 1,2     # subset of tiers
#   ./ci.sh --out DIR       # output dir (default generated/ci_local/<UTC timestamp>)
#
# Not covered here (documented gaps):
#   * topk_large_indices production_perf_check cells: IOMMU-runner-only, and
#     their expected_duration pins are stale post-P-cap (re-baseline there).
#   * sweep-framework large_k suite: runs via the sweeps infrastructure.
#   * WH smoke: no WH silicon on this box.
set -u
cd "$(dirname "$0")"

TIERS="1,2,3,4"; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) TIERS="1";;
    --tiers) TIERS="$2"; shift;;
    --out) OUT="$2"; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
  shift
done
OUT="${OUT:-generated/ci_local/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
R="$OUT/RESULTS.txt"
: > "$R"

# Serialize against every other device user in this repo's conventions.
exec 9>/tmp/tt-device.lock
flock 9

source python_env/bin/activate
TT_SMI=tt_metal/tt-llk/tests/.venv/bin/tt-smi

in_tiers () { case ",$TIERS," in *",$1,"*) return 0;; *) return 1;; esac; }

run_suite () {
  local name="$1" tmo="$2"; shift 2
  local log="$OUT/${name}.log"
  echo "[$(date +%H:%M:%S)] START $name" | tee -a "$R"
  timeout "$tmo" pytest "$@" -q --timeout=600 --timeout-method=thread > "$log" 2>&1
  local rc=$?
  local summary
  summary=$(grep -E "^=+ .*(passed|failed|error|skipped|no tests ran).*=+" "$log" | tail -1)
  echo "[$(date +%H:%M:%S)] END   $name rc=$rc :: ${summary:-NO-SUMMARY (see $log)}" | tee -a "$R"
  if [ $rc -ge 124 ]; then
    echo "[$(date +%H:%M:%S)] RECOVER $name (timeout) — tt-smi -r" | tee -a "$R"
    pkill -f pytest 2>/dev/null; sleep 5; "$TT_SMI" -r >> "$R" 2>&1; sleep 10
  fi
}

if in_tiers 1; then
  # Tier 1 — top-k core proof
  run_suite contract_default 2400 tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py
  TOPK_CONTRACT_FULL=1 run_suite contract_full 4800 tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py
  run_suite reduce_topk 4800 tests/ttnn/unit_tests/operations/reduce/test_topk.py
  run_suite tli_nightly 4800 tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py -k "not production_perf_check"
fi
if in_tiers 2; then
  # Tier 2 — consumers of the touched surfaces
  run_suite reduce_sampling 3600 tests/ttnn/unit_tests/operations/reduce/test_sampling.py
  run_suite reduce_moe 3600 tests/ttnn/unit_tests/operations/reduce/test_moe.py
  run_suite dm_sort 7200 tests/ttnn/unit_tests/operations/data_movement/test_sort.py
  run_suite reduce_dir_full 10800 tests/ttnn/unit_tests/operations/reduce/
fi
if in_tiers 3; then
  # Tier 3 — L2-nightly groups that carry our tests
  run_suite nightly_reduction 10800 tests/ttnn/nightly/unit_tests/operations/reduction/
  run_suite nightly_experimental 14400 tests/ttnn/nightly/unit_tests/operations/experimental/
fi
if in_tiers 4; then
  # Tier 4 — broad unit long tail (one suite per operations subdir)
  run_suite unit_reduction_dir 3600 tests/ttnn/unit_tests/operations/reduction/
  for d in tests/ttnn/unit_tests/operations/*/; do
    base=$(basename "$d")
    case "$base" in reduce|reduction|__pycache__) continue;; esac
    run_suite "unit_${base}" 10800 "$d"
  done
fi

echo "[$(date +%H:%M:%S)] CI-LOCAL-DONE tiers=$TIERS out=$OUT" | tee -a "$R"
fails=$(grep -cE "rc=[1-9]" "$R")
echo "suites with nonzero rc: $fails (see $R)"
exit 0
