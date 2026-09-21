#!/usr/bin/env bash
# COLLECTION PROBE -- no device work.
#
# Prints, for every perf module, its item count and three sample ids, so the
# predecessor experiment can be written with -k expressions known to select
# something. The first attempt died at the third module: piping pytest into
# `head` closes the pipe, and `set -o pipefail` turned that SIGPIPE into a
# failure. Collection now goes to a file and the file is read twice.
set -euo pipefail
GROUP="${1:?}"
N_GROUPS="${2:?}"
if [ "$GROUP" != "1" ]; then
  echo "probe: only group 1 collects; this group exits."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data

collect() {
  # $1 = label, rest = pytest selection. Never fails the script: an empty
  # selection is a result, not an error.
  local label="$1"; shift
  local out=/tmp/collect.txt
  set +e
  pytest -q --collect-only -m "perf and not accuracy" "$@" >"$out" 2>&1
  local rc=$?
  set -e
  echo "===== $label  (rc=$rc)"
  tail -2 "$out"
  echo "----- samples"
  head -3 "$out"
}

collect "TOTAL" .

for f in perf_*.py; do
  collect "MODULE $f" "$f"
done

echo "===== matmul narrow candidates ====="
for k in "matmul_config0" "LoFi" "throttle:0" "num_blocks:1"; do
  collect "matmul -k '$k'" -k "$k" perf_math_matmul.py
done
echo "===== probe done ====="
