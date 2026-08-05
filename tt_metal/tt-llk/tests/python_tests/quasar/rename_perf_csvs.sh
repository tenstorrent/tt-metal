#!/usr/bin/env bash
# Tag combined perf CSVs with the PerfRunType that produced them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=perf_suite_common.sh
source "${SCRIPT_DIR}/perf_suite_common.sh"

RUN_TYPE=""
TEST_NAMES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llk-root) LLK_ROOT="$2"; shift 2 ;;
    --run-type) RUN_TYPE="$2"; shift 2 ;;
    --test-name) TEST_NAMES+=("$2"); shift 2 ;;
    --help|-h)
      echo "Usage: $0 --run-type PerfRunType [--llk-root DIR] [--test-name NAME ...]"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_TYPE" ]]; then
  echo "ERROR: --run-type is required" >&2
  exit 2
fi

PERF_DATA="${LLK_ROOT}/perf_data"
if [[ ! -d "$PERF_DATA" ]]; then
  echo "ERROR: perf_data not found: ${PERF_DATA}" >&2
  exit 1
fi

if [[ ${#TEST_NAMES[@]} -eq 0 ]]; then
  for entry in "${PERF_SUITE_TESTS[@]}"; do
    TEST_NAMES+=("${entry#*:}")
    TEST_NAMES[-1]="${TEST_NAMES[-1]%.py}"
  done
fi

rename_one() {
  local test_name="$1"
  local dir="${PERF_DATA}/${test_name}"
  if [[ ! -d "$dir" ]]; then
    echo "SKIP  ${test_name}: no directory ${dir}"
    return
  fi

  local renamed=0
  local file base destination
  shopt -s nullglob
  for file in "${dir}"/*.csv; do
    base=$(basename "$file")
    if [[ "$base" =~ _(L1_TO_L1|UNPACK_ISOLATE|MATH_ISOLATE|PACK_ISOLATE|L1_CONGESTION)(\.post)?\.csv$ ]]; then
      echo "KEEP  ${test_name}/${base} (already tagged)"
      continue
    fi

    if [[ "$base" == *.post.csv ]]; then
      destination="${base%.post.csv}_${RUN_TYPE}.post.csv"
    else
      destination="${base%.csv}_${RUN_TYPE}.csv"
    fi

    if [[ -e "${dir}/${destination}" ]]; then
      echo "SKIP  ${test_name}/${base} -> ${destination} (destination exists)"
      continue
    fi
    mv -- "$file" "${dir}/${destination}"
    echo "RENAME ${test_name}/${base} -> ${destination}"
    renamed=$((renamed + 1))
  done
  shopt -u nullglob

  if [[ $renamed -eq 0 ]]; then
    echo "NONE  ${test_name}: no untagged CSVs to rename"
  fi
}

echo "Renaming CSVs under ${PERF_DATA} with suffix _${RUN_TYPE}"
for name in "${TEST_NAMES[@]}"; do
  rename_one "$name"
done
