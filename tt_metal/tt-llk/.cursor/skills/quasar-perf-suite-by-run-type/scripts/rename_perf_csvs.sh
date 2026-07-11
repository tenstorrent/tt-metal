#!/usr/bin/env bash
# Rename combined perf CSVs under tt-llk/perf_data/<test_name>/ to append
# _<PerfRunType> before the .csv / .post.csv suffix.
#
# Usage:
#   rename_perf_csvs.sh --llk-root DIR --run-type PerfRunType [--test-name NAME ...]
#   rename_perf_csvs.sh --llk-root DIR --run-type PerfRunType --from-suite-dir DIR
#
# With --from-suite-dir, discovers test names from perf_output_<run_type>_*.txt
# basenames / suite mapping. Without test names, renames every directory under
# perf_data/ that looks like a suite perf test (perf_*_quasar).
set -euo pipefail

LLK_ROOT=""
RUN_TYPE=""
SUITE_DIR=""
TEST_NAMES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llk-root) LLK_ROOT="$2"; shift 2 ;;
    --run-type) RUN_TYPE="$2"; shift 2 ;;
    --from-suite-dir) SUITE_DIR="$2"; shift 2 ;;
    --test-name) TEST_NAMES+=("$2"); shift 2 ;;
    --help|-h)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$LLK_ROOT" || -z "$RUN_TYPE" ]]; then
  echo "ERROR: --llk-root and --run-type are required" >&2
  exit 2
fi

PERF_DATA="${LLK_ROOT}/perf_data"
if [[ ! -d "$PERF_DATA" ]]; then
  echo "ERROR: perf_data not found: ${PERF_DATA}" >&2
  exit 1
fi

# Discover test names from suite outputs if requested.
if [[ -n "$SUITE_DIR" ]]; then
  shopt -s nullglob
  for out in "${SUITE_DIR}/perf_output_${RUN_TYPE}_"*.txt; do
    # Prefer PASSED / collected runs; still rename CSVs even on partial failure
    # if the directory exists — caller decides which names to pass.
    :
  done
  shopt -u nullglob
fi

if [[ ${#TEST_NAMES[@]} -eq 0 ]]; then
  # Default: all perf_*_quasar dirs present under perf_data.
  while IFS= read -r -d '' d; do
    TEST_NAMES+=("$(basename "$d")")
  done < <(find "$PERF_DATA" -mindepth 1 -maxdepth 1 -type d -name 'perf_*_quasar' -print0 | sort -z)
fi

rename_one() {
  local test_name="$1"
  local dir="${PERF_DATA}/${test_name}"
  if [[ ! -d "$dir" ]]; then
    echo "SKIP  ${test_name}: no directory ${dir}"
    return 0
  fi

  local renamed=0
  local f base dest
  shopt -s nullglob
  for f in "${dir}"/*.csv; do
    base=$(basename "$f")
    # Already tagged for this (or any) run type? skip if suffix already present.
    if [[ "$base" == *"_${RUN_TYPE}.csv" || "$base" == *"_${RUN_TYPE}.post.csv" ]]; then
      echo "KEEP  ${test_name}/${base} (already tagged)"
      continue
    fi
    # Skip other run-type tags (L1_TO_L1, UNPACK_ISOLATE, ...).
    if [[ "$base" =~ _(L1_TO_L1|UNPACK_ISOLATE|MATH_ISOLATE|PACK_ISOLATE|L1_CONGESTION)(\.post)?\.csv$ ]]; then
      echo "KEEP  ${test_name}/${base} (other run-type tag)"
      continue
    fi

    if [[ "$base" == *.post.csv ]]; then
      dest="${base%.post.csv}_${RUN_TYPE}.post.csv"
    else
      dest="${base%.csv}_${RUN_TYPE}.csv"
    fi

    if [[ -e "${dir}/${dest}" ]]; then
      echo "SKIP  ${test_name}/${base} -> ${dest} (destination exists)"
      continue
    fi
    mv -- "${f}" "${dir}/${dest}"
    echo "RENAME ${test_name}/${base} -> ${dest}"
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
