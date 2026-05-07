#!/usr/bin/env bash
set -u

usage() {
  echo "Usage: $0 <dir_a> <dir_b> [--threshold F] [--seq-len N] [--full]" >&2
  echo "  Compares matching kv_cache_stage_*_layer_*.pt files; requires repo root layout." >&2
  exit 2
}

if [[ $# -lt 2 ]]; then
  usage
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

DIR_A="${1:?}"
DIR_B="${2:?}"
shift 2

PYTHON_BIN="${PYTHON:-python3}"
PY_MOD=(python -m models.demos.deepseek_v3_b1.demo.compare_kv_cache)
PY_MOD[0]="$PYTHON_BIN"

EXTRA=("$@")

missing=0
failed=0
compared=0
lowest_pcc=""
lowest_name=""

shopt -s nullglob
mapfile -t basenames < <(
  for f in "${DIR_A}"/kv_cache_stage_*_layer_*.pt; do
    basename "$f"
  done | sort
)

if [[ ${#basenames[@]} -eq 0 ]]; then
  echo "No kv_cache_stage_*_layer_*.pt files under ${DIR_A}" >&2
  exit 2
fi

echo "Comparing ${#basenames[@]} file(s) from"
echo "  A: ${DIR_A}"
echo "  B: ${DIR_B}"
echo ""

for base in "${basenames[@]}"; do
  fa="${DIR_A%/}/${base}"
  fb="${DIR_B%/}/${base}"
  if [[ ! -f "$fb" ]]; then
    echo "MISSING|${base}|(not in dir_b)"
    missing=$((missing + 1))
    continue
  fi

  compared=$((compared + 1))
  py_rc=0
  line="$("${PY_MOD[@]}" "$fa" "$fb" --compact "${EXTRA[@]}")" || py_rc=$?
  if [[ "$py_rc" -ne 0 ]]; then
    failed=$((failed + 1))
  fi
  echo "$line"
  if [[ "$line" == KV_CACHE_CMP\|* ]]; then
    pcc_field="$(echo "$line" | awk -F'|' '{print $5}')"
    if [[ -n "$pcc_field" ]]; then
      update=false
      if [[ -z "$lowest_pcc" ]]; then
        update=true
      elif awk -v n="$pcc_field" -v o="$lowest_pcc" 'BEGIN { exit !(n + 0 < o + 0) }'; then
        update=true
      fi
      if [[ "$update" == true ]]; then
        lowest_pcc="$pcc_field"
        lowest_name="$base"
      fi
    fi
  fi
done

echo ""
echo "Summary: compared=${compared} missing=${missing} failed_checks=${failed}"
if [[ -n "$lowest_pcc" ]]; then
  echo "Lowest PCC: ${lowest_pcc} (${lowest_name})"
fi

if [[ "$missing" -gt 0 || "$failed" -gt 0 ]]; then
  exit 1
fi
exit 0
