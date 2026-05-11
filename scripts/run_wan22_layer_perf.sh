#!/usr/bin/env bash
set -euo pipefail

# Run WAN2.2 transformer block at 1536x2048 across mesh sweep,
# wrapped in tracy profiler, and summarize perf CSVs.
#
# Usage:
#   scripts/run_wan22_layer_perf.sh                # run all IDs
#   scripts/run_wan22_layer_perf.sh sweep_2x2 ...  # run selected IDs
#
# Env (optional):
#   ARCH=wh|bh        (default: wh)
#   LOG_DIR=run-logs/w22-layer-perf
#   OUT_DIR=perf-summaries
#

ARCH="${ARCH:-wh}"
if [[ "${ARCH}" != "wh" && "${ARCH}" != "bh" ]]; then
  echo "ERROR: ARCH must be 'wh' or 'bh', got '${ARCH}'" >&2
  exit 1
fi

PYTEST_TEST="models/tt_dit/tests/models/wan2_2/test_transformer_wan.py::test_mesh_sweep_1536p_wan22_block"

# Param IDs from the test's @pytest.mark.parametrize(ids=[...])
WH_IDS=(
  wh_sweep_2x2
  wh_sweep_1x4_sp0tp1
  wh_sweep_1x4_sp1tp0
  wh_sweep_2x4_sp0tp1
  wh_sweep_2x4_sp1tp0
  wh_sweep_4x4
  wh_sweep_4x8_sp0tp1
  wh_sweep_4x8_sp1tp0
)

BH_IDS=(
  bh_sweep_2x2
  bh_sweep_1x4_sp0tp1
  bh_sweep_1x4_sp1tp0
  bh_sweep_2x4_sp0tp1
  bh_sweep_2x4_sp1tp0
  bh_sweep_4x4
  bh_sweep_4x8_sp0tp1
  bh_sweep_4x8_sp1tp0
)

if [[ "${ARCH}" == "wh" ]]; then
  DEFAULT_IDS=("${WH_IDS[@]}")
  DEVICE_TAG="wormhole_b0"
else
  DEFAULT_IDS=("${BH_IDS[@]}")
  DEVICE_TAG="blackhole"
fi

LOG_DIR="${LOG_DIR:-run-logs/w22-layer-perf}"
OUT_DIR="${OUT_DIR:-perf-summaries}"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

run_one() {
  local id="$1"
  echo "=== Running ${id} (${ARCH}) ==="
  local pytest_cmd="pytest -q ${PYTEST_TEST}[${DEVICE_TAG}-${id}] -s"
  local tracy_cmd="python -m tracy -r -m \"${pytest_cmd}\""
  echo "+ ${tracy_cmd}"

  local log="${LOG_DIR}/${id}.log"
  # Run tracy-wrapped pytest and tee output to log
  bash -lc "${tracy_cmd}" 2>&1 | tee "${log}" || true

  # Try to extract the CSV path from the output log
  local csv
  csv="$(grep -Eo 'generated/profiler/reports/[0-9_]+/ops_perf_results_[0-9_]+\.csv' "${log}" | tail -n 1 || true)"

  # Fallback: pick the latest generated report CSV
  if [[ -z "${csv:-}" ]]; then
    if [[ -d generated/profiler/reports ]]; then
      local latest_dir
      latest_dir="$(ls -1dt generated/profiler/reports/* 2>/dev/null | head -n1 || true)"
      if [[ -n "${latest_dir}" ]]; then
        csv="$(ls -1 "${latest_dir}"/ops_perf_results_*.csv 2>/dev/null | head -n1 || true)"
      fi
    fi
  fi

  if [[ -z "${csv:-}" ]]; then
    echo "[WARN] Could not locate ops_perf_results CSV for ${id}" >&2
    return 0
  fi

  echo "Found CSV: ${csv}"
  local summary="${OUT_DIR}/ops_perf_summary_${id}.csv"
  echo "+ tt-perf-report \"${csv}\" --csv \"${summary}\""
  if ! tt-perf-report "${csv}" --csv "${summary}"; then
    echo "[WARN] tt-perf-report failed for ${id}" >&2
  else
    echo "Summary written: ${summary}"
  fi
}

main() {
  local ids=()
  if [[ $# -gt 0 ]]; then
    ids=("$@")
  else
    ids=("${DEFAULT_IDS[@]}")
  fi

  for id in "${ids[@]}"; do
    run_one "${id}"
  done
}

main "$@"
