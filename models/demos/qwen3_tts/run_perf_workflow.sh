#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TIMESTAMP="$(date +%Y_%m_%d_%H_%M_%S)"
LOG_DIR="${REPO_ROOT}/generated/profiler/workflow_logs"
LOG_FILE="${LOG_DIR}/qwen3_tts_perf_${TIMESTAMP}.txt"

CSV_PATH=""
RUN_PCC=1
RUN_TRACY=1
RUN_DEMO=1
RUN_AUDIO_QUALITY=1
TRACY_OP_SUPPORT_COUNT=2600
TRACY_TEST="models/demos/qwen3_tts/tests/profile_single_layer.py"
PCC_TEST="models/demos/qwen3_tts/tests/test_chain_pcc.py::test_chained_28_layer_pcc"
AUDIO_QUALITY_TEST="models/demos/qwen3_tts/tests/test_ttnn_audio_quality.py"
DEMO_TEXT="Hello, this is a test of the Qwen3 TTS speech system running on Tenstorrent hardware."
DEMO_REF_AUDIO="models/demos/qwen3_tts/demo/jim_reference.wav"
DEMO_REF_TEXT="Jason, can we take a look at the review slides"
DEMO_OUTPUT="/tmp/ttnn_tts_output.wav"
DEMO_SEED=42
DEMO_USE_2CQ=1

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --csv PATH                 Use an existing perf CSV instead of auto-detecting latest.
  --skip-tracy               Skip tracy profiling run.
  --skip-demo                Skip demo generation run.
  --skip-audio-quality       Skip audio quality pytest run.
  --skip-pcc                 Skip PCC test run.
  --demo-output PATH         Demo output wav path (default: ${DEMO_OUTPUT}).
  --demo-text TEXT           Demo text prompt.
  --demo-ref-audio PATH      Demo reference audio path.
  --demo-ref-text TEXT       Demo reference transcript.
  --demo-seed N              Demo seed (default: ${DEMO_SEED}).
  --no-use-2cq               Disable --use-2cq for demo.
  --tracy-op-support-count N Override --op-support-count for tracy (default: ${TRACY_OP_SUPPORT_COUNT}).
  -h, --help                 Show this help.

Examples:
  $(basename "$0")
  $(basename "$0") --skip-pcc --skip-audio-quality
  $(basename "$0") --skip-tracy --csv /abs/path/to/ops_perf_results_xxx.csv
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv)
            CSV_PATH="${2:-}"
            shift 2
            ;;
        --skip-pcc)
            RUN_PCC=0
            shift
            ;;
        --skip-demo)
            RUN_DEMO=0
            shift
            ;;
        --skip-audio-quality)
            RUN_AUDIO_QUALITY=0
            shift
            ;;
        --skip-tracy)
            RUN_TRACY=0
            shift
            ;;
        --demo-output)
            DEMO_OUTPUT="${2:-}"
            shift 2
            ;;
        --demo-text)
            DEMO_TEXT="${2:-}"
            shift 2
            ;;
        --demo-ref-audio)
            DEMO_REF_AUDIO="${2:-}"
            shift 2
            ;;
        --demo-ref-text)
            DEMO_REF_TEXT="${2:-}"
            shift 2
            ;;
        --demo-seed)
            DEMO_SEED="${2:-}"
            shift 2
            ;;
        --no-use-2cq)
            DEMO_USE_2CQ=0
            shift
            ;;
        --tracy-op-support-count)
            TRACY_OP_SUPPORT_COUNT="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "Qwen3-TTS Perf Workflow"
echo "Timestamp : ${TIMESTAMP}"
echo "Repo root : ${REPO_ROOT}"
echo "Log file  : ${LOG_FILE}"
echo "============================================================"

cd "${REPO_ROOT}"

if [[ ! -f "python_env/bin/activate" ]]; then
    echo "ERROR: python_env not found at ${REPO_ROOT}/python_env/bin/activate"
    exit 1
fi

# Required by request: activate project python env first.
source "python_env/bin/activate"

step_idx=1
total_steps=5

if [[ "${RUN_TRACY}" -eq 1 ]]; then
    echo "[${step_idx}/${total_steps}] Running tracy profile..."
    python -m tracy -p -v -r \
        --op-support-count "${TRACY_OP_SUPPORT_COUNT}" \
        --dump-device-data-mid-run \
        "${TRACY_TEST}"
else
    echo "[${step_idx}/${total_steps}] Skipping tracy profile (--skip-tracy)"
fi
step_idx=$((step_idx + 1))

if [[ -z "${CSV_PATH}" ]]; then
    echo "[${step_idx}/${total_steps}] Resolving latest perf CSV..."
    shopt -s nullglob
    csv_candidates=("${REPO_ROOT}"/generated/profiler/reports/*/ops_perf_results_*.csv)
    shopt -u nullglob
    if [[ ${#csv_candidates[@]} -eq 0 ]]; then
        echo "ERROR: No perf CSV found under generated/profiler/reports/*/"
        exit 1
    fi
    IFS=$'\n' sorted_csv=($(printf '%s\n' "${csv_candidates[@]}" | sort))
    unset IFS
    CSV_PATH="${sorted_csv[-1]}"
fi

if [[ ! -f "${CSV_PATH}" ]]; then
    echo "ERROR: CSV file does not exist: ${CSV_PATH}"
    exit 1
fi

echo "[${step_idx}/${total_steps}] Processing CSV with excel_process_combined.py"
echo "CSV: ${CSV_PATH}"
python3 excel_process_combined.py "${CSV_PATH}" .
step_idx=$((step_idx + 1))

if [[ "${RUN_DEMO}" -eq 1 ]]; then
    echo "[${step_idx}/${total_steps}] Running demo generation..."
    demo_cmd=(
        python models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py
        --text "${DEMO_TEXT}"
        --ref-audio "${DEMO_REF_AUDIO}"
        --ref-text "${DEMO_REF_TEXT}"
        --output "${DEMO_OUTPUT}"
        --seed "${DEMO_SEED}"
    )
    if [[ "${DEMO_USE_2CQ}" -eq 1 ]]; then
        demo_cmd+=(--use-2cq)
    fi
    "${demo_cmd[@]}"
else
    echo "[${step_idx}/${total_steps}] Skipping demo generation (--skip-demo)"
fi
step_idx=$((step_idx + 1))

if [[ "${RUN_AUDIO_QUALITY}" -eq 1 ]]; then
    echo "[${step_idx}/${total_steps}] Running audio quality tests..."
    pytest "${AUDIO_QUALITY_TEST}"
else
    echo "[${step_idx}/${total_steps}] Skipping audio quality tests (--skip-audio-quality)"
fi
step_idx=$((step_idx + 1))

if [[ "${RUN_PCC}" -eq 1 ]]; then
    echo "[${step_idx}/${total_steps}] Running PCC test..."
    pytest -q "${PCC_TEST}"
else
    echo "[${step_idx}/${total_steps}] Skipping PCC test (--skip-pcc)"
fi

echo "============================================================"
echo "Workflow finished successfully."
echo "Log saved to: ${LOG_FILE}"
echo "============================================================"
