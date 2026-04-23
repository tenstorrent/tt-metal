#!/usr/bin/env bash
# Generate raw tracy ops CSV for Qwen3-32B 2-layer prefill profile.
# Usage:
#   ./models/demos/llama3_70b_galaxy/tools/run_qwen_profile_2l_raw_csv.sh baseline [out_root]
#   ./models/demos/llama3_70b_galaxy/tools/run_qwen_profile_2l_raw_csv.sh fused    [out_root]
#
# Produces:
#   <out_root>/<baseline|fused>/ops_perf_<baseline|fused>.csv

set -euo pipefail

run_kind="${1:?must pass baseline or fused}"
out_root="${2:-$HOME/qwen_ff2_profile_wh}"

case "${run_kind}" in
    baseline)
        export QWEN_FF2_DISABLE_FUSED=1
        ;;
    fused)
        unset QWEN_FF2_DISABLE_FUSED || true
        ;;
    *)
        echo "unknown run kind: ${run_kind}" >&2
        exit 2
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
out_dir="${out_root}/${run_kind}"
mkdir -p "${out_dir}"

cd "${repo_root}"
source python_env/bin/activate

export TT_METAL_HOME="${repo_root}"
export PYTHONPATH="${repo_root}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_MODEL="${HF_MODEL:-Qwen/Qwen3-32B}"
export TT_CACHE_PATH="${TT_CACHE_PATH:-$HOME/tt_cache}"
export TT_METAL_DEVICE_PROFILER=1

# Exit after prefill to avoid decode-trace capture issues under device profiler.
export QWEN_EXIT_AFTER_PREFILL=1

echo "[$(date -Is)] starting ${run_kind} profile run" | tee "${out_dir}/run.log"
echo "ARCH_NAME=${ARCH_NAME:-<unset>}" | tee -a "${out_dir}/run.log"
echo "QWEN_FF2_DISABLE_FUSED=${QWEN_FF2_DISABLE_FUSED:-<unset>}" | tee -a "${out_dir}/run.log"

python3 -m tracy -v -r -p \
    -o "${out_dir}" \
    -n "${run_kind}" \
    -m pytest -sv models/demos/llama3_70b_galaxy/demo/text_qwen_demo.py \
        -k "bh-glx-prefill-4k-profile-2L" \
    2>&1 | tee -a "${out_dir}/run.log"
rc=${PIPESTATUS[0]}
echo "[$(date -Is)] ${run_kind} run finished, rc=${rc}" | tee -a "${out_dir}/run.log"

csv_path="$(ls "${out_dir}"/reports/"${run_kind}"/*/ops_perf_results_"${run_kind}"_*.csv 2>/dev/null | head -1 || true)"
if [[ -z "${csv_path}" ]]; then
    echo "ERROR: ops_perf CSV not found under ${out_dir}/reports/${run_kind}/" >&2
    exit 1
fi

cp -f "${csv_path}" "${out_dir}/ops_perf_${run_kind}.csv"
echo "raw csv copied to: ${out_dir}/ops_perf_${run_kind}.csv"
exit "${rc}"
