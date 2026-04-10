#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ROOT_DIR="${REPO_ROOT}/models/experimental/mole/demo_checkpoints"
DATASET_FILE="ETTh1.csv"
DATASET_URL="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh1.csv?download=true"
CHECKPOINT_FILE="checkpoint.pth"
CHECKPOINT_REMOTE_ROOT="demo_checkpoints"
CHECKPOINT_REPO_CLONE_URL="https://huggingface.co/hybelj/mole"
BATCH_SIZE=1
SEED=0

EXTRA_SITE_PACKAGES="${REPO_ROOT}/python_env/lib/python3.10/site-packages"
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${REPO_ROOT}:${EXTRA_SITE_PACKAGES}:${PYTHONPATH}"
else
    export PYTHONPATH="${REPO_ROOT}:${EXTRA_SITE_PACKAGES}"
fi

download_file() {
    local url="$1"
    local output_path="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${url}" -o "${output_path}"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "${output_path}" "${url}"
    else
        return 1
    fi
}

download_checkpoint_folder() {
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "${tmp_dir}"' RETURN

    if ! command -v git >/dev/null 2>&1; then
        echo "git is required to download checkpoint folder from ${CHECKPOINT_REPO_CLONE_URL}" >&2
        return 1
    fi

    echo "downloading checkpoint folder from ${CHECKPOINT_REPO_CLONE_URL}"
    git clone --depth 1 --filter=blob:none --sparse "${CHECKPOINT_REPO_CLONE_URL}" "${tmp_dir}/repo" >/dev/null 2>&1 || return 1
    (
        cd "${tmp_dir}/repo"
        git sparse-checkout set "${CHECKPOINT_REMOTE_ROOT}" >/dev/null 2>&1 || exit 1
    )

    mkdir -p "${ROOT_DIR}"
    cp -a "${tmp_dir}/repo/${CHECKPOINT_REMOTE_ROOT}/." "${ROOT_DIR}/" || return 1
}

PYTHON_BIN=""
python_candidates=(
    "${REPO_ROOT}/python_env/bin/python"
    "/opt/venv/bin/python3"
    "python3"
)
for candidate in "${python_candidates[@]}"; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
        continue
    fi
    if "${candidate}" - <<'PY' >/dev/null 2>&1
import torch
import ttnn
import models.experimental.mole.demo.run
PY
    then
        PYTHON_BIN="${candidate}"
        break
    fi
done

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "no usable python interpreter found (failed to import torch+ttnn+mole demo modules)" >&2
    exit 1
fi

data_csv="${ROOT_DIR}/${DATASET_FILE}"
if [[ ! -f "${data_csv}" ]]; then
    echo "dataset CSV not found, downloading: ${data_csv}"
    mkdir -p "${ROOT_DIR}"
    if ! download_file "${DATASET_URL}" "${data_csv}"; then
        echo "neither curl nor wget is available; cannot download ${DATASET_FILE}" >&2
        exit 1
    fi
fi

runs=(
    "dlinear|4|336|96|${ROOT_DIR}/etth1_DLinear_mole_sl336_pl96_td4_lr0.01_hd0.2_sd2021_MoLE_DLinear_ETTh1_ftM_sl336_ll336_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_DemoMatrix_0_32_4_f_mask_0.5_0.01_sd2021_hd0.2"
    "dlinear|1|336|96|${ROOT_DIR}/etth1_DLinear_linear_sl336_pl96_td1_lr0.01_hd0.0_sd2021_MoLE_DLinear_ETTh1_ftM_sl336_ll336_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_DemoMatrix_0_32_1_f_mask_0.5_0.01_sd2021_hd0.0"
    "rlinear|4|336|96|${ROOT_DIR}/etth1_RLinear_mole_sl336_pl96_td4_lr0.005_hd0.2_sd2021_MoLE_RLinear_ETTh1_ftM_sl336_ll336_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_DemoMatrix_0_32_4_f_mask_0.5_0.005_sd2021_hd0.2"
    "rmlp|4|336|96|${ROOT_DIR}/etth1_RMLP_mole_sl336_pl96_td4_lr0.005_hd0.2_sd2021_MoLE_RMLP_ETTh1_ftM_sl336_ll336_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_DemoMatrix_0_32_4_f_mask_0.5_0.005_sd2021_hd0.2"
)

missing_checkpoints=0
for run_item in "${runs[@]}"; do
    IFS='|' read -r _model_type _num_experts _seq_len _pred_len checkpoint_dir <<< "${run_item}"
    if [[ ! -f "${checkpoint_dir}/${CHECKPOINT_FILE}" ]]; then
        missing_checkpoints=1
        break
    fi
done

if (( missing_checkpoints > 0 )); then
    if ! download_checkpoint_folder; then
        echo "failed to download checkpoint folder from ${CHECKPOINT_REPO_CLONE_URL}" >&2
        exit 1
    fi
fi

echo "Running ${#runs[@]} bundled checkpoint(s) from: ${ROOT_DIR}"

failures=()
index=0
for run_item in "${runs[@]}"; do
    index=$((index + 1))
    IFS='|' read -r model_type num_experts seq_len pred_len checkpoint_dir <<< "${run_item}"

    name="$(basename "${checkpoint_dir}")"
    checkpoint_path="${checkpoint_dir}/${CHECKPOINT_FILE}"
    if [[ ! -f "${checkpoint_path}" ]]; then
        echo "[${index}/${#runs[@]}] ${name}"
        echo "  -> FAILED (missing checkpoint file: ${checkpoint_path})"
        failures+=("${name}:missing")
        continue
    fi

    echo "[${index}/${#runs[@]}] ${name} (model=${model_type}, experts=${num_experts}, seq=${seq_len}, pred=${pred_len})"

    if "${PYTHON_BIN}" -m models.experimental.mole.demo.run \
        --base-model-type "${model_type}" \
        --num-experts "${num_experts}" \
        --seq-len "${seq_len}" \
        --pred-len "${pred_len}" \
        --dataset-dir "${ROOT_DIR}" \
        --dataset-file "${DATASET_FILE}" \
        --checkpoint-dir "${checkpoint_dir}" \
        --checkpoint-file "${CHECKPOINT_FILE}" \
        --batch-size "${BATCH_SIZE}" \
        --seed "${SEED}"; then
        echo "  -> OK"
    else
        code=$?
        echo "  -> FAILED (exit=${code})"
        failures+=("${name}:${code}")
    fi
done

passed=$(( ${#runs[@]} - ${#failures[@]} ))
echo
echo "Summary: passed=${passed} failed=${#failures[@]} total=${#runs[@]}"
if (( ${#failures[@]} > 0 )); then
    for item in "${failures[@]}"; do
        checkpoint_name="${item%%:*}"
        code="${item##*:}"
        echo "  - ${checkpoint_name}: exit=${code}"
    done
    exit 1
fi
