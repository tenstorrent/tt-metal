#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# Purpose: Setup and deploy Llama 3.1 70B Instruct model with dependencies.

set -euo pipefail

# Model information
MODEL=llama-3.1-70b-instruct
META_MODEL_NAME="Meta-Llama-3.1-70B-Instruct"
META_DIR_FILTER="llama3_1"

# Require commit SHA or tag for TT_METAL and vLLM
TT_METAL_COMMIT_SHA_OR_TAG=${1:-""}
TT_VLLM_COMMIT_SHA_OR_TAG=${2:-""}

# Ensure required arguments are passed
if [[ -z "${TT_METAL_COMMIT_SHA_OR_TAG}" || -z "${TT_VLLM_COMMIT_SHA_OR_TAG}" ]]; then
    echo "❌ Error: Both TT_METAL_COMMIT_SHA_OR_TAG and TT_VLLM_COMMIT_SHA_OR_TAG are required."
    echo "Usage: $0 <TT_METAL_COMMIT_SHA_OR_TAG> <TT_VLLM_COMMIT_SHA_OR_TAG>"
    exit 1
fi

# Defined variables
DEFAULT_PERSISTENT_VOLUME_ROOT=~/persistent_volume
DEFAULT_LLAMA_REPO=~/llama-models

# functions
error_exit() {
    echo "⛔ Error: $1" >&2
    exit 1
}

print_step() {
    echo -e "\n👉 $1...\n"
}

setup_environment() {
    print_step "Setting up environment"
    export LLAMA3_CKPT_DIR="${DEFAULT_PERSISTENT_VOLUME_ROOT}/model_weights/repacked-${MODEL}"
    export LLAMA3_TOKENIZER_PATH="${LLAMA3_CKPT_DIR}/tokenizer.model"
    export LLAMA3_CACHE_PATH="${DEFAULT_PERSISTENT_VOLUME_ROOT}/tt_metal_cache/cache_repacked-${MODEL}"
    export ARCH_NAME=wormhole_b0
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    echo "Environment variables set."
}

check_and_build_tt_metal() {
    print_step "Checking and building tt-metal"
    pushd "${TT_METAL_HOME}" >/dev/null
    if [[ ! -d "python_env" ]]; then
        git checkout "${TT_METAL_COMMIT_SHA_OR_TAG}"
        git submodule update --init --recursive
        git submodule foreach 'git lfs fetch --all && git lfs pull'
        ./build_metal.sh
        ./create_venv.sh
        source python_env/bin/activate
        pip install -r models/demos/t3000/llama2_70b/reference/llama/requirements.txt
    else
        echo "🔔 tt-metal Python environment already exists. Skipping build."
        source python_env/bin/activate
    fi
    popd >/dev/null
}

clone_repo() {
    local REPO_PATH=$1
    local REPO_URL=$2
    local COMMIT_HASH=$3

    print_step "Cloning Llama repository"
    if [[ ! -d "${REPO_PATH}" ]]; then
        git clone "${REPO_URL}" "${REPO_PATH}"
        pushd "${REPO_PATH}" >/dev/null
        git checkout "${COMMIT_HASH}"
        popd >/dev/null
    else
        echo "🔔 Repository already exists at ${REPO_PATH}, skipping clone."
    fi
}

setup_weights() {
    print_step "Setting up weights"
    local LLAMA_REPO=$1
    local LLAMA_DIR="${LLAMA_REPO}/models/${META_DIR_FILTER}"
    local LLAMA_WEIGHTS_DIR="${LLAMA_DIR}/${META_MODEL_NAME}"
    local WEIGHTS_DIR="${LLAMA3_CKPT_DIR}"

    mkdir -p "${WEIGHTS_DIR}" "${LLAMA3_CACHE_PATH}"

    if [[ -d "${LLAMA_WEIGHTS_DIR}" && -n "$(ls -A "${LLAMA_WEIGHTS_DIR}")" ]]; then
        echo "Weights already downloaded in ${LLAMA_WEIGHTS_DIR}"
    else
        print_step "Downloading weights"
        pushd "${LLAMA_DIR}" >/dev/null
        [[ -x "./download.sh" ]] && ./download.sh || error_exit "Download script not found!"
        popd >/dev/null
    fi

    print_step "Repacking weights"
    source python_env/bin/activate
    huggingface-cli login
    cp "${LLAMA_WEIGHTS_DIR}/tokenizer.model" "${WEIGHTS_DIR}/tokenizer.model"
    cp "${LLAMA_WEIGHTS_DIR}/params.json" "${WEIGHTS_DIR}/params.json"
    python models/demos/t3000/llama2_70b/scripts/repack_weights.py "${LLAMA_WEIGHTS_DIR}" "${WEIGHTS_DIR}" 5
}

install_vllm() {
    print_step "Installing vLLM"
    if [[ ! -d "vllm" ]]; then
        source python_env/bin/activate
        export VLLM_TARGET_DEVICE="tt"
        git clone https://github.com/tenstorrent/vllm.git
        pushd vllm >/dev/null
        git checkout "${TT_VLLM_COMMIT_SHA_OR_TAG}"
        pip install -e .
        popd >/dev/null
    else
        echo "🔔 vLLM already installed. Skipping install."
    fi
}

deploy_server() {
    print_step "Deploying Llama server"
    source python_env/bin/activate
    export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
    python vllm/examples/server_example_tt.py
    echo "✅ Deployment complete! Interact via http://localhost:8000."
}

# ---- MAIN ----
setup_environment
check_and_build_tt_metal
clone_repo "${DEFAULT_LLAMA_REPO}" "https://github.com/meta-llama/llama-models.git" "685ac4c107c75ce8c291248710bf990a876e1623"
setup_weights "${DEFAULT_LLAMA_REPO}"
install_vllm
deploy_server
