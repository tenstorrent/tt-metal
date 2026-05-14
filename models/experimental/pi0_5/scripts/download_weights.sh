#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Download PI0.5 weights from HuggingFace into the weights/ directory.
#
# Usage:
#   ./scripts/download_weights.sh              # download both checkpoints
#   ./scripts/download_weights.sh base         # download pi05_base only
#   ./scripts/download_weights.sh libero       # download pi05_libero only
#
# Prerequisites:
#   pip install huggingface_hub[cli]
#
# The Gemma-family weights require accepting the license on HuggingFace.
# Either log in (`huggingface-cli login`) or set HF_TOKEN in your environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/../weights"

HF_REPO_BASE="lerobot/pi05_base"
HF_REPO_LIBERO="lerobot/pi05_libero_finetuned_v044"

LOCAL_DIR_BASE="${WEIGHTS_DIR}/pi05_base"
LOCAL_DIR_LIBERO="${WEIGHTS_DIR}/pi05_libero_finetuned"

download_base() {
    echo "Downloading pi05_base from ${HF_REPO_BASE} ..."
    huggingface-cli download "${HF_REPO_BASE}" \
        --local-dir "${LOCAL_DIR_BASE}" \
        --local-dir-use-symlinks False
    echo "pi05_base saved to ${LOCAL_DIR_BASE}"
}

download_libero() {
    echo "Downloading pi05_libero from ${HF_REPO_LIBERO} ..."
    huggingface-cli download "${HF_REPO_LIBERO}" \
        --local-dir "${LOCAL_DIR_LIBERO}" \
        --local-dir-use-symlinks False
    echo "pi05_libero saved to ${LOCAL_DIR_LIBERO}"
}

TARGET="${1:-all}"

case "${TARGET}" in
    base)   download_base ;;
    libero) download_libero ;;
    all)    download_base; echo ""; download_libero ;;
    *)
        echo "Usage: $0 [base|libero|all]"
        exit 1
        ;;
esac

echo ""
echo "Done. Weights directory:"
ls -lh "${WEIGHTS_DIR}"/
