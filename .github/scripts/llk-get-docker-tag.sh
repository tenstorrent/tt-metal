#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# MIGRATION: This script was moved from tt_llk/.github/scripts/get-docker-tag.sh
# during the LLK-in-Metal migration. It now references files from both:
#   - The tt_llk submodule (Dockerfile.*, tests/requirements.txt)
#   - The parent tt-metal repo (llk-build-docker-images.sh)
#
# Calculate hash from the following files. This hash is used to tag the docker images.
# Any change in these files will result in a new docker image build.

LLK_PATH="${LLK_PATH:-tt_metal/third_party/tt_llk}"

# Files within the tt_llk submodule
DOCKERFILE_HASH_FILES=(
    "${LLK_PATH}/.github/Dockerfile.base"
    "${LLK_PATH}/.github/Dockerfile.ci"
    "${LLK_PATH}/.github/Dockerfile.ird"
    "${LLK_PATH}/.github/Dockerfile.ird.slim"
    "${LLK_PATH}/tests/requirements.txt"
    "${LLK_PATH}/.github/scripts/install-tests-dependencies.sh"
)

# Files within the parent tt-metal repo (this is the migrated addition)
METAL_BUILD_FILES=(
    ".github/scripts/llk-build-docker-images.sh"
    ".github/scripts/llk-get-docker-tag.sh"
)

# Combine hashes from both submodule and parent repo files
SUBMODULE_HASH=$(sha256sum "${DOCKERFILE_HASH_FILES[@]}" | sha256sum | cut -d ' ' -f 1)
PARENT_HASH=$(sha256sum "${METAL_BUILD_FILES[@]}" | sha256sum | cut -d ' ' -f 1)
COMBINED_HASH=$(echo "$SUBMODULE_HASH $PARENT_HASH" | sha256sum | cut -d ' ' -f 1)

echo "dt-$COMBINED_HASH"
