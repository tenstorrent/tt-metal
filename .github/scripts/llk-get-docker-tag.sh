#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Calculate hash from the following files. This hash is used to tag the docker images.
# Any change in these files will result in a new docker image build.

LLK_PATH="${LLK_PATH:-tt_metal/tt-llk}"

LLK_FILES=(
    "${LLK_PATH}/.github/Dockerfile.base"
    "${LLK_PATH}/.github/Dockerfile.ci"
    "${LLK_PATH}/.github/Dockerfile.ird"
    "${LLK_PATH}/.github/Dockerfile.ird.slim"
    "${LLK_PATH}/tests/requirements.txt"
    "${LLK_PATH}/.github/scripts/install-tests-dependencies.sh"
)

METAL_BUILD_FILES=(
    ".github/scripts/llk-build-docker-images.sh"
    ".github/scripts/llk-get-docker-tag.sh"
)

LLK_HASH=$(sha256sum "${LLK_FILES[@]}" | sha256sum | cut -d ' ' -f 1)
PARENT_HASH=$(sha256sum "${METAL_BUILD_FILES[@]}" | sha256sum | cut -d ' ' -f 1)
COMBINED_HASH=$(echo "$LLK_HASH $PARENT_HASH" | sha256sum | cut -d ' ' -f 1)

echo "dt-$COMBINED_HASH"
