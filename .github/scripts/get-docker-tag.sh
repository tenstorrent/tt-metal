#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Calculate hash from the following files. This hash is used to tag the docker images.
# Any change in these files will result in a new docker image build
DOCKERFILE_HASH_FILES=".github/Dockerfile.base \
    .github/Dockerfile.ci \
    .github/Dockerfile.ird \
    tests/requirements.txt \
    .github/scripts/build-docker-images.sh \
    .github/scripts/install-exalens.sh \
    .github/scripts/install-smi.sh \
    .github/scripts/install-tests-dependencies.sh"
DOCKERFILE_HASH=$(sha256sum $DOCKERFILE_HASH_FILES | sha256sum | cut -d ' ' -f 1)
echo dt-$DOCKERFILE_HASH
