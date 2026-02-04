#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

REPO=tenstorrent/tt-llk
BASE_IMAGE_NAME=ghcr.io/$REPO/tt-llk-base-ubuntu-22-04
CI_IMAGE_NAME=ghcr.io/$REPO/tt-llk-ci-ubuntu-22-04
BASE_IRD_IMAGE_NAME=ghcr.io/$REPO/tt-llk-base-ird-ubuntu-22-04
IRD_IMAGE_NAME=ghcr.io/$REPO/tt-llk-ird-ubuntu-22-04
SLIM_IRD_IMAGE_NAME=ghcr.io/$REPO/tt-llk-slim-ird-ubuntu-22-04

# Compute the hash of the Dockerfile
DOCKER_TAG=$(./.github/scripts/get-docker-tag.sh)
echo "Docker tag: $DOCKER_TAG"

# Are we on main branch
ON_MAIN=$(git branch --show-current | grep -q main && echo "true" || echo "false")

export DOCKER_BUILDKIT=1

# Ensure a buildx builder exists and is active
docker buildx create --use --name tt-builder >/dev/null 2>&1 || docker buildx use tt-builder
docker buildx inspect --bootstrap >/dev/null

build_and_push() {
    local image_name=$1
    local dockerfile=$2
    local on_main=$3
    local from_image=$4

    if docker manifest inspect $image_name:$DOCKER_TAG > /dev/null; then
        echo "Image $image_name:$DOCKER_TAG already exists"
        return
    fi

    echo "Building and pushing image $image_name:$DOCKER_TAG"

    if [ "$on_main" = "true" ]; then
        tags="-t $image_name:$DOCKER_TAG -t $image_name:latest"
    else
        tags="-t $image_name:$DOCKER_TAG"
    fi

    docker buildx build \
        --push \
        --output type=image,compression=zstd,oci-mediatypes=true \
        --build-arg FROM_TAG=$DOCKER_TAG \
        ${from_image:+--build-arg FROM_IMAGE=$from_image} \
        $tags \
        -f $dockerfile .
}

build_and_push $BASE_IMAGE_NAME .github/Dockerfile.base $ON_MAIN
build_and_push $BASE_IRD_IMAGE_NAME .github/Dockerfile.ird $ON_MAIN base
build_and_push $CI_IMAGE_NAME .github/Dockerfile.ci $ON_MAIN
build_and_push $IRD_IMAGE_NAME .github/Dockerfile.ird $ON_MAIN ci
build_and_push $SLIM_IRD_IMAGE_NAME .github/Dockerfile.ird.slim $ON_MAIN ci

echo "All images built and pushed successfully"
echo "CI_IMAGE_NAME:"
echo $CI_IMAGE_NAME:$DOCKER_TAG
