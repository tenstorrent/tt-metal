#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# LLK Docker images are built from the submodule content
LLK_PATH="tt_metal/third_party/tt_llk"

REPO=tenstorrent/tt-metal
BASE_IMAGE_NAME=ghcr.io/$REPO/tt-llk-base-ubuntu-22-04
CI_IMAGE_NAME=ghcr.io/$REPO/tt-llk-ci-ubuntu-22-04

# Compute the hash of the Dockerfile (run from LLK path since script uses relative paths)
DOCKER_TAG=$(cd $LLK_PATH && ./.github/scripts/get-docker-tag.sh)
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

    if docker manifest inspect $image_name:$DOCKER_TAG > /dev/null 2>&1; then
        echo "Image $image_name:$DOCKER_TAG already exists"

        # If we're on main, update the latest tag even if the image exists
        if [ "$on_main" = "true" ]; then
            # Check if latest already points to this tag (compare manifest digests)
            latest_digest=$(docker buildx imagetools inspect "$image_name:latest" 2>/dev/null | awk '/^Digest: / {print $2; exit}' || echo "")
            current_digest=$(docker buildx imagetools inspect "$image_name:$DOCKER_TAG" 2>/dev/null | awk '/^Digest: / {print $2; exit}')

            if [ "$latest_digest" != "$current_digest" ]; then
                echo "Updating latest tag for $image_name"
                docker buildx imagetools create -t $image_name:latest $image_name:$DOCKER_TAG
            else
                echo "Latest tag already points to $DOCKER_TAG"
            fi
        fi
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
        -f $dockerfile \
        $LLK_PATH
}

# Build base image from LLK submodule
build_and_push $BASE_IMAGE_NAME $LLK_PATH/.github/Dockerfile.base $ON_MAIN

# Build CI image from LLK submodule (depends on base)
build_and_push $CI_IMAGE_NAME $LLK_PATH/.github/Dockerfile.ci $ON_MAIN base

echo "All LLK images built and pushed successfully"
echo "CI_IMAGE_NAME:"
echo $CI_IMAGE_NAME:$DOCKER_TAG
