#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# MIGRATION NOTE: These images are staging for LLK-in-Metal; not consumed by CI yet.
# tt_llk/.github/Dockerfile.ci still FROM ghcr.io/tenstorrent/tt-llk/... while this
# script pushes to ghcr.io/$GITHUB_REPOSITORY/... — reconcile when CI starts using
# these images or when LLK moves in-tree, then remove this note.

set -euo pipefail

# Base for Dockerfile.base (patches FROM ubuntu:22.04 in a temp file; tt_llk submodule stays vanilla).
# Default matches upstream tt_llk. CI uses mirror.gcr.io for reliable Docker Hub access.
#   LLK_UBUNTU_BASE_IMAGE=mirror.gcr.io/ubuntu:22.04
LLK_UBUNTU_BASE_IMAGE="${LLK_UBUNTU_BASE_IMAGE:-ubuntu:22.04}"

# LLK Docker images are built from the submodule content
LLK_PATH="tt_metal/third_party/tt_llk"
if [[ ! -d "$LLK_PATH" || ! -f "$LLK_PATH/.github/Dockerfile.base" ]]; then
  echo "::error::tt_llk submodule is missing or not checked out (expected $LLK_PATH with .github/Dockerfile.base)." >&2
  exit 1
fi

REPO="${GITHUB_REPOSITORY:-tenstorrent/tt-metal}"
BASE_IMAGE_NAME=ghcr.io/$REPO/tt-llk-base-ubuntu-22-04
CI_IMAGE_NAME=ghcr.io/$REPO/tt-llk-ci-ubuntu-22-04

LLK_BASE_DOCKERFILE_PATCHED=$(mktemp)
LLK_CI_DOCKERFILE_PATCHED=$(mktemp)
trap 'rm -f "${LLK_BASE_DOCKERFILE_PATCHED}" "${LLK_CI_DOCKERFILE_PATCHED}"' EXIT

# Patch Dockerfile.base to use the specified Ubuntu base image
awk -v img="$LLK_UBUNTU_BASE_IMAGE" '
  $0 == "FROM ubuntu:22.04" { print "FROM " img; next }
  { print }
' "$LLK_PATH/.github/Dockerfile.base" >"$LLK_BASE_DOCKERFILE_PATCHED"
echo "LLK Dockerfile.base rootfs: ${LLK_UBUNTU_BASE_IMAGE}"

# Patch Dockerfile.ci to use the Metal repo's base image location instead of tt-llk
awk -v base_img="$BASE_IMAGE_NAME" '
  /^FROM ghcr\.io\/tenstorrent\/tt-llk\/tt-llk-base-ubuntu-22-04:/ {
    sub(/ghcr\.io\/tenstorrent\/tt-llk\/tt-llk-base-ubuntu-22-04/, base_img)
  }
  { print }
' "$LLK_PATH/.github/Dockerfile.ci" >"$LLK_CI_DOCKERFILE_PATCHED"
echo "LLK Dockerfile.ci base image: ${BASE_IMAGE_NAME}"

# Compute the hash of the Dockerfile (uses migrated script in parent repo)
# MIGRATION: This now uses .github/scripts/llk-get-docker-tag.sh instead of
# the submodule's get-docker-tag.sh, so changes to this script will trigger rebuilds
DOCKER_TAG=$(./.github/scripts/llk-get-docker-tag.sh)
echo "Docker tag: $DOCKER_TAG"

# Are we on main branch - use GITHUB_REF_NAME if available (GitHub Actions), otherwise fall back to git
if [ -n "${GITHUB_REF_NAME-}" ]; then
    ON_MAIN=$([ "${GITHUB_REF_NAME}" = "main" ] && echo "true" || echo "false")
else
    ON_MAIN=$(git branch --show-current 2>/dev/null | grep -q main && echo "true" || echo "false")
fi

export DOCKER_BUILDKIT=1

build_and_push() {
    local image_name=$1
    local dockerfile=$2
    local on_main=$3

    local prefixed_image="${HARBOR_PREFIX:-}${image_name}"

    if docker manifest inspect "${prefixed_image}:${DOCKER_TAG}" > /dev/null 2>&1; then
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

    local cache_from_flags=""
    if [ -n "${HARBOR_PREFIX:-}" ]; then
        cache_from_flags="--cache-from type=registry,ref=${prefixed_image}:${DOCKER_TAG}"
    fi

    docker buildx build \
        --output type=image,compression=zstd,oci-mediatypes=true,push=true \
        --build-arg FROM_TAG=$DOCKER_TAG \
        $cache_from_flags \
        $tags \
        -f $dockerfile \
        $LLK_PATH
}

# Build base image (patched Dockerfile: see LLK_UBUNTU_BASE_IMAGE above)
build_and_push "$BASE_IMAGE_NAME" "$LLK_BASE_DOCKERFILE_PATCHED" "$ON_MAIN"

# Build CI image from LLK submodule (using patched Dockerfile.ci with correct base image reference)
build_and_push "$CI_IMAGE_NAME" "$LLK_CI_DOCKERFILE_PATCHED" "$ON_MAIN"

echo "All LLK images built and pushed successfully"
echo "CI_IMAGE_NAME:"
echo "$CI_IMAGE_NAME:$DOCKER_TAG"

# Output for GitHub Actions (if running in GHA)
if [ -n "${GITHUB_OUTPUT-}" ]; then
    echo "docker-image=$CI_IMAGE_NAME:$DOCKER_TAG" >> "$GITHUB_OUTPUT"
fi
