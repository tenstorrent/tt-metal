#!/usr/bin/env bash

set -o nounset
set -o errexit
set -x

DISTRO="$1"

BASE_IMAGE_NAME="tt-metal-base-${DISTRO}"
METAL_IMAGE_NAME="tt-metal-${DISTRO}"

# Step 1: Build base image with distro-specific dependencies
echo "Building base image: ${BASE_IMAGE_NAME}"
docker build --progress=plain -f "dockerfile/Dockerfile.conan-test.${DISTRO}" -t "${BASE_IMAGE_NAME}" .

# Step 2: Build tt-metal image using the base
echo "Building tt-metal image: ${METAL_IMAGE_NAME}"
docker build --progress=plain \
    --build-arg "BASE_IMAGE=${BASE_IMAGE_NAME}" \
    -f "dockerfile/Dockerfile.conan-test" \
    -t "${METAL_IMAGE_NAME}" .

# Step 3: Run conan create in the container
echo "Running conan create in container"
docker run -it --rm -v "$(pwd):/workspace" \
    --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G \
    "${METAL_IMAGE_NAME}" ./run_metal_with_conan.sh "/tt-conan/default_cxx20.txt"
