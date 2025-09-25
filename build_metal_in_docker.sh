#!/usr/bin/env bash

set -o nounset
set -x

DISTRO="$1"

# Map distro to conan profile
case "$DISTRO" in
    "debian")
        CONAN_PROFILE="conan_profile_gcc_12.txt"
        ;;
    "fedora")
        CONAN_PROFILE="conan_profile_clang_17.txt"
        ;;
    "rhel")
        CONAN_PROFILE="conan_profile_clang_19.txt"
        ;;
    "ubuntu")
        CONAN_PROFILE="conan_profile_clang_17.txt"
        ;;
    *)
        echo "Expected DISTRO as debian/fedora/rhel/ubuntu, got '${DISTRO}'"
        exit 1
        ;;
esac

BASE_IMAGE_NAME="tt-metal-base-${DISTRO}"
METAL_IMAGE_NAME="tt-metal-${DISTRO}"

# Step 1: Build base image with distro-specific dependencies
echo "Building base image: ${BASE_IMAGE_NAME}"
docker build --progress=plain -f "dockerfile/Dockerfile.conan-test.${DISTRO}" -t "${BASE_IMAGE_NAME}" .

# Step 2: Build tt-metal image using the base
echo "Building tt-metal image: ${METAL_IMAGE_NAME}"
docker build --progress=plain \
    --build-arg "BASE_IMAGE=${BASE_IMAGE_NAME}" \
    --build-arg "CONAN_PROFILE=${CONAN_PROFILE}" \
    -f "dockerfile/Dockerfile.conan-test" \
    -t "${METAL_IMAGE_NAME}" .

# Step 3: Run conan create in the container
echo "Running conan create in container"
docker run -it --rm -v "$(pwd):/workspace" "${METAL_IMAGE_NAME}" conan create . --build=missing
