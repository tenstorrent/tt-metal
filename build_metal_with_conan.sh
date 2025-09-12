#!/usr/bin/env bash

CONAN_INSTALL_DIR="$1"
DISTRO="$2"

source $HOME/tt-metal-virtualenv/bin/activate
source "${CONAN_INSTALL_DIR}/conanbuild.sh"

case "$DISTRO" in
    "ubuntu")
        ./build_metal.sh --conan-install-path "${CONAN_INSTALL_DIR}"
        ;;
    "debian")
        ./build_metal.sh --conan-install-path "${CONAN_INSTALL_DIR}" --toolchain-path "cmake/x86_64-linux-gcc-12-toolchain.cmake"
        ;;
    "fedora")
        ./build_metal.sh --conan-install-path "${CONAN_INSTALL_DIR}"
        ;;
    "rhel")
        ./build_metal.sh --conan-install-path "${CONAN_INSTALL_DIR}" --toolchain-path "cmake/x86_64-linux-clang-19-libstdcpp-toolchain.cmake"
        ;;
    *)
        echo "Error: COMPILER must be 'clang_17' or 'gcc_12', got: '$COMPILER'"
        exit 1
        ;;
esac
