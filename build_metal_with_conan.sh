#!/usr/bin/env bash

CONAN_INSTALL_DIR="$1"

source "${CONAN_INSTALL_DIR}/conanbuild.sh"
./build_metal.sh --conan-install-path "${CONAN_INSTALL_DIR}"
