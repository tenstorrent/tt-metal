#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o xtrace

CONAN_INSTALL_DIR="$1"

echo "INFO: CONAN_INSTALL_DIR = ${CONAN_INSTALL_DIR}"

conan profile detect --force
conan install tool_conanfile.txt "--output-folder=${CONAN_INSTALL_DIR}" --build-require --build=missing -c tools.cmake.cmakedeps:new=recipe_will_break --profile conan_profile.txt
cat "${CONAN_INSTALL_DIR}/conanbuild.sh"
source "${CONAN_INSTALL_DIR}/conanbuild.sh"
conan install conanfile.txt "--output-folder=${CONAN_INSTALL_DIR}" --build-require --build=missing -c tools.cmake.cmakedeps:new=recipe_will_break --profile conan_profile.txt
ls "${CONAN_INSTALL_DIR}"
cmake --version
which cmake
ninja --version
which ninja
echo "$PATH"
