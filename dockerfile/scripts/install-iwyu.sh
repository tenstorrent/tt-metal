#!/bin/bash
# Install include-what-you-use from source
set -euo pipefail

# IWYU links against Clang's internals, so a release only works with the
# matching Clang major version. 0.24 is the clang-20 release; bumping one
# without the other fails to configure.
IWYU_VERSION="${IWYU_VERSION:-0.24}"
IWYU_LLVM_MAJOR="${IWYU_LLVM_MAJOR:-20}"
# SHA256 for the include-what-you-use 0.24 source tarball
# Verified by downloading and computing hash with compute-hashes.sh
IWYU_SHA256="${IWYU_SHA256:-897b4c864a983f493c8efef4a1a9a2d429fd7ead1011f7a41743ed7b6dbe8c2e}"

INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
IWYU_LLVM_PREFIX="${IWYU_LLVM_PREFIX:-/usr/lib/llvm-${IWYU_LLVM_MAJOR}}"
DOWNLOAD_URL="https://github.com/include-what-you-use/include-what-you-use/archive/refs/tags/${IWYU_VERSION}.tar.gz"
TMPDIR="/tmp/iwyu"
CLANG_RESOURCE_DIR="${IWYU_LLVM_PREFIX}/lib/clang/${IWYU_LLVM_MAJOR}"

echo "Installing include-what-you-use ${IWYU_VERSION} (LLVM ${IWYU_LLVM_MAJOR})..."

# Create temp directory
mkdir -p "${TMPDIR}"

# Download (use curl if wget not available)
if command -v wget &> /dev/null; then
    wget -q -O "${TMPDIR}/iwyu.tar.gz" "${DOWNLOAD_URL}"
else
    curl -fsSL -o "${TMPDIR}/iwyu.tar.gz" "${DOWNLOAD_URL}"
fi

# Verify hash
if ! echo "${IWYU_SHA256}  ${TMPDIR}/iwyu.tar.gz" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for iwyu.tar.gz. Aborting." >&2
    exit 1
fi

# Extract
tar -xzf "${TMPDIR}/iwyu.tar.gz" -C "${TMPDIR}" --strip-components=1

# Create install prefix directory
mkdir -p "${INSTALL_PREFIX}"

# Build with CMake.
#
# IWYU needs Clang's builtin headers (stddef.h and friends) at runtime. Its
# default is to locate them next to a `clang` on PATH, but the LLVM apt
# packages install only versioned names, so there is no `clang` to find. Point
# the lookup at IWYU's own tree instead and ship the headers there, which is
# the packaging mode IWYU documents for exactly this case.
cmake -S "${TMPDIR}" -B "${TMPDIR}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_PREFIX_PATH="${IWYU_LLVM_PREFIX}" \
    -DIWYU_RESOURCE_RELATIVE_TO=iwyu \
    -DIWYU_RESOURCE_DIR="../lib/clang/${IWYU_LLVM_MAJOR}"
cmake --build "${TMPDIR}/build" -j"$(nproc)"
cmake --install "${TMPDIR}/build"

# Ship the builtin headers at the path configured above. Only include/ is
# needed; the rest of Clang's resource dir is compiler-rt runtime libraries.
mkdir -p "${INSTALL_PREFIX}/lib/clang/${IWYU_LLVM_MAJOR}"
cp -r "${CLANG_RESOURCE_DIR}/include" "${INSTALL_PREFIX}/lib/clang/${IWYU_LLVM_MAJOR}/include"

# Cleanup
rm -rf "${TMPDIR}"

# Verify installation, including that the builtin headers resolve without a
# `clang` on PATH.
"${INSTALL_PREFIX}/bin/include-what-you-use" --version
printf '#include <stddef.h>\nsize_t f(void) { return 0; }\n' > /tmp/iwyu-check.c
env -i PATH=/usr/bin:/bin "${INSTALL_PREFIX}/bin/include-what-you-use" -xc /tmp/iwyu-check.c 2>&1 \
    | grep -q 'has correct #includes' \
    || { echo "[ERROR] iwyu cannot find its builtin headers. Aborting." >&2; exit 1; }
rm -f /tmp/iwyu-check.c
echo "include-what-you-use ${IWYU_VERSION} installed successfully"
