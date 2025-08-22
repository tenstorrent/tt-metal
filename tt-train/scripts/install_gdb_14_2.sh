# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash
# install-gdb-14.2.sh
# Build & install GDB 14.2 on Ubuntu into /opt/gdb-14.2
# Creates /usr/local/bin/gdb-14.2 symlink

set -euo pipefail

GDB_VERSION="14.2"
PREFIX="/opt/gdb-${GDB_VERSION}"
SYMLINK="/usr/local/bin/gdb-${GDB_VERSION}"
SRC_URL="https://ftp.gnu.org/gnu/gdb/gdb-${GDB_VERSION}.tar.xz"

sudo apt update
sudo apt install -y \
  build-essential texinfo bison flex \
  libreadline-dev libncurses-dev zlib1g-dev libexpat1-dev python3-dev \
  pkg-config libgmp-dev libmpfr-dev libmpc-dev

INSTALL_GDBSERVER=0
if [[ "${1-}" == "--gdbserver" ]]; then
  INSTALL_GDBSERVER=1
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1"; exit 1; }
}

echo "[*] Checking prerequisites..."
require_cmd sudo
require_cmd wget
require_cmd tar

if command -v apt >/dev/null 2>&1; then
  echo "[*] Installing build dependencies with apt..."
  sudo apt update
  sudo apt install -y \
    build-essential texinfo bison flex \
    libreadline-dev libncurses-dev zlib1g-dev libexpat1-dev python3-dev
else
  echo "[!] 'apt' not found. This script targets Ubuntu/Debian."
  echo "    Install equivalent packages for your distro, then re-run."
  exit 1
fi

if [[ -x "${PREFIX}/bin/gdb" ]]; then
  echo "[✓] GDB ${GDB_VERSION} already installed at ${PREFIX}."
  echo "    Symlink step next..."
else
  TMPDIR="$(mktemp -d)"
  trap 'rm -rf "$TMPDIR"' EXIT

  echo "[*] Downloading GDB ${GDB_VERSION}..."
  cd "$TMPDIR"
  wget -q --show-progress "${SRC_URL}"

  echo "[*] Extracting..."
  tar -xf "gdb-${GDB_VERSION}.tar.xz"
  cd "gdb-${GDB_VERSION}"

  echo "[*] Configuring..."
  mkdir -p build && cd build
  ../configure \
    --prefix="${PREFIX}" \
    --with-system-readline \
    --with-python=/usr/bin/python3

  echo "[*] Building (this may take a while)..."
  CORES="$(nproc 2>/dev/null || echo 1)"
  make -j"${CORES}"

  echo "[*] Installing to ${PREFIX}..."
  sudo make install

  if [[ "${INSTALL_GDBSERVER}" -eq 1 ]]; then
    echo "[*] Installing gdbserver..."
    sudo make -C ../gdbserver install
  else
    echo "[*] Skipping gdbserver installation."
  fi
fi

echo "[*] Creating/updating symlink ${SYMLINK} -> ${PREFIX}/bin/gdb"
sudo ln -sf "${PREFIX}/bin/gdb" "${SYMLINK}"

echo "[*] Verifying..."
"${SYMLINK}" --version | head -n1 || {
  echo "[!] Verification failed: ${SYMLINK} not working."
  exit 1
}

echo
echo "==============================================="
echo " GDB ${GDB_VERSION} installed successfully!"
echo " Binary: ${PREFIX}/bin/gdb"
echo " Symlink: ${SYMLINK}"
echo
echo " Use this path in VS Code:"
echo "   \"miDebuggerPath\": \"${SYMLINK}\""
echo "==============================================="
