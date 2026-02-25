#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: If this script is moved, update tests/scripts/multihost/setup_shared_venv.sh
# which references it via BUNDLE_SCRIPT="${SCRIPT_DIR}/../../../scripts/bundle_python_into_venv.sh"
#
# Target platforms: Linux (Docker, CI, SLURM multi-host). Uses readlink -f (Linux-specific).
# Not tested on macOS or other Unix variants.

set -eo pipefail

# This script requires Linux: readlink -f and uv's Python layout are Linux-specific.
# Targeted at Docker/CI and SLURM environments. On macOS, exit with a clear message.
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: $0 is designed for Linux (Docker, CI, SLURM). macOS is not supported." >&2
  exit 1
fi

usage() {
    cat <<EOF
    Bundle Python interpreter into a virtual environment.

    This script deep-copies the Python interpreter into the venv, making it
    fully self-contained and portable. This is necessary for multi-host SLURM
    environments where the venv may be shared over NFS and each host needs
    access to the Python interpreter without relying on symlinks to a Docker
    container's local filesystem.

    Usage: $0 <venv_dir> [--force]

    Arguments:
      venv_dir    Path to the virtual environment directory
      --force     Bundle even if Python is not symlinked (optional)

    The script uses uv's Python installation structure:
      <UV_PYTHON_INSTALL_DIR>/cpython-<version>-<platform>/bin/python

    Exit codes:
      0 - Success (bundled or already bundled)
      1 - Error (missing venv, permission denied, etc.)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

VENV_DIR="${1:-}"
FORCE_BUNDLE="${2:-}"

if [[ -z "$VENV_DIR" ]]; then
    usage >&2
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Virtual environment directory does not exist: $VENV_DIR" >&2
    exit 1
fi

# Check if Python interpreter needs to be bundled by checking if any python executables are symlinks
needs_bundling() {
    local venv_dir="$1"
    (
        shopt -s nullglob
        for py_exec in "$venv_dir/bin/python" "$venv_dir/bin/python3" "$venv_dir/bin/python3".*; do
            if [[ -e "$py_exec" && -L "$py_exec" ]]; then
                return 0  # true - needs bundling
            fi
        done
        return 1  # false - already bundled
    )
}

if [[ "$FORCE_BUNDLE" != "--force" ]] && ! needs_bundling "$VENV_DIR"; then
    echo "INFO: Python interpreter is already bundled in $VENV_DIR, skipping"
    exit 0
fi

echo "Bundling Python interpreter into venv: $VENV_DIR"

# Get symlink target for diagnostic purposes
if [[ -L "$VENV_DIR/bin/python" ]]; then
    SYMLINK_TARGET=$(readlink "$VENV_DIR/bin/python")
    echo "  Symlink target: $SYMLINK_TARGET"
fi

# Get the real path to the Python interpreter
# Note: If this fails with "Permission denied", the source Python installation
# is likely in an inaccessible location (e.g., /root/.local/share/uv).
# Fix: Ensure UV_PYTHON_INSTALL_DIR is set to an accessible location like
# /usr/local/share/uv before running 'uv python install'.
if ! REAL_PYTHON_PATH=$(readlink -f "$VENV_DIR/bin/python" 2>&1); then
    echo "ERROR: Failed to resolve Python symlink" >&2
    echo "  readlink error: $REAL_PYTHON_PATH" >&2
    if [[ -n "${SYMLINK_TARGET:-}" ]]; then
        echo "  Symlink points to: $SYMLINK_TARGET" >&2
    fi
    echo "" >&2
    echo "This typically means Python was installed to an inaccessible location" >&2
    echo "(e.g., /root/.local/share/uv/) during environment setup." >&2
    echo "" >&2
    echo "Fix: Set UV_PYTHON_INSTALL_DIR=/usr/local/share/uv (or another accessible" >&2
    echo "location) before running 'uv python install'." >&2
    exit 1
fi
echo "  Python interpreter: $REAL_PYTHON_PATH"

# Verify the Python interpreter is accessible
if [[ ! -r "$REAL_PYTHON_PATH" ]]; then
    echo "ERROR: Python interpreter is not readable: $REAL_PYTHON_PATH" >&2
    echo "Check that UV_PYTHON_INSTALL_DIR was set correctly during setup." >&2
    exit 1
fi

# Extract the cpython installation directory (parent of bin/python)
# Path structure from uv: <UV_PYTHON_INSTALL_DIR>/cpython-<version>-<platform>/bin/python
CPYTHON_DIR=$(dirname "$(dirname "$REAL_PYTHON_PATH")")
echo "  CPython directory: $CPYTHON_DIR"

# Verify the CPython directory looks valid
if [[ ! -d "$CPYTHON_DIR/lib" ]]; then
    echo "ERROR: CPython directory does not contain expected structure: $CPYTHON_DIR" >&2
    echo "Expected to find 'lib' subdirectory." >&2
    exit 1
fi

# Remove python symlinks in venv (they will be replaced with bundled interpreter)
echo "  Removing venv python symlinks..."
rm -f "$VENV_DIR/bin/python"*

# Copy the cpython directory contents into the venv
echo "  Copying Python interpreter files into venv..."
cp -r "$CPYTHON_DIR"/* "$VENV_DIR/"

echo "  Python interpreter bundled successfully"
