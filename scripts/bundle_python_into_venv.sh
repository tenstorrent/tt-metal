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

# --- Locate the cpython source directory ---
# Strategy 1: follow symlinks from the venv's python binary.
# Strategy 2: read 'home' from pyvenv.cfg (for --link-mode copy venvs where
#             the binary is already a real file, not a symlink).
CPYTHON_DIR=""

if [[ -L "$VENV_DIR/bin/python" || -L "$VENV_DIR/bin/python3" ]]; then
    # Binary is a symlink — resolve it to find the cpython install.
    local_link="$VENV_DIR/bin/python"
    [[ -L "$local_link" ]] || local_link="$VENV_DIR/bin/python3"

    echo "  Symlink target: $(readlink "$local_link")"

    if ! REAL_PYTHON_PATH=$(readlink -f "$local_link" 2>&1); then
        echo "ERROR: Failed to resolve Python symlink" >&2
        echo "  readlink error: $REAL_PYTHON_PATH" >&2
        echo "" >&2
        echo "This typically means Python was installed to an inaccessible location" >&2
        echo "(e.g., /root/.local/share/uv/) during environment setup." >&2
        echo "Fix: Set UV_PYTHON_INSTALL_DIR=/usr/local/share/uv" >&2
        exit 1
    fi
    echo "  Python interpreter (resolved): $REAL_PYTHON_PATH"
    CPYTHON_DIR=$(dirname "$(dirname "$REAL_PYTHON_PATH")")
else
    # Binary is a regular file (--link-mode copy / --relocatable).
    # Fall back to pyvenv.cfg's 'home' key to locate the cpython source.
    echo "  Python binary is a regular file (not a symlink)"
    if [[ -f "$VENV_DIR/pyvenv.cfg" ]]; then
        CFG_HOME=$(grep -E '^home\s*=' "$VENV_DIR/pyvenv.cfg" | head -1 | sed 's/^home\s*=\s*//')
        if [[ -n "$CFG_HOME" && -d "$CFG_HOME" ]]; then
            # home points to .../bin; cpython root is one level up.
            CPYTHON_DIR=$(dirname "$CFG_HOME")
            echo "  CPython directory (from pyvenv.cfg home): $CPYTHON_DIR"
        else
            echo "  pyvenv.cfg home='${CFG_HOME:-<not set>}' — directory does not exist"
        fi
    fi
fi

# Validate the cpython source directory
if [[ -z "$CPYTHON_DIR" || ! -d "$CPYTHON_DIR/lib" ]]; then
    # If we can't find the cpython source but the stdlib is already present
    # in the venv, we only need to fix pyvenv.cfg (handled below).
    if ls "$VENV_DIR"/lib/python3*/os.py &>/dev/null; then
        echo "  CPython source not reachable, but stdlib already present in venv"
        echo "  Skipping file copy; will update pyvenv.cfg only"
        CPYTHON_DIR=""
    else
        echo "ERROR: Cannot locate CPython source directory and stdlib is not in venv" >&2
        echo "  Tried: symlink resolution and pyvenv.cfg home key" >&2
        echo "  The venv may have been created with --link-mode copy on a different host." >&2
        echo "  Re-create with:  create_venv.sh --bundle-python" >&2
        exit 1
    fi
fi

# Copy cpython files into the venv (if we found the source)
if [[ -n "$CPYTHON_DIR" ]]; then
    VENV_DIR_REAL=$(cd "$VENV_DIR" && pwd)
    CPYTHON_DIR_REAL=$(cd "$CPYTHON_DIR" && pwd)
    if [[ "$CPYTHON_DIR_REAL" == "$VENV_DIR_REAL" ]]; then
        echo "  CPython source IS the venv itself (--link-mode copy); skipping self-copy"
    else
        if [[ ! -r "$CPYTHON_DIR/bin/python3" && ! -r "$CPYTHON_DIR/bin/python" ]]; then
            echo "ERROR: Python interpreter is not readable in $CPYTHON_DIR/bin/" >&2
            echo "Check that UV_PYTHON_INSTALL_DIR was set correctly during setup." >&2
            exit 1
        fi

        echo "  CPython directory: $CPYTHON_DIR"
        echo "  Removing venv python binaries..."
        rm -f "$VENV_DIR/bin/python"*

        echo "  Copying Python interpreter files into venv..."
        cp -r "$CPYTHON_DIR"/* "$VENV_DIR/"
    fi
fi

# Update pyvenv.cfg so 'home' points to the venv's own bin/ directory.
# Before bundling, home pointed to the original cpython install (e.g.
# /usr/local/share/uv/python/cpython-.../install/bin) which won't exist
# on NFS compute nodes.  Python uses the home path to locate its stdlib;
# if the path is invalid it falls back to the compiled-in prefix and
# fails with "No module named 'encodings'".
if [[ -f "$VENV_DIR/pyvenv.cfg" ]]; then
    VENV_BIN_DIR="$(cd "$VENV_DIR/bin" && pwd)"
    echo "  Updating pyvenv.cfg home -> $VENV_BIN_DIR"
    sed -i "s|^home\s*=.*|home = $VENV_BIN_DIR|" "$VENV_DIR/pyvenv.cfg"
fi

echo "  Python interpreter bundled successfully"
