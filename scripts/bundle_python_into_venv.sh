#!/bin/bash
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
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

    This script deep-copies the Python interpreter into the venv's _python/
    subdirectory, making it fully self-contained and portable. This is necessary
    for multi-host SLURM environments where the venv may be shared over NFS and
    each host needs access to the Python interpreter without relying on symlinks
    to a per-host managed Python installation.

    Python binaries in bin/ are replaced with relative symlinks into _python/bin/,
    and pyvenv.cfg.home is rewritten to match. Using a subdirectory (not the venv
    root) ensures base_prefix (_python/) != prefix (venv root), which is required
    for uv, pip, and Python's own site.py to recognise this as a virtual environment.

    Usage: $0 <venv_dir> [--force]

    Arguments:
      venv_dir    Path to the virtual environment directory
      --force     Re-bundle even if _python/ already exists (optional)

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

BUNDLED_PYTHON_DIR="$VENV_DIR/_python"

# Bundling is needed if the _python/ subdirectory does not yet exist.
needs_bundling() {
    [[ ! -d "$BUNDLED_PYTHON_DIR" ]]
}

if [[ "$FORCE_BUNDLE" != "--force" ]] && ! needs_bundling; then
    echo "INFO: Python interpreter already bundled at $BUNDLED_PYTHON_DIR, skipping"
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

# Copy CPython into a temporary location first. This is necessary because on a
# --force re-run against an already-bundled venv, bin/python is a relative symlink
# (../_python/bin/python) that readlink -f resolves to $VENV_DIR/_python/bin/python,
# making CPYTHON_DIR == BUNDLED_PYTHON_DIR. Copying to a temp dir before removing
# anything means the source is always intact when the copy runs.
echo "  Copying Python interpreter into $BUNDLED_PYTHON_DIR/..."
TMP_DIR="${BUNDLED_PYTHON_DIR}.tmp"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
cp -r "$CPYTHON_DIR"/* "$TMP_DIR/"

# Remove Python symlinks in venv/bin/ (they point to either the external managed
# install or, on re-bundle, into the _python/ dir that is about to be replaced).
echo "  Removing Python symlinks from $VENV_DIR/bin/..."
rm -f "$VENV_DIR/bin/python"*

# Atomically replace _python/ with the new copy.
# We deliberately do NOT copy into the venv root. If we did, base_prefix would
# equal prefix ($VENV_DIR), which causes Python, uv, and pip to conclude that
# this is NOT a virtual environment — the same symptom as the bug we are fixing.
# Bundling into _python/ keeps:
#   base_prefix = $VENV_DIR/_python   (derived from pyvenv.cfg.home)
#   prefix      = $VENV_DIR
#   base_prefix != prefix  =>  venv is correctly recognised.
rm -rf "$BUNDLED_PYTHON_DIR"
mv "$TMP_DIR" "$BUNDLED_PYTHON_DIR"

# Create relative symlinks in venv/bin/ that point to the bundled interpreter.
# Relative symlinks remain valid when the venv is mounted at any NFS path —
# the link resolves within the venv regardless of the absolute mount point.
echo "  Creating relative symlinks to bundled interpreter..."
for py_exec in python python3; do
    [[ -f "$BUNDLED_PYTHON_DIR/bin/$py_exec" ]] && \
        ln -sf "../_python/bin/$py_exec" "$VENV_DIR/bin/$py_exec"
done
for versioned_py in "$BUNDLED_PYTHON_DIR/bin/python3".[0-9]*; do
    [[ -f "$versioned_py" ]] || continue
    py_name=$(basename "$versioned_py")
    ln -sf "../_python/bin/$py_name" "$VENV_DIR/bin/$py_name"
done

# Rewrite pyvenv.cfg.home to point at the bundled interpreter location.
#
# Before bundling: home = /usr/local/share/uv/cpython-3.12.X.../bin
#   (the external managed install; no longer accessible after Docker build or on
#    SLURM worker nodes that lack the uv managed Python installation)
#
# After bundling:  home = <venv>/_python/bin
#   (bundled inside the venv; always accessible, relative to the venv root)
#
# uv runs Interpreter::query against venv/bin/python to validate the environment.
# If home points to a path Python cannot locate its stdlib from, the query fails
# and uv reports "No virtual environment found" even though VIRTUAL_ENV is set.
PYVENV_CFG="$VENV_DIR/pyvenv.cfg"
if [[ -f "$PYVENV_CFG" ]]; then
    awk -v home="$BUNDLED_PYTHON_DIR/bin" '
        /^home = /  { print "home = " home; next }
        { print }
    ' "$PYVENV_CFG" > "$PYVENV_CFG.tmp" && mv "$PYVENV_CFG.tmp" "$PYVENV_CFG"
    echo "  Updated pyvenv.cfg: home = $BUNDLED_PYTHON_DIR/bin"
else
    echo "WARNING: $PYVENV_CFG not found — venv may not be recognised correctly" >&2
fi

echo "  Python interpreter bundled successfully"
