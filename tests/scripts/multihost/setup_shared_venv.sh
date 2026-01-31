#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Setup shared Python venv for multi-host MPI scenarios.
# This script handles race conditions when multiple hosts may access
# the same shared workspace simultaneously by using file locking
# and atomic directory creation.

set -eo pipefail

SOURCE_VENV="${1:-/opt/venv}"
TARGET_VENV="${2:-./python_env}"

# Deactivate container's pre-activated venv if active
deactivate 2>/dev/null || true

# Use a file lock and atomic directory creation to avoid multi-host race conditions.
lockfile="${TARGET_VENV}.lock"
tmp_env_dir="${TARGET_VENV}.tmp.$$"

(
  # Try to acquire the lock non-blocking; if it is held, wait.
  flock -n 9 || {
    echo "INFO: Another host is preparing ${TARGET_VENV}, waiting for lock..."
    flock 9
  }
  if [ -d "${TARGET_VENV}" ]; then
    echo "INFO: ${TARGET_VENV} already exists, skipping copy"
  else
    echo "INFO: Creating ${TARGET_VENV} via temporary directory ${tmp_env_dir}"
    cp -a "${SOURCE_VENV}" "${tmp_env_dir}"
    mv "${tmp_env_dir}" "${TARGET_VENV}"

    # Check if Python interpreter needs to be bundled by checking if any python executables are symlinks
    NEEDS_BUNDLING=false
    for py_exec in "$TARGET_VENV/bin/python" "$TARGET_VENV/bin/python3" "$TARGET_VENV/bin/python3".*; do
      if [ -L "$py_exec" ]; then
        NEEDS_BUNDLING=true
        break
      fi
    done

    if [ "$NEEDS_BUNDLING" = true ]; then
      echo "INFO: Python interpreter is symlinked, bundling into venv..."

      # Get the real path to the Python interpreter
      REAL_PYTHON_PATH=$(readlink -f "$TARGET_VENV/bin/python")
      echo "  Python interpreter: $REAL_PYTHON_PATH"

      # Extract the cpython installation directory (parent of bin/python)
      # Path is in form: <prefix>/python/cpython<version>/bin/python
      CPYTHON_DIR=$(dirname "$(dirname "$REAL_PYTHON_PATH")")
      echo "  CPython directory: $CPYTHON_DIR"

      # Remove python symlinks in venv (they will be replaced with bundled interpreter)
      echo "  Removing venv python symlinks..."
      rm -f "$TARGET_VENV/bin/python"*

      # Copy the cpython directory contents into the venv
      echo "  Copying Python interpreter files into venv..."
      cp -r "$CPYTHON_DIR"/* "$TARGET_VENV/"

      echo "  Python interpreter bundled successfully"
    else
      echo "INFO: Python interpreter is already bundled, skipping bundling step"
    fi
  fi
) 9>"${lockfile}"
