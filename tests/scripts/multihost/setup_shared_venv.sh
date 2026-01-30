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
  fi
) 9>"${lockfile}"
