#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


set -eo pipefail

# Parse arguments
ACTIVATE_MODE=false
POSITIONAL_ARGS=()

usage() {
    cat <<USAGE
    Setup shared Python venv for multi-host MPI scenarios.
    This script handles race conditions when multiple hosts may access
    the same shared workspace simultaneously by using file locking
    and atomic directory creation.

    Usage:
      $0 [OPTIONS] [SOURCE_VENV] [TARGET_VENV]

    Options:
      -h, --help    Show this help message and exit
      --activate    Output shell commands to activate the venv (for use with eval)

    Arguments:
      SOURCE_VENV   Source venv to copy from (default: /opt/venv)
      TARGET_VENV   Target venv path (default: ./python_env)

    Examples:
      # Just setup the shared venv (no activation)
      $0

      # Setup and activate the shared venv in the caller's shell
      eval "\$($0 --activate)"

    The --activate flag outputs shell commands that, when eval'd, switch the
    caller's active venv. This is required because environment changes made
    in a subshell don't propagate to the parent.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --activate)
            ACTIVATE_MODE=true
            shift
            ;;
        -*)
            echo "ERROR: Unknown option: $1" >&2
            echo "Usage: $0 [--activate] [SOURCE_VENV] [TARGET_VENV]" >&2
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set positional arguments with defaults
SOURCE_VENV="${POSITIONAL_ARGS[0]:-/opt/venv}"
TARGET_VENV="${POSITIONAL_ARGS[1]:-./python_env}"

# Send diagnostic output to stderr when in activate mode (so stdout is clean for eval)
# Otherwise send to stdout for normal script behavior
if [[ "$ACTIVATE_MODE" == "true" ]]; then
    log() { echo "$@" >&2; }
else
    log() { echo "$@"; }
fi

# Deactivate container's pre-activated venv if active
deactivate 2>/dev/null || true

# Use a file lock and atomic directory creation to avoid multi-host race conditions.
lockfile="${TARGET_VENV}.lock"
tmp_env_dir="${TARGET_VENV}.tmp.$$"

(
  # Try to acquire the lock non-blocking; if it is held, wait.
  flock -n 9 || {
    log "INFO: Another host is preparing ${TARGET_VENV}, waiting for lock..."
    flock 9
  }
  if [ -d "${TARGET_VENV}" ]; then
    log "INFO: ${TARGET_VENV} already exists, skipping copy"
  else
    log "INFO: Creating ${TARGET_VENV} via temporary directory ${tmp_env_dir}"
    cp -a "${SOURCE_VENV}" "${tmp_env_dir}"
    mv "${tmp_env_dir}" "${TARGET_VENV}"

    # Bundle Python interpreter if needed (uses shared script)
    # The script checks if bundling is needed and handles errors gracefully
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BUNDLE_SCRIPT="${SCRIPT_DIR}/../../../scripts/bundle_python_into_venv.sh"
    PATCH_SCRIPT="${SCRIPT_DIR}/../../../scripts/patch_activate_posix.sh"
    if [ -x "$BUNDLE_SCRIPT" ]; then
      "$BUNDLE_SCRIPT" "${TARGET_VENV}" >&2
    else
      log "WARNING: bundle_python_into_venv.sh not found at $BUNDLE_SCRIPT"
      log "Python interpreter may not be properly bundled for multi-host use."
    fi
    # Re-patch activate after copy: the copied venv's activate still has /opt/venv hardcoded.
    # For POSIX sh sourcing of the copied venv, the fallback must use TARGET_VENV's path.
    if [ -x "$PATCH_SCRIPT" ]; then
      "$PATCH_SCRIPT" "${TARGET_VENV}" >&2 || true
    fi
  fi
) 9>"${lockfile}"

# If --activate was specified, output commands to stdout for eval
if [[ "$ACTIVATE_MODE" == "true" ]]; then
    cat <<EOF
# Deactivate any currently active venv and activate the shared venv
deactivate 2>/dev/null || true
source ${TARGET_VENV}/bin/activate
EOF
fi
