#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# --- Functions ---

# Function to clean up the temporary directory on exit
cleanup() {
    if [[ -n "${TEMP_DIR:-}" && -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
}

# --- Globals ---
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Function to setup pre-commit hooks
setup_precommit() {
    echo "Setting up pre-commit hooks..."

    # Navigate to repository root (parent of tests directory)
    local repo_root="${SCRIPT_DIR}/.."

    # Check if we're in a git repository
    if ! git -C "$repo_root" rev-parse --git-dir >/dev/null 2>&1; then
        echo "WARNING: Not in a git repository, skipping pre-commit setup"
        return
    fi

    # Install pre-commit if not available
    if ! command -v pre-commit >/dev/null 2>&1; then
        echo "Installing pre-commit..."
        pip install pre-commit
    fi

    # Install hooks
    echo "Installing pre-commit hooks..."
    if (cd "$repo_root" && pre-commit install); then
        echo "Pre-commit hooks installed successfully!"
    else
        echo "WARNING: Failed to install pre-commit hooks"
    fi
}

# --- Main Script ---

main() {
    # Set up trap for cleanup on exit
    trap cleanup EXIT

    # Get script directory and version file
    local version_file="$SCRIPT_DIR/sfpi-info.sh"

    # Check if version file exists and is readable
    if [[ ! -r "$version_file" ]]; then
        echo "ERROR: sfpi-info.sh not found or not readable at '$version_file'" >&2
        exit 1
    fi

    # Setup pre-commit hooks
    setup_precommit

    # shellcheck source=/dev/null
    eval local $($version_file SHELL txz)

    # Check if SFPI is already installed and up to date
    if [[ -f "${SCRIPT_DIR}/sfpi/sfpi.version" ]] &&
       [[ "$sfpi_version" == "$(cat "${SCRIPT_DIR}/sfpi/sfpi.version")" ]] ; then
        echo "SFPI is already at the correct version: $sfpi_version"
        exit 0
    fi

    if [[ -z $sfpi_hash ]] ; then
	echo "[ERROR] SFPI $sfpi_version $sfpi_pkg package for $sfpi_arch $sfpi_dist is not available" >&2
	exit 1
    fi

    # Download SFPI
    echo "SFPI not present or out of date. Fetching version ${sfpi_version}..."
    local TEMP_DIR=$(mktemp -d)
    if ! wget -q -P $TEMP_DIR --waitretry=5 --retry-connrefused "$sfpi_url/$sfpi_filename" ; then
        echo "ERROR: Failed to download $sfpi_url/$sfpi_filename" >&2
        exit 1
    fi
    if [ $(${sfpi_hashtype}sum -b "${TEMP_DIR}/$sfpi_filename" | cut -d' ' -f1) \
	     != "$sfpi_hash" ] ; then
	echo "[ERROR] SFPI $sfpi_filename ${sfpi_hashtype} mismatch" >&2
	rm -rf $TEMP_DIR
	exit 1
    fi

    # Remove old installation and extract the new one
    echo "Extracting SFPI release..."
    rm -rf "${SCRIPT_DIR}/sfpi"
    if ! tar xJf "$TEMP_DIR/$sfpi_filename" -C "${SCRIPT_DIR}"; then
        echo "ERROR: Failed to extract SFPI release from $sfpi_filename" >&2
	rm -rf $TEMP_DIR
        exit 1
    fi
    rm -rf $TEMP_DIR

    # Write the new version file
    echo "$sfpi_version" > "${SCRIPT_DIR}/sfpi/sfpi.version"
    echo "SFPI successfully installed version $sfpi_version"
    echo "Setup complete. You can now run your tests."
    echo "Pre-commit hooks have been configured for code quality checks."
}

# Execute the main function
main
