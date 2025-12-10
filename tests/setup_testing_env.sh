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

# Function to get chip architecture using tt-smi
get_chip_architecture() {
    local smi_output
    if ! smi_output=$(tt-smi -ls 2>/dev/null); then
        echo "ERROR: tt-smi command failed or not found. Please ensure tt-smi is installed and in your PATH." >&2
        exit 1
    fi

    if echo "$smi_output" | grep -q "Blackhole"; then
        echo "blackhole"
    elif echo "$smi_output" | grep -q "Wormhole"; then
        echo "wormhole"
    else
        echo "ERROR: Unable to determine chip architecture from tt-smi output." >&2
        echo "tt-smi output:" >&2
        echo "$smi_output" >&2
        exit 1
    fi
}

# Function to download headers
download_headers() {
    local chip_arch=$1
    local header_dir="${SCRIPT_DIR}/hw_specific/${chip_arch}/inc"
    local stamp_file="${header_dir}/.headers_downloaded"

    if [[ -f "$stamp_file" ]]; then
        echo "Headers for ${chip_arch} already downloaded."
        return
    fi

    echo "Downloading headers for ${chip_arch}..."
    mkdir -p "$header_dir"

    local base_url="https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/tt_metal/hw/inc/tt-1xx/${chip_arch}"
    local headers=("cfg_defines.h" "dev_mem_map.h" "tensix.h" "tensix_types.h")

    local specific_url=""
    if [[ "$chip_arch" == "wormhole" ]]; then
        specific_url="https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/tt_metal/hw/inc/tt-1xx/${chip_arch}/wormhole_b0_defines"
    fi

    for header in "${headers[@]}"; do
        local download_url="${base_url}/${header}"
        if ! wget -O "${header_dir}/${header}" --waitretry=5 --retry-connrefused "$download_url" > /dev/null; then
            if [[ -n "$specific_url" ]]; then
                local fallback_url="${specific_url}/${header}"
                echo "Could not find ${header} at ${download_url}, trying ${fallback_url}..."
                if ! wget -O "${header_dir}/${header}" --waitretry=5 --retry-connrefused "$fallback_url" > /dev/null; then
                    echo "ERROR: Failed to download ${header} from both primary and fallback URLs." >&2
                    exit 1
                fi
            else
                echo "ERROR: Failed to download ${header} from ${download_url}" >&2
                exit 1
            fi
        fi
    done

    touch "$stamp_file"
    echo "Headers for ${chip_arch} downloaded successfully."
}

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

    # Get chip architecture
    local chip_arch
    if [[ -n "${CHIP_ARCH:-}" ]]; then
        chip_arch="$CHIP_ARCH"
        echo "Using CHIP_ARCH environment variable: $chip_arch"
    else
        chip_arch=$(get_chip_architecture)
    fi

    # Download headers
    if [[ "$chip_arch" != "quasar" ]]; then
        download_headers "$chip_arch"
    else
        echo "No external headers needed for quasar architecture."
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
    if ! wget -P $TEMP_DIR --waitretry=5 --retry-connrefused "$sfpi_url/$sfpi_filename" ; then
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
