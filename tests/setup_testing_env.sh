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

    local base_url="https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/tt_metal/hw/inc/${chip_arch}"
    local headers=("cfg_defines.h" "dev_mem_map.h" "tensix.h" "tensix_dev_map.h" "tensix_types.h")

    local specific_url=""
    if [[ "$chip_arch" == "wormhole" ]]; then
        specific_url="https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/tt_metal/hw/inc/${chip_arch}/wormhole_b0_defines"
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
    local version_file="$SCRIPT_DIR/sfpi-version.sh"

    # Check if version file exists and is readable
    if [[ ! -r "$version_file" ]]; then
        echo "ERROR: sfpi-version.sh not found or not readable at '$version_file'" >&2
        exit 1
    fi

    # Source the version file to load variables
    # shellcheck source=/dev/null
    source "$version_file"

    # Get chip architecture
    local chip_arch
    chip_arch=$(get_chip_architecture)

    # Download headers
    download_headers "$chip_arch"

    # Setup pre-commit hooks
    setup_precommit

    # Determine architecture, OS, file extension, and extraction flags
    local arch_os
    arch_os="$(uname -m)_$(uname -s)"
    local file_ext tar_flags
    case "$(uname -m)" in
        x86_64)
            file_ext="txz"
            tar_flags="-xJf"
            ;;
        aarch64)
            file_ext="tar"
            tar_flags="-xf"
            ;;
        *)
            echo "ERROR: Unsupported architecture: $(uname -m)" >&2
            exit 1
            ;;
    esac

    # Build the variable name for the MD5 hash and get its value securely
    local md5_var="sfpi_${arch_os}_${file_ext}_md5"
    if [[ -z "${!md5_var:-}" ]]; then
        echo "ERROR: SFPI package for ${arch_os} is not available. MD5 variable '$md5_var' is not set." >&2
        exit 1
    fi
    local expected_md5="${!md5_var}"

    # Check if SFPI is already installed and up to date
    local current_version=""
    if [[ -f "${SCRIPT_DIR}/sfpi/sfpi.version" ]]; then
        current_version="$(cat "${SCRIPT_DIR}/sfpi/sfpi.version")"
    fi

    if [[ "$current_version" == "$sfpi_version" ]]; then
        echo "SFPI is already at the correct version: $sfpi_version"
        exit 0
    fi

    # Download and install SFPI
    echo "SFPI not present or out of date. Fetching version ${sfpi_version}..."

    # Create a temporary directory for the download
    TEMP_DIR="$(mktemp -d)"
    local download_file="$TEMP_DIR/sfpi-${arch_os}.${file_ext}"
    local download_url="$sfpi_url/$sfpi_version/sfpi-${arch_os}.${file_ext}"

    # Download the file
    if ! wget -O "$download_file" --waitretry=5 --retry-connrefused "$download_url"; then
        echo "ERROR: Failed to download $download_url" >&2
        exit 1
    fi

    # Verify the MD5 checksum
    local actual_md5
    actual_md5="$(md5sum -b "$download_file" | cut -d' ' -f1)"
    if [[ "$actual_md5" != "$expected_md5" ]]; then
        echo "ERROR: MD5 checksum mismatch for sfpi-${arch_os}.${file_ext}" >&2
        echo "  Expected: $expected_md5" >&2
        echo "  Actual:   $actual_md5" >&2
        exit 1
    fi

    # Remove old installation and extract the new one
    echo "Extracting SFPI release..."
    rm -rf "${SCRIPT_DIR}/sfpi"
    # The -f flag is already included in the $tar_flags variable.
    # Passing it again was causing the error.
    if ! tar "${tar_flags}" "$download_file" -C "${SCRIPT_DIR}"; then
        echo "ERROR: Failed to extract SFPI release from $download_file" >&2
        exit 1
    fi

    # Write the new version file
    echo "$sfpi_version" > "${SCRIPT_DIR}/sfpi/sfpi.version"
    echo "SFPI successfully installed version $sfpi_version"
    echo "Setup complete. You can now run your tests."
    echo "Pre-commit hooks have been configured for code quality checks."
}

# Execute the main function
main
