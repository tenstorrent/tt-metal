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

# --- Main Script ---

main() {
    # Set up trap for cleanup on exit
    trap cleanup EXIT

    # Get script directory and version file
    local script_dir
    script_dir="$(dirname "$0")"
    local version_file="$script_dir/sfpi-version.sh"

    # Check if version file exists and is readable
    if [[ ! -r "$version_file" ]]; then
        echo "ERROR: sfpi-version.sh not found or not readable at '$version_file'" >&2
        exit 1
    fi

    # Source the version file to load variables
    # shellcheck source=/dev/null
    source "$version_file"

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
    if [[ -f "sfpi/sfpi.version" ]]; then
        current_version="$(cat "sfpi/sfpi.version")"
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
    rm -rf sfpi
    # The -f flag is already included in the $tar_flags variable.
    # Passing it again was causing the error.
    if ! tar "$tar_flags" "$download_file"; then
        echo "ERROR: Failed to extract SFPI release from $download_file" >&2
        exit 1
    fi

    # Write the new version file
    echo "$sfpi_version" > sfpi/sfpi.version
    echo "SFPI successfully installed version $sfpi_version"
    echo "Setup complete. You can now run your tests."
}

# Execute the main function
main
