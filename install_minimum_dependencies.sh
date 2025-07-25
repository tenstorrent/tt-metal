#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

set -e

usage() {
    echo "Usage: sudo ./install_sfpi.sh [options]"
    echo
    echo "[--help, -h]                List this help"
    echo
    echo "This script installs only the SFPI package for TT-Metal development."
    exit 1
}

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="$ID"
        OS_VERSION="$VERSION_ID"
        OS_CODENAME="${UBUNTU_CODENAME:VERSION_CODENAME}"
        OS_ID_LIKE="$ID_LIKE"
    else
        echo "Error: /etc/os-release not found. Unsupported system."
        exit 1
    fi

    echo "Detected OS: $OS_ID $OS_VERSION"
}

check_packaging_system() {
    if dpkg-query -f '${Version}' -W libc-bin >/dev/null 2>&1; then
        PKG_SYSTEM="deb"
    elif rpm -q --qf '%{VERSION}' glibc >/dev/null 2>&1; then
        PKG_SYSTEM="rpm"
    else
        echo "[ERROR] Unknown packaging system. SFPI installation requires either dpkg or rpm."
        exit 1
    fi
    echo "[INFO] Detected packaging system: $PKG_SYSTEM"
}

find_sfpi_version_file() {
    local version_file=$(dirname $0)/tt_metal/sfpi-version.sh
    if ! [[ -r $version_file ]]; then
        version_file=$(dirname $0)/sfpi-version.sh
        if ! [[ -r $version_file ]]; then
            echo "[ERROR] sfpi-version.sh not found. Please ensure this script is run from the tt-metal repository root or that sfpi-version.sh is in the same directory."
            exit 1
        fi
    fi
    echo "$version_file"
}

install_sfpi() {
    echo "[INFO] Installing SFPI package..."

    local version_file=$(find_sfpi_version_file)

    # Source the version file to get SFPI variables
    local $(grep -v '^#' $version_file)

    local sfpi_arch_os=$(uname -m)_$(uname -s)
    local sfpi_pkg_md5=$(eval echo "\$sfpi_${sfpi_arch_os}_${PKG_SYSTEM}_md5")

    # Check if SFPI package is available for this architecture/OS/packaging combination
    if [ -z $(eval echo "$sfpi_${PKG_SYSTEM}_md5") ]; then
        echo "[ERROR] SFPI $PKG_SYSTEM package for ${sfpi_arch_os} is not available"
        exit 1
    fi

    # Create temporary directory for download
    local TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT INT TERM

    echo "[INFO] Downloading SFPI package for ${sfpi_arch_os}..."
    wget -P $TEMP_DIR "$sfpi_url/$sfpi_version/sfpi-${sfpi_arch_os}.${PKG_SYSTEM}"

    # Verify MD5 checksum
    echo "[INFO] Verifying package integrity..."
    if [ $(md5sum -b "${TEMP_DIR}/sfpi-${sfpi_arch_os}.${PKG_SYSTEM}" | cut -d' ' -f1) != "$sfpi_pkg_md5" ]; then
        echo "[ERROR] SFPI sfpi-${sfpi_arch_os}.${PKG_SYSTEM} md5 mismatch"
        exit 1
    fi

    # Install the package based on packaging system
    echo "[INFO] Installing SFPI package..."
    case "$PKG_SYSTEM" in
        deb)
            apt-get install -y --allow-downgrades $TEMP_DIR/sfpi-${sfpi_arch_os}.deb
            ;;
        rpm)
            rpm --upgrade --force $TEMP_DIR/sfpi-${sfpi_arch_os}.rpm
            ;;
    esac

    echo "[INFO] SFPI package installed successfully!"
}

# Main execution

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root. Please use sudo."
    usage
fi

# Parse command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Perform necessary checks
detect_os
check_packaging_system

# Install SFPI
install_sfpi

echo "[INFO] SFPI installation completed successfully!"
