#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

set -e

usage()
{
    echo "Usage: sudo ./install_dependencies.sh [options]"
    echo
    echo "[--help, -h]                List this help"
    echo "[--validate, -v]            Validate that required packages are installed"
    echo "[--docker, -d]              Specialize execution for docker"
    echo "[--no-distributed]          Don't install distributed compute dependencies (OpenMPI)"
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

    # Detect package manager
    for mgr in apt dnf yum; do
        if command -v "$mgr" >/dev/null 2>&1; then
            PKG_MANAGER="$mgr"
            break
        fi
    done
    if [ -z "$PKG_MANAGER" ]; then
        echo "Error: No supported package manager found"
        exit 1
    fi

    echo "Detected OS: $OS_ID $OS_VERSION ($PKG_MANAGER)"

    get_package_family
}

get_package_family() {
    case "$OS_ID" in
        ubuntu|debian)
            PKG_FAMILY="debian"
            ;;
        fedora|centos|rhel|rocky|almalinux)
            PKG_FAMILY="redhat"
            ;;
        *)
            # For distributions that are similar to supported ones
            if [[ "$OS_ID_LIKE" == *"debian"* ]] || [[ "$OS_ID_LIKE" == *"ubuntu"* ]]; then
                PKG_FAMILY="debian"
            elif [[ "$OS_ID_LIKE" == *"rhel"* ]] || [[ "$OS_ID_LIKE" == *"fedora"* ]]; then
                PKG_FAMILY="redhat"
            else
                PKG_FAMILY="unknown"
            fi
            ;;
    esac
}

is_debian_based() {
    [[ "$PKG_FAMILY" == "debian" ]]
}

is_redhat_based() {
    [[ "$PKG_FAMILY" == "redhat" ]]
}

is_supported_os() {
    case "$OS_ID" in
        ubuntu|debian)
            return 0
            ;;
        fedora|centos|rhel|rocky|almalinux)
            return 0
            ;;
        *)
            if [[ "$OS_ID_LIKE" == *"debian"* ]] || [[ "$OS_ID_LIKE" == *"ubuntu"* ]]; then
                return 0
            elif [[ "$OS_ID_LIKE" == *"rhel"* ]] || [[ "$OS_ID_LIKE" == *"fedora"* ]]; then
                return 0
            else
                return 1
            fi
            ;;
    esac
}

update_package_cache() {
    echo "[INFO] Updating package cache..."
    case "$PKG_MANAGER" in
        apt)
            apt-get update
            ;;
        dnf)
            dnf makecache
            ;;
        yum)
            yum makecache
            ;;
    esac
}

install_packages() {
    echo "[INFO] Installing packages: ${PACKAGES[*]}"
    case "$PKG_MANAGER" in
        apt)
            DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends "${PACKAGES[@]}"
            ;;
        dnf)
            dnf install -y "${PACKAGES[@]}"
            ;;
        yum)
            yum install -y "${PACKAGES[@]}"
            ;;
    esac
}

validate_packages() {
    echo "[INFO] Validating packages: ${PACKAGES[*]}"
    case "$PKG_MANAGER" in
        apt)
            dpkg -l "${PACKAGES[@]}"
            ;;
        dnf|yum)
            rpm -q "${PACKAGES[@]}"
            ;;
    esac
}

cleanup_package_cache() {
    echo "[INFO] Cleaning up package cache..."
    case "$PKG_MANAGER" in
        apt)
            apt-get clean && rm -rf /var/lib/apt/lists/*
            ;;
        dnf)
            dnf clean all
            ;;
        yum)
            yum clean all
            ;;
    esac
}

# Initialize packages
init_packages() {
    # Check if package family is supported
    if [[ "$PKG_FAMILY" == "unknown" ]]; then
        echo "[ERROR] No package list available for $OS_ID (ID_LIKE: $OS_ID_LIKE) (PKG_FAMILY: $PKG_FAMILY)"
        exit 1
    fi

    # Set packages based on determined family
    case "$PKG_FAMILY" in
        debian)
            # Determine g++ version based on Ubuntu version
            local gpp_package="g++"
            if [[ "$OS_ID" == "ubuntu" ]]; then
                case "$OS_VERSION" in
                    "22.04")
                        gpp_package="g++-12"
                        echo "[INFO] Using g++-12 for Ubuntu 22.04 (gcc-12 will be installed as dependency)"
                        ;;
                    "24.04")
                        gpp_package="g++-14"
                        echo "[INFO] Using g++-14 for Ubuntu 24.04 (gcc-14 will be installed as dependency)"
                        ;;
                    *)
                        echo "[INFO] Using default g++ for Ubuntu $OS_VERSION"
                        ;;
                esac
            fi

            # All packages needed for TT-Metal development
            PACKAGES=(
                "git"
                "build-essential"
                "cmake"
                "ninja-build"
                "pkg-config"
                "cargo"
                "$gpp_package"
                "pandoc"
                "xz-utils"
                "python3-dev"
                "python3-pip"
                "python3-venv"
                "libhwloc-dev"
                "libnuma-dev"
                "libatomic1"
                "libstdc++6"
                "libboost-dev"
                "libtbb-dev"
                "libcapstone-dev"
                "libc++-17-dev"
                "libc++abi-17-dev"
                "wget"
            )
            if [ "$distributed" -eq 1 ]; then
                PACKAGES+=("openmpi-bin" "libopenmpi-dev")
            fi
            ;;
        redhat)
            PACKAGES=(
                "git"
                "gcc"
                "gcc-c++"
                "make"
                "clang"
                "cmake"
                "ninja-build"
                "pkgconf-pkg-config"
                "cargo"
                "xz"
                "python3-devel"
                "python3-pip"
                "hwloc-devel"
                "numactl-devel"
                "libatomic"
                "libstdc++"
                "boost-devel"
                "tbb-devel"
                "capstone-devel"
                "wget"
            )
            if [ "$distributed" -eq 1 ]; then
                PACKAGES+=("openmpi" "openmpi-devel")
            fi
            ;;
    esac
}

prep_system() {
    echo "[INFO] Preparing system for TT-Metal development ($OS_ID)..."

    case "$PKG_FAMILY" in
        debian)
            prep_ubuntu_system
            ;;
        redhat)
            prep_redhat_system
            ;;
        *)
            echo "[WARNING] No specific system preparation for $OS_ID"
            ;;
    esac
}

prep_ubuntu_system() {
    echo "[INFO] Preparing Ubuntu/Debian system..."
    # Update package lists and install basic tools
    apt-get update
    apt-get install -y --no-install-recommends ca-certificates gpg lsb-release wget software-properties-common gnupg jq

    # Add LLVM repository for Clang 17
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
    echo "deb http://apt.llvm.org/$OS_CODENAME/ llvm-toolchain-$OS_CODENAME-17 main" | tee /etc/apt/sources.list.d/llvm-17.list

    # Add Kitware repository for latest CMake
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $OS_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null

    # Add GCC toolchain repository for specific g++ versions if needed
    if [[ "$OS_ID" == "ubuntu" ]]; then
        case "$OS_VERSION" in
            "24.04")
                echo "[INFO] Adding toolchain repository for g++-14 on Ubuntu 24.04"
                add-apt-repository -y ppa:ubuntu-toolchain-r/test
                ;;
        esac
    fi

    apt-get update
}

prep_redhat_system() {
    echo "[INFO] Preparing Red Hat family system..."
    # TODO: Implement Red Hat family system preparation
}

# We currently have an affinity to clang as it is more thoroughly tested in CI
# However g++-12 and later should also work

install_llvm() {
    # Only install LLVM on debian-based systems
    if ! is_debian_based; then
        echo "[WARNING] Skipping LLVM installation for non-debian distribution ($OS_ID)"
        return
    fi

    LLVM_VERSION="17"
    echo "[INFO] Checking if LLVM $LLVM_VERSION is already installed..."
    if command -v clang-$LLVM_VERSION &> /dev/null; then
        echo "[INFO] LLVM $LLVM_VERSION is already installed. Skipping installation."
    else
        echo "[INFO] Installing LLVM $LLVM_VERSION..."
        TEMP_DIR=$(mktemp -d)
        wget -P $TEMP_DIR https://apt.llvm.org/llvm.sh
        chmod u+x $TEMP_DIR/llvm.sh
        $TEMP_DIR/llvm.sh $LLVM_VERSION
        rm -rf "$TEMP_DIR"
    fi
}

install_sfpi() {
    local version_file=$(dirname $0)/tt_metal/sfpi-version.sh
    if ! [[ -r $version_file ]] ; then
	version_file=$(dirname $0)/sfpi-version.sh
	if ! [[ -r $version_file ]] ; then
	    echo "[ERROR] sfpi-version.sh not found" >&2
	    exit 1
	fi
    fi
    # determine packaging system
    local pkg
    if dpkg-query -f '${Version}' -W libc-bin >/dev/null 2>&1 ; then
	pkg=deb
    elif rpm -q --qf '%{VERSION}' glibc >/dev/null 2>&1 ; then
	pkg=rpm
    else
	echo "[ERROR] Unknown packaging system" >&2
	exit 1
    fi
    local $(grep -v '^#' $version_file)
    local sfpi_arch_os=$(uname -m)_$(uname -s)
    local sfpi_pkg_md5=$(eval echo "\$sfpi_${sfpi_arch_os}_${pkg}_md5")
    if [ -z $(eval echo "$sfpi_${pkg}_md5") ] ; then
	echo "[ERROR] SFPI $pkg package for ${sfpi_arch_os} is not available" >&2
	exit 1
    fi
    local TEMP_DIR=$(mktemp -d)
    wget -P $TEMP_DIR "$sfpi_url/$sfpi_version/sfpi-${sfpi_arch_os}.${pkg}"
    if [ $(md5sum -b "${TEMP_DIR}/sfpi-${sfpi_arch_os}.${pkg}" | cut -d' ' -f1) \
	     != "$sfpi_pkg_md5" ] ; then
	echo "[ERROR] SFPI sfpi-${sfpi_arch_os}.${pkg} md5 mismatch" >&2
	rm -rf $TEMP_DIR
	exit 1
    fi
    # we must select exactly this version
    case "$pkg" in
	deb)
	    apt-get install -y --allow-downgrades $TEMP_DIR/sfpi-${sfpi_arch_os}.deb
	    ;;
	rpm)
	    rpm --upgrade --force $TEMP_DIR/sfpi-${sfpi_arch_os}.rpm
	    ;;
    esac
    rm -rf $TEMP_DIR
}

install_mpi_ulfm(){
    # Only install if distributed flag is set
    if [ "$distributed" -ne 1 ]; then
        echo "[INFO] Skipping MPI ULFM installation (distributed mode not enabled)"
        return
    fi

    # Check if OS is Ubuntu/Debian-based
    if ! is_debian_based; then
        echo "[WARNING] MPI ULFM installation is currently only supported on Ubuntu/Debian-based distributions"
        echo "[WARNING] This function needs to be expanded to support $OS_ID"
        return
    fi

    # Only install MPI ULFM for Ubuntu 24.04 or older
    local VERSION_NUM=$(echo "$VERSION" | sed 's/\.//')

    if [ "$VERSION_NUM" -gt "2404" ]; then
        echo "[INFO] Skipping MPI ULFM installation for Ubuntu $VERSION (only needed for 24.04 or older)"
        return
    fi

    DEB_URL="https://github.com/tenstorrent/ompi/releases/download/v5.0.7/openmpi-ulfm_5.0.7-1_amd64.deb"
    DEB_FILE="$(basename "$DEB_URL")"

    # 1. Create temp workspace
    TMP_DIR="$(mktemp -d)"
    cleanup_mpi_temp() { rm -rf "$TMP_DIR"; }
    trap cleanup_mpi_temp EXIT INT TERM

    echo "[INFO] Downloading $DEB_FILE …"
    wget -q --show-progress -O "$TMP_DIR/$DEB_FILE" "$DEB_URL"

    # 2. Install
    echo "[INFO] Installing $DEB_FILE …"
    apt-get update -qq
    apt-get install -f -y "$TMP_DIR/$DEB_FILE"
}

# We don't really want to have hugepages dependency
# This could be removed in the future

configure_hugepages() {
    # Check if OS is Ubuntu/Debian-based
    if ! is_debian_based; then
        echo "[WARNING] Hugepages configuration is currently only supported on Ubuntu/Debian-based distributions"
        echo "[WARNING] This function needs to be expanded to support $OS_ID"
        return
    fi

    # Fetch the lastest tt-tools release link and name of package
    TT_TOOLS_LINK=$(wget -qO- https://api.github.com/repos/tenstorrent/tt-system-tools/releases/latest | jq -r '.assets[] | select(.name | endswith(".deb")) | .browser_download_url')
    TT_TOOLS_NAME=$(wget -qO- https://api.github.com/repos/tenstorrent/tt-system-tools/releases/latest | jq -r '.assets[] | select(.name | endswith(".deb")) | .name')

    echo "[INFO] Installing Tenstorrent Hugepages Service $TT_TOOLS_NAME..."
    TEMP_DIR=$(mktemp -d)
    wget -P $TEMP_DIR $TT_TOOLS_LINK
    apt-get install -y --no-install-recommends $TEMP_DIR/$TT_TOOLS_NAME
    systemctl enable --now tenstorrent-hugepages.service
    rm -rf "$TEMP_DIR"
}

install() {
    echo "[INFO] Installing TT-Metalium dependencies for $OS_ID ($PKG_MANAGER)..."

    # Update package cache first
    update_package_cache

    # Prepare system (repositories, keys, etc.)
    prep_system

    # Install core packages
    install_packages

    # Install specialized components
    install_sfpi
    install_llvm
    install_mpi_ulfm

    # Configure system (hugepages, etc.) - only for baremetal (not docker)
    if [ "$docker" -ne 1 ]; then
        configure_hugepages
    fi
}

cleanup() {
    if [ "$docker" -eq 1 ]; then
        cleanup_package_cache
    fi
}

# Alright, lets run some things!

if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root. Please use sudo."
    usage
fi

VERSION=`grep '^VERSION_ID=' /etc/os-release | awk -F= '{print $2}' | tr -d '"'`

# Initialize OS detection and validation
detect_os

if ! is_supported_os; then
    echo "Error: $OS_ID is not currently supported."
    echo "Supported distributions: Ubuntu, Debian, Fedora, CentOS, RHEL, Rocky Linux, AlmaLinux"
    exit 1
fi

validate=0
docker=0
distributed=1

while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h)
            usage
            ;;
        --validate|-v)
            validate=1
            shift
            ;;
        --docker|-d)
            docker=1
            shift
            ;;
        --no-distributed)
            distributed=0
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

init_packages

if [ "$validate" -eq 1 ]; then
    validate_packages
else
    install
fi

cleanup

echo "[INFO] TT-Metalium dependencies installed successfully!"
