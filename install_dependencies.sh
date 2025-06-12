#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024 Tenstorrent, Inc. All rights reserved.

set -e

usage()
{
    echo "Usage: sudo ./install_dependencies.sh [options]"
    echo
    echo "[--help, -h]                List this help"
    echo "[--validate, -v]            Validate that required packages are installed"
    echo "[--docker, -d]              Specialize execution for docker"
    echo "[--no-distributed]          Don't install distributed compute dependencies (OpenMPI)"
    echo "[--mode, -m <mode>]         Select installation mode: runtime, build, baremetal"
    exit 1
}

FLAVOR=`grep '^ID=' /etc/os-release | awk -F= '{print $2}' | tr -d '"'`
VERSION=`grep '^VERSION_ID=' /etc/os-release | awk -F= '{print $2}' | tr -d '"'`
MAJOR=${VERSION%.*}
ARCH=`uname -m`

if [ $FLAVOR != "ubuntu" ]; then
    echo "Error: Only Ubuntu is supported"
    exit 1
fi

UBUNTU_CODENAME=$(grep '^VERSION_CODENAME=' /etc/os-release | awk -F= '{print $2}' | tr -d '"')
export UBUNTU_CODENAME

if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root. Please use sudo."
    usage
fi

validate=0
docker=0
distributed=1
mode="baremetal"

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
	--mode|-m)
            mode="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# libc++ runtime dependency could eventually go away
# It is favored on Ubuntu20.04 for C++20 support

# At the time of this writing the following libraries are linked at runtime by sfpi cross compiler
# libmpc, libmfpr, libgmp, libz
# For the time being it will be assumed that these packages come from the base Ubuntu image

# I would prefer to not be using -dev packages for runtime dependencies
# But I have not been able to verify any alternative package

# Packages needed at runtime and therefore needed by release docker image
ub_runtime_packages()
{
    UB_RUNTIME_LIST=(\
     python3-dev \
     python3-pip \
     python3-venv \
     libhwloc-dev \
     libnuma-dev \
     libatomic1 \
     libc++-17-dev \
     libc++abi-17-dev \
     libstdc++6 \
    )

    if [ "$distributed" -eq 1 ]; then
        UB_RUNTIME_LIST+=(openmpi-bin)
    fi
}

ub_buildtime_packages()
{
    UB_BUILDTIME_LIST=(\
     git \
     python3-dev \
     pkg-config \
     cargo \
     cmake \
     ninja-build \
     libboost-dev \
     libhwloc-dev \
     libc++-17-dev \
     libc++abi-17-dev \
     build-essential \
     xz-utils \
     pandoc \
     libtbb-dev \
     libcapstone-dev \
     pkg-config \
    )

    if [ "$distributed" -eq 1 ]; then
        UB_BUILDTIME_LIST+=(libopenmpi-dev)
    fi
}

# Packages needed to setup a baremetal machine to build from source and run

ub_baremetal_packages() {
    ub_runtime_packages
    ub_buildtime_packages
    UB_BAREMETAL_LIST=("${UB_RUNTIME_LIST[@]}" "${UB_BUILDTIME_LIST[@]}")
}

update_package_list()
{
    if [ $FLAVOR == "ubuntu" ]; then
	case "$mode" in
            runtime)
                ub_runtime_packages
                PKG_LIST=("${UB_RUNTIME_LIST[@]}")
                ;;
            build)
                ub_buildtime_packages
                PKG_LIST=("${UB_BUILDTIME_LIST[@]}")
                ;;
            baremetal)
                ub_baremetal_packages
                PKG_LIST=("${UB_BAREMETAL_LIST[@]}")
                ;;
            *)
                echo "Invalid mode: $mode"
                usage
                ;;
        esac
    fi
}

validate_packages()
{
    if [ $FLAVOR == "ubuntu" ]; then
        dpkg -l "${PKG_LIST[@]}"
    fi
}

prep_ubuntu_runtime()
{
    echo "Preparing ubuntu ..."
    # Update the list of available packages
    apt-get update
    apt-get install -y --no-install-recommends ca-certificates gpg lsb-release wget software-properties-common gnupg jq
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
    echo "deb http://apt.llvm.org/$UBUNTU_CODENAME/ llvm-toolchain-$UBUNTU_CODENAME-17 main" | tee /etc/apt/sources.list.d/llvm-17.list
    apt-get update
}

prep_ubuntu_build()
{
    echo "Preparing ubuntu ..."
    # Update the list of available packages
    apt-get update
    apt-get install -y --no-install-recommends ca-certificates gpg lsb-release wget software-properties-common gnupg jq
    # The below is to bring cmake from kitware
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null
    apt-get update
}

# We currently have an affinity to clang as it is more thoroughly tested in CI
# However g++-12 and later should also work

install_llvm() {
    LLVM_VERSION="17"
    echo "Checking if LLVM $LLVM_VERSION is already installed..."
    if command -v clang-$LLVM_VERSION &> /dev/null; then
        echo "LLVM $LLVM_VERSION is already installed. Skipping installation."
    else
        echo "Installing LLVM $LLVM_VERSION..."
        TEMP_DIR=$(mktemp -d)
        wget -P $TEMP_DIR https://apt.llvm.org/llvm.sh
        chmod u+x $TEMP_DIR/llvm.sh
        $TEMP_DIR/llvm.sh $LLVM_VERSION
        rm -rf "$TEMP_DIR"
    fi
}

install_gcc() {
    case "$VERSION" in
        "22.04")
            GCC_VER=12
            ;;
        "24.04")
            GCC_VER=14
            ;;
        *)
            echo "Unknown or unsupported Ubuntu version: $VERSION"
            echo "Falling back to installing default g++..."
            apt-get install -y --no-install-recommends g++
            echo "Using g++ version: $(g++ --version | head -n1)"
            return
            ;;
    esac

    echo "Detected Ubuntu $VERSION, installing g++-$GCC_VER..."

    apt-get install -y --no-install-recommends g++-$GCC_VER gcc-$GCC_VER

    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$GCC_VER $GCC_VER
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$GCC_VER $GCC_VER
    update-alternatives --set gcc /usr/bin/gcc-$GCC_VER
    update-alternatives --set g++ /usr/bin/g++-$GCC_VER

    echo "Using g++ version: $(g++ --version | head -n1)"
}

install_sfpi() {
    local version_file=$(dirname $0)/tt_metal/sfpi-version.sh
    if ! [[ -r $version_file ]] ; then
	version_file=$(dirname $0)/sfpi-version.sh
	if ! [[ -r $version_file ]] ; then
	    echo "sfpi-version.sh not found" >&2
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
	echo "Unknown packaging system" >&2
	exit 1
    fi
    local $(grep -v '^#' $version_file)
    local sfpi_arch_os=$(uname -m)_$(uname -s)
    local sfpi_pkg_md5=$(eval echo "\$sfpi_${sfpi_arch_os}_${pkg}_md5")
    if [ -z $(eval echo "$sfpi_${pkg}_md5") ] ; then
	echo "SFPI $pkg package for ${sfpi_arch_os} is not available" >&2
	exit 1
    fi
    local TEMP_DIR=$(mktemp -d)
    wget -P $TEMP_DIR "$sfpi_url/$sfpi_version/sfpi-${sfpi_arch_os}.${pkg}"
    if [ $(md5sum -b "${TEMP_DIR}/sfpi-${sfpi_arch_os}.${pkg}" | cut -d' ' -f1) \
	     != "$sfpi_pkg_md5" ] ; then
	echo "SFPI sfpi-${sfpi_arch_os}.${pkg} md5 mismatch" >&2
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
        echo "→ Skipping MPI ULFM installation (distributed mode not enabled)"
        return
    fi

    # Only install MPI ULFM for Ubuntu 24.04 or older
    local VERSION_NUM=$(echo "$VERSION" | sed 's/\.//')

    if [ "$VERSION_NUM" -gt "2404" ]; then
        echo "→ Skipping MPI ULFM installation for Ubuntu $VERSION (only needed for 24.04 or older)"
        return
    fi

    DEB_URL="https://github.com/tenstorrent/ompi/releases/download/v5.0.7/openmpi-ulfm_5.0.7-1_amd64.deb"
    DEB_FILE="$(basename "$DEB_URL")"

    # 1. Create temp workspace
    TMP_DIR="$(mktemp -d)"
    cleanup() { rm -rf "$TMP_DIR"; }
    trap cleanup EXIT INT TERM

    echo "→ Downloading $DEB_FILE …"
    wget -q --show-progress -O "$TMP_DIR/$DEB_FILE" "$DEB_URL"

    # 2. Install
    echo "→ Installing $DEB_FILE …"
    apt-get update -qq
    apt-get install -f -y "$TMP_DIR/$DEB_FILE"
}

# We don't really want to have hugepages dependency
# This could be removed in the future

configure_hugepages() {
    # Fetch the lastest tt-tools release link and name of package
    TT_TOOLS_LINK=$(wget -qO- https://api.github.com/repos/tenstorrent/tt-system-tools/releases/latest | jq -r '.assets[] | select(.name | endswith(".deb")) | .browser_download_url')
    TT_TOOLS_NAME=$(wget -qO- https://api.github.com/repos/tenstorrent/tt-system-tools/releases/latest | jq -r '.assets[] | select(.name | endswith(".deb")) | .name')

    echo "Installing Tenstorrent Hugepages Service $TT_TOOLS_NAME..."
    TEMP_DIR=$(mktemp -d)
    wget -P $TEMP_DIR $TT_TOOLS_LINK
    apt-get install -y --no-install-recommends $TEMP_DIR/$TT_TOOLS_NAME
    systemctl enable --now tenstorrent-hugepages.service
    rm -rf "$TEMP_DIR"
}

install() {
    if [ $FLAVOR == "ubuntu" ]; then
        echo "Installing packages..."
	case "$mode" in
            runtime)
                prep_ubuntu_runtime
                install_sfpi
                install_mpi_ulfm
                ;;
            build)
                prep_ubuntu_build
                install_llvm
                install_gcc
                install_mpi_ulfm
                ;;
            baremetal)
                prep_ubuntu_runtime
                install_sfpi
                prep_ubuntu_build
                install_llvm
                install_gcc
                configure_hugepages
                install_mpi_ulfm
                ;;
        esac

	DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends "${PKG_LIST[@]}"

    fi
}

cleanup() {
    if [ $FLAVOR == "ubuntu" ]; then
        rm -rf /var/lib/apt/lists/*
    fi
}

update_package_list

if [ "$validate" -eq 1 ]; then
    validate_packages
else
    install
fi

if [ "$docker" -eq 1 ]; then
    cleanup
fi
