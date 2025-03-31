#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024 Tenstorrent, Inc. All rights reserved.

set -ex

usage()
{
    echo "Usage: sudo ./install_dependencies.sh [options]"
    echo
    echo "[--help, -h]                List this help"
    echo "[--validate, -v]            Validate that required packages are installed"
    echo "[--docker, -d]              Specialize execution for docker"
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

ub_runtime_packages()
{
    UB_RUNTIME_LIST=(\
     python3-pip \
     python3-venv \
     libhwloc-dev \
     libnuma-dev \
     libc++-17-dev \
     libc++abi-17-dev \
    )
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
    )
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
    apt-get install -y --no-install-recommends ca-certificates gpg lsb-release wget software-properties-common gnupg
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
    echo "deb http://apt.llvm.org/$UBUNTU_CODENAME/ llvm-toolchain-$UBUNTU_CODENAME-17 main" | tee /etc/apt/sources.list.d/llvm-17.list
    apt-get update
}

prep_ubuntu_build()
{
    echo "Preparing ubuntu ..."
    # Update the list of available packages
    apt-get update
    apt-get install -y --no-install-recommends ca-certificates gpg lsb-release wget software-properties-common gnupg
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

# Install g++-12 if on Ubuntu 22.04
install_gcc12() {
    if [ $VERSION == "22.04" ]; then
        echo "Detected Ubuntu 22.04, installing g++-12..."
        apt-get install -y --no-install-recommends g++-12 gcc-12
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
        update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12
        update-alternatives --set gcc /usr/bin/gcc-12
        update-alternatives --set g++ /usr/bin/g++-12
    fi
}

# We don't really want to have hugepages dependency
# This could be removed in the future

configure_hugepages() {
    TT_TOOLS_VERSION='1.1-5_all'
    echo "Installing Tenstorrent Hugepages Service $TT_TOOLS_VERSION..."
    TEMP_DIR=$(mktemp -d)
    wget -P $TEMP_DIR https://github.com/tenstorrent/tt-system-tools/releases/download/upstream%2F1.1/tenstorrent-tools_${TT_TOOLS_VERSION}.deb
    apt-get install -y --no-install-recommends $TEMP_DIR/tenstorrent-tools_${TT_TOOLS_VERSION}.deb
    systemctl enable --now tenstorrent-hugepages.service
    rm -rf "$TEMP_DIR"
}

install() {
    if [ $FLAVOR == "ubuntu" ]; then
        echo "Installing packages..."

	case "$mode" in
            runtime)
                prep_ubuntu_runtime
                ;;
            build)
                prep_ubuntu_build
                install_llvm
		install_gcc12
                ;;
            baremetal)
                prep_ubuntu_build
                install_llvm
		install_gcc12
                configure_hugepages
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
