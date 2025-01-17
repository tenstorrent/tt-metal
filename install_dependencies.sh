#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024 Tenstorrent, Inc. All rights reserved.
#
# This script is based on `xrtdeps.sh` from the Xilinx XRT project.
# Original source: https://github.com/Xilinx/XRT/blob/master/src/runtime_src/tools/scripts/xrtdeps.sh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FLAVOR=`grep '^ID=' /etc/os-release | awk -F= '{print $2}' | tr -d '"'`
VERSION=`grep '^VERSION_ID=' /etc/os-release | awk -F= '{print $2}' | tr -d '"'`
MAJOR=${VERSION%.*}
ARCH=`uname -m`

usage()
{
    echo "Usage: sudo ./install_dependencies.sh [options]"
    echo
    echo "[--help, -h]                List this help"
    echo "[--validate, -v]            Validate that required packages are installed"
    exit 1
}

validate=0

while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h)
            usage
            ;;
        --validate|-v)
            validate=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

ub_package_list()
{
    UB_LIST=(\
     git \
     build-essential \
     cmake \
     software-properties-common \
     libhwloc-dev \
     graphviz \
     ninja-build \
     libpython3-dev \
     libcapstone-dev \
     python3-pip \
     python3-dev \
     python3.8-venv \
     libc++-17-dev \
     libc++abi-17-dev \
    )

}

update_package_list()
{
    if [ $FLAVOR == "ubuntu" ]; then
        ub_package_list
    else
        echo "unknown OS flavor $FLAVOR"
        exit 1
    fi
}

validate_packages()
{
    if [ $FLAVOR == "ubuntu" ]; then
        dpkg -l "${UB_LIST[@]}"
        #dpkg -l "${UB_LIST[@]}" > /dev/null
    else
        echo "unknown OS flavor $FLAVOR"
        exit 1
    fi
}

prep_ubuntu()
{
    echo "Preparing ubuntu ..."
    # Update the list of available packages
    apt-get update
}

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

configure_hugepages() {
    TT_TOOLS_VERSION='1.1-5_all'
    echo "Installing Tenstorrent Hugepages Service $TT_TOOLS_VERSION..."
    TEMP_DIR=$(mktemp -d)
    wget -P $TEMP_DIR https://github.com/tenstorrent/tt-system-tools/releases/download/upstream%2F1.1/tenstorrent-tools_${TT_TOOLS_VERSION}.deb
    apt-get install $TEMP_DIR/tenstorrent-tools_${TT_TOOLS_VERSION}.deb
    systemctl enable --now tenstorrent-hugepages.service
    rm -rf "$TEMP_DIR"
}

install()
{
    if [ $FLAVOR == "ubuntu" ]; then
        prep_ubuntu

        echo "Installing packages..."
        DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends "${UB_LIST[@]}"
    fi
}

if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root. Please use sudo."
    usage
fi

update_package_list

if [ $validate == 1 ]; then
    validate_packages
else
    configure_hugepages
    install_llvm
    install
fi
