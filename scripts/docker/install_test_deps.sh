#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <gtest_version> <doxygen_version>"
    exit 1
fi

GTEST_VERSION=$1
DOXYGEN_VERSION=$2

# Installs Google test
mkdir -p /opt/tt_metal_infra/googletest
chmod ugo+w /opt/tt_metal_infra/googletest
wget -O /opt/tt_metal_infra/googletest/googletest-release-${GTEST_VERSION}.tar.gz https://github.com/google/googletest/archive/refs/tags/v${GTEST_VERSION}.tar.gz
tar -xzf /opt/tt_metal_infra/googletest/googletest-release-${GTEST_VERSION}.tar.gz -C /opt/tt_metal_infra/googletest/
cd /opt/tt_metal_infra/googletest/googletest-${GTEST_VERSION}
cmake -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_SHARED_LIBS=ON .
make
make install

# Install doxygen
mkdir -p /opt/tt_metal_infra/doxygen
wget -O /opt/tt_metal_infra/doxygen/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz "https://www.doxygen.nl/files/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz"
tar -xzf /opt/tt_metal_infra/doxygen/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz -C /opt/tt_metal_infra/doxygen/
rm /opt/tt_metal_infra/doxygen/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz
cd /opt/tt_metal_infra/doxygen/doxygen-${DOXYGEN_VERSION}
make install
