#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <doxygen_version>"
    exit 1
fi

DOXYGEN_VERSION=$1

# Install doxygen
mkdir -p /opt/tt_metal_infra/doxygen
wget -O /opt/tt_metal_infra/doxygen/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz "https://www.doxygen.nl/files/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz"
tar -xzf /opt/tt_metal_infra/doxygen/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz -C /opt/tt_metal_infra/doxygen/
rm /opt/tt_metal_infra/doxygen/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz
cd /opt/tt_metal_infra/doxygen/doxygen-${DOXYGEN_VERSION}
make install
