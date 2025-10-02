#!/bin/bash
cd $TT_METAL_HOME
./build_metal_custom.sh \
    -e -p \
    --install-prefix=/usr/local/ \
    --build-dir=build-cmake \
    --build-all --release 
