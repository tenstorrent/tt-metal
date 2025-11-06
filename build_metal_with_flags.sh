#!/bin/bash
cd $TT_METAL_HOME
./build_metal_custom.sh \
    -e -g make \
    --install-prefix=$(pwd)/build/install \
    --build-all --release
