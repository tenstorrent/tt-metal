#!/bin/bash
cd $TT_METAL_HOME
./build_metal_custom.sh \
    -e -p \
    --install-prefix=/opt/tt-metal/ \
    --build-all --release
