#!/bin/bash

set -x

echo $@

# Launch the container with appropriate device mapping
docker run --rm --device /dev/tenstorrent \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    --network=host \
    -e TT_METAL_HOME=$(pwd) \
    -e PYTHONPATH=$(pwd) \
    -e LD_LIBRARY_PATH=$(pwd)/build/lib \
    -e ARCH_NAME=wormhole_b0 \
    -e LOGURU_LEVEL=INFO \
    ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64 \
    orted "$@"
