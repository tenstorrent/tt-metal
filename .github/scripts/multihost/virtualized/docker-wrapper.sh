#!/bin/bash

# Inspired by https://qnib.org/2016/03/31/dssh/index.html

set -x

# Get container name from first argument
container_num=$1
shift

# Determine device mapping based on rankfile and container name
num_nodes=$(wc -l < rankfile)
device_args=""

case $num_nodes in
    1)
        device_args="--device /dev/tenstorrent"
        ;;
    2)
        case $container_num in
            0)
                device_args="--device /dev/tenstorrent/0 --device /dev/tenstorrent/1"
                ;;
            1)
                device_args="--device /dev/tenstorrent/2 --device /dev/tenstorrent/3"
                ;;
        esac
        ;;
    4)
        device_args="--device /dev/tenstorrent/$container_num"
        ;;
esac

# Launch the container with appropriate device mapping
docker run --rm $device_args \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -e TT_METAL_HOME=$(pwd) \
    -e PYTHONPATH=$(pwd) \
    -e LD_LIBRARY_PATH=$(pwd)/build/lib \
    -e ARCH_NAME=wormhole_b0 \
    -e LOGURU_LEVEL=INFO \
    ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64 \
    $@

