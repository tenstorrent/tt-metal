#!/bin/bash

set -x

case $OMPI_COMM_WORLD_LOCAL_RANK in
    0)
        device_args="--device /dev/tenstorrent/0" # set from ansible inventory?
        ;;
    1)
        device_args="--device /dev/tenstorrent/1" # set from ansible inventory?
        ;;
esac

docker run --rm $device_args -v /dev/hugepages-1G:/dev/hugepages-1G ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64 ${@:1}
