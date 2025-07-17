#!/bin/bash

set -x

echo $@

# Launch the container with appropriate device mapping
docker run --rm --device /dev/tenstorrent \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    --network=host \
    ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64 \
    orted "$@"
