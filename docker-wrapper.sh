#!/bin/bash

set -x

docker run --rm --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64 ${@:1}
