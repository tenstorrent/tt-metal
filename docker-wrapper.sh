#!/bin/bash

set -x

case $(wc -l rankfile) in
    1)
        case $OMPI_COMM_WORLD_LOCAL_RANK in
            0)
                device_args="--device /dev/tenstorrent/"
                ;;
        esac
        ;;
    2)
        case $OMPI_COMM_WORLD_LOCAL_RANK in
            0)
                device_args="--device /dev/tenstorrent/0 --device /dev/tenstorrent/1"
                ;;
            1)
                device_args="--device /dev/tenstorrent/2 --device /dev/tenstorrent/3"
                ;;
        esac
        ;;
    4)
        case $OMPI_COMM_WORLD_LOCAL_RANK in
            0)
                device_args="--device /dev/tenstorrent/0"
                ;;
            1)
                device_args="--device /dev/tenstorrent/1"
                ;;
            2)
                device_args="--device /dev/tenstorrent/2"
                ;;
            3)
                device_args="--device /dev/tenstorrent/3"
                ;;
        esac
        ;;
esac
docker run --rm $device_args --network host -v /home/ansible/actions-runner/_work:/home/ansible/actions-runner/_work -v /dev/hugepages-1G:/dev/hugepages-1G ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64 ${@:1}
