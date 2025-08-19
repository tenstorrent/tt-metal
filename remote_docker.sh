#!/bin/bash
HOST=$1
shift

export USER=local-aliu
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=/home/$USER/tt-metal
export PYTHONPATH=/home/$USER/tt-metal
export TT_METAL_ENV=dev

echo "$(date): Called with HOST=$HOST, COMMAND=$*" >> /home/$USER/tt-metal/mpi_wrapper_calls.log

ssh -l "$USER" "$HOST" sudo docker exec \
  -e ARCH_NAME=wormhole_b0 \
  -e TT_METAL_HOME=/home/$USER/tt-metal \
  -e PYTHONPATH=/home/$USER/tt-metal \
  -e TT_METAL_ENV=dev \
  local-aliu-host-mapped "$@"
