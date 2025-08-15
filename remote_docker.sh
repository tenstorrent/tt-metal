#!/bin/bash

HOST=$1
shift

echo "$(date): Called with HOST=$HOST, COMMAND=$*" >> /home/aliu/tt-metal/mpi_wrapper_calls.log

ssh -l aliu "$HOST" sudo docker exec \
  -e PYTHONPATH=/home/aliu/tt-metal \
  -e TT_METAL_HOME=/home/aliu/tt-metal \
  aliu-host-mapped "$@"
