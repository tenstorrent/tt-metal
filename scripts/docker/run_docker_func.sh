#!/bin/bash

set -e


GID=$(id -g "${USER}")

if [[ -z "${TT_METAL_HOME}" ]]; then
  TT_METAL_HOME=$(git rev-parse --show-toplevel)
fi

function run_docker_common {

    docker run \
        --rm \
        -v ${TT_METAL_HOME}:/${TT_METAL_HOME} \
        -v /home:/home \
        -v /dev/hugepages-1G:/dev/hugepages-1G \
        -v /etc/group:/etc/group:ro \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/shadow:/etc/shadow:ro \
        -w ${TT_METAL_HOME} \
        -e TT_METAL_HOME=${TT_METAL_HOME} \
        -e LOGURU_LEVEL=${LOGURU_LEVEL} \
        -e LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
        -e ARCH_NAME=${ARCH_NAME} \
        -e PYTHONPATH=${TT_METAL_HOME} \
        -e SILENT=${SILENT} \
		    -e VERBOSE=${VERBOSE} \
        -u ${UID}:${GID} \
        --net host \
        "$1" \
        "$2:latest" \
        "$3"
}
