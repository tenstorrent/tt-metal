#!/bin/bash

set -e

if [[ -z "${TT_METAL_DOCKER_IMAGE_TAG}" ]]; then
  echo "TT_METAL_DOCKER_IMAGE_TAG is not set or is empty, setting to ubuntu-22.04-amd64:latest"
  TT_METAL_DOCKER_IMAGE_TAG="ubuntu-22.04-amd64:latest"
else
  echo "TT_METAL_DOCKER_IMAGE_TAG is set to ${TT_METAL_DOCKER_IMAGE_TAG}"
fi

GID=$(id -g "${USER}")

if [[ -z "${TT_METAL_HOME}" ]]; then
  TT_METAL_HOME=$(git rev-parse --show-toplevel)
else
  echo "TT_METAL_DOCKER_IMAGE_TAG is set to ${TT_METAL_DOCKER_IMAGE_TAG}"
fi

[ -d ${TT_METAL_HOME}/.pipcache ] || mkdir ${TT_METAL_HOME}/.pipcache

function run_docker_common {
    # Split the arguments into docker options and command
    local docker_opts=()
    local cmd=()
    local append_cmd=false
    for arg in "$@"; do
        if $append_cmd; then
            cmd+=("$arg")
        elif [[ $arg == "--" && $append_cmd == false ]]; then
            append_cmd=true
        else
            docker_opts+=("$arg")
        fi
    done

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
        -e XDG_CACHE_HOME=${TT_METAL_HOME}/.pipcache \
        -e SILENT=${SILENT} \
		    -e VERBOSE=${VERBOSE} \
        -u ${UID}:${GID} \
        --net host \
        "${docker_opts[@]}" \
        ${TT_METAL_DOCKER_IMAGE_TAG} \
        "${cmd[@]}"
}
