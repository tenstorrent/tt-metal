#!/bin/bash

set -e

if [[ -z "${TT_METAL_DOCKER_IMAGE_TAG}" ]]; then
  echo "TT_METAL_DOCKER_IMAGE_TAG is not set or is empty, setting to ubuntu-22.04-amd64:latest"
  TT_METAL_DOCKER_IMAGE_TAG="ubuntu-22.04-amd64:latest"
else
  echo "TT_METAL_DOCKER_IMAGE_TAG is set to ${TT_METAL_DOCKER_IMAGE_TAG}"
fi

if [[ -z "${ARCH_NAME}" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

if [[ $# -eq 0 ]] ; then
    echo 'You must provide an argument to run in docker!'
    exit 1
fi
TT_METAL_HOME=$(git rev-parse --show-toplevel)
# Allows this script to be called anywhere in the tt-metal repo
source $TT_METAL_HOME/scripts/docker/run_docker_func.sh

run_docker_common "$@"
