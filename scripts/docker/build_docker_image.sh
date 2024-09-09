#!/bin/bash

TT_METAL_DOCKERFILE="${1:-ubuntu-20.04-amd64}"
TT_METAL_DOCKER_IMAGE_TAG="${2:-$TT_METAL_DOCKERFILE}"

TT_METAL_HOME=$(git rev-parse --show-toplevel)
(
  cd ${TT_METAL_HOME} || exit
  docker build -f dockerfile/${TT_METAL_DOCKERFILE}.Dockerfile -t ${TT_METAL_DOCKER_IMAGE_TAG} .
)
