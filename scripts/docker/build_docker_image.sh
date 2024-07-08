#!/bin/bash

TT_METAL_DOCKER_IMAGE_TAG=${1:-ubuntu-22.04-amd64:latest}

TT_METAL_HOME=$(git rev-parse --show-toplevel)
(
  cd ${TT_METAL_HOME} || exit
  docker build -f dockerfile/ubuntu-22.04-amd64.Dockerfile -t ${TT_METAL_DOCKER_IMAGE_TAG} .
)
