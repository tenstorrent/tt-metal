#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="tt-metal-basic-dev-container"
DOCKER_CMD=(docker)

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not in PATH." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "Error: cannot access Docker daemon and sudo is unavailable." >&2
    exit 1
  fi
  echo "Docker daemon requires elevated permissions; using sudo."
  DOCKER_CMD=(sudo docker)
fi

if ! "${DOCKER_CMD[@]}" container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Error: container ${CONTAINER_NAME} does not exist." >&2
  echo "Run ./create-docker-container.sh first." >&2
  exit 1
fi

if [[ "$("${DOCKER_CMD[@]}" inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
  "${DOCKER_CMD[@]}" start "${CONTAINER_NAME}" >/dev/null
fi

"${DOCKER_CMD[@]}" exec -it "${CONTAINER_NAME}" bash
