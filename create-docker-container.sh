#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_PATH="${SCRIPT_DIR}/dockerfile/Dockerfile.basic-dev"
IMAGE_NAME="tt-metal-basic-dev:local"
CONTAINER_NAME="tt-metal-basic-dev-container"
MANAGED_LABEL="tt-metal.managed_by=create-docker-container.sh"
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

if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
  echo "Error: Dockerfile not found at ${DOCKERFILE_PATH}" >&2
  exit 1
fi

echo "Removing previously managed containers..."
mapfile -t MANAGED_CONTAINERS < <("${DOCKER_CMD[@]}" ps -aq --filter "label=${MANAGED_LABEL}")
if ((${#MANAGED_CONTAINERS[@]})); then
  "${DOCKER_CMD[@]}" rm -f "${MANAGED_CONTAINERS[@]}"
fi

if "${DOCKER_CMD[@]}" container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  "${DOCKER_CMD[@]}" rm -f "${CONTAINER_NAME}"
fi

echo "Removing previously managed images..."
mapfile -t MANAGED_IMAGES < <("${DOCKER_CMD[@]}" images -q --filter "label=${MANAGED_LABEL}" | sort -u)
if ((${#MANAGED_IMAGES[@]})); then
  "${DOCKER_CMD[@]}" rmi -f "${MANAGED_IMAGES[@]}"
fi

if "${DOCKER_CMD[@]}" image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  "${DOCKER_CMD[@]}" rmi -f "${IMAGE_NAME}" || true
fi

echo "Building image ${IMAGE_NAME} from ${DOCKERFILE_PATH}..."
"${DOCKER_CMD[@]}" build \
  --file "${DOCKERFILE_PATH}" \
  --tag "${IMAGE_NAME}" \
  --label "${MANAGED_LABEL}" \
  "${SCRIPT_DIR}"

echo "Creating and starting container ${CONTAINER_NAME}..."
"${DOCKER_CMD[@]}" run -d \
  --name "${CONTAINER_NAME}" \
  --label "${MANAGED_LABEL}" \
  --workdir /workspace \
  --volume "${SCRIPT_DIR}:/workspace" \
  "${IMAGE_NAME}" \
  tail -f /dev/null

echo "Container is ready: ${CONTAINER_NAME}"
