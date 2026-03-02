#!/bin/bash
# Launched by mpirun on each node. Runs the given command in an ephemeral
# Docker container, forwarding MPI and TT-Metal env vars.
#
# tt-run sets per-rank env vars (TT_MESH_ID, TT_VISIBLE_DEVICES, etc.)
# via mpirun's -x flag. This wrapper captures them and passes them into
# the container so the process inside behaves as a proper MPI rank.
#
# Uses tt-metal from the container image only (no host tt-metal mount).
# Run as host uid:gid so the image does not need a specific user (works with
# tt-metalium and tt-media-inference-server). Override with TT_METAL_CONTAINER_USER
# for images that have that user (e.g. container_app_user).
#
# CONTAINER_IMAGE_TT_METAL must match the image's TT_METAL_HOME/WORKDIR (see tt-metal/dockerfile/Dockerfile:
# ENV TT_METAL_HOME=/tt-metal, WORKDIR /tt-metal). Override with TT_METAL_CONTAINER_TT_METAL for other images.

IMAGE="${TT_METAL_IMAGE:?Set TT_METAL_IMAGE to your container image}"
CONTAINER_IMAGE_TT_METAL="${TT_METAL_CONTAINER_TT_METAL:-/tt-metal}"
CONTAINER_USER="${TT_METAL_CONTAINER_USER:-$(id -u):$(id -g)}"
# Host tt-metal path: taken from TT_METAL_HOME in the environment (e.g. export in inference.sbatch).
# All forwarded vars that contain this path are remapped to CONTAINER_IMAGE_TT_METAL so paths resolve in the image.
HOST_TT_METAL="${TT_METAL_HOME:-/data/dmadic/tt-metal}"

# Forward MPI/PMI env as-is; remap any host tt-metal path to container path so paths resolve inside the image.
ENV_ARGS=()
while IFS='=' read -r key value; do
  value="${value//${HOST_TT_METAL}/${CONTAINER_IMAGE_TT_METAL}}"
  ENV_ARGS+=(-e "${key}=${value}")
done < <(env | grep -E '^(TT_|OMPI_|PMIX_|PMI_)')

ENV_ARGS+=(-e "HOME=/tmp")
ENV_ARGS+=(-e "TT_METAL_HOME=${CONTAINER_IMAGE_TT_METAL}")
ENV_ARGS+=(-e "TT_METAL_RUNTIME_ROOT=${CONTAINER_IMAGE_TT_METAL}")
ENV_ARGS+=(-e "PYTHONPATH=${CONTAINER_IMAGE_TT_METAL}")
# PMIx: use hash GDS instead of gds_shmem2 so job setup works across containers on different nodes (no shared memory).
ENV_ARGS+=(-e "PMIX_MCA_gds=hash")

# Writable dir for tt-metal generated artifacts (image's /tt-metal is read-only for host uid).
GENERATED_MOUNT=$(mktemp -d)
trap 'rm -rf "$GENERATED_MOUNT"' EXIT

docker run --rm \
  --privileged \
  --net=host \
  --shm-size=4g \
  --user "${CONTAINER_USER}" \
  -v /dev/hugepages:/dev/hugepages \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /etc/udev/rules.d:/etc/udev/rules.d \
  -v /lib/modules:/lib/modules \
  -v /var/run/tenstorrent:/var/run/tenstorrent \
  -v "${GENERATED_MOUNT}:${CONTAINER_IMAGE_TT_METAL}/generated" \
  "${ENV_ARGS[@]}" \
  -w "${CONTAINER_IMAGE_TT_METAL}" \
  "$IMAGE" \
  bash -c "if [ -f ${CONTAINER_IMAGE_TT_METAL}/python_env/bin/activate ]; then source ${CONTAINER_IMAGE_TT_METAL}/python_env/bin/activate; elif [ -f /opt/venv/bin/activate ]; then source /opt/venv/bin/activate; fi; export PYTHONPATH=${CONTAINER_IMAGE_TT_METAL}; exec $(printf '%q ' "$@")"
