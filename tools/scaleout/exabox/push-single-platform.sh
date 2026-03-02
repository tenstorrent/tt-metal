#!/usr/bin/env bash
# Push a single-platform image by stripping any manifest list.
# Use this when your image was created from a multi-platform base (e.g. docker commit)
# and "docker push" fails with "manifest list ... not all of them are available locally".
#
# Usage:
#   docker commit <container> ghcr.io/<user>/exabox-metal-test-dmadictt:amd64
#   tools/scaleout/exabox/push-single-platform.sh ghcr.io/dmadictt/exabox-metal-test-dmadictt:amd64
#
# Optional: set PUSH=0 to only save/load/tag without pushing.
set -e
IMAGE="${1:?Usage: $0 <image:tag>}"
PUSH="${PUSH:-1}"
TAR="${TMPDIR:-/tmp}/exabox-metal-single-platform-$$.tar"

# Save by image ID so we get the single platform image, not the manifest list.
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
echo "Saving image ID ${IMAGE_ID} to ${TAR}..."
docker save "$IMAGE_ID" -o "$TAR"

echo "Removing local image..."
docker rmi "$IMAGE" 2>/dev/null || true

echo "Loading from tar (produces single-platform image)..."
LOADED=$(docker load -i "$TAR" | sed -n 's/^Loaded image ID: sha256:\(.*\)/sha256:\1/p; s/^Loaded image: \(.*\)/\1/p' | head -1)
rm -f "$TAR"

# Reload may not preserve tag; ensure our tag points at the loaded image.
if [[ "$LOADED" == sha256:* ]]; then
  echo "Tagging loaded image as ${IMAGE}"
  docker tag "$LOADED" "$IMAGE"
fi

if [[ "$PUSH" == 1 ]]; then
  echo "Pushing ${IMAGE}..."
  docker push "$IMAGE"
  echo "Pushed ${IMAGE} successfully."
fi
