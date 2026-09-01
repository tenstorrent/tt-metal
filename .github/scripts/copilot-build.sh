#!/usr/bin/env bash
#
# Build tt-metal from inside the CI build image. Intended for coding agents
# (see AGENTS.md); harmless for humans, but you probably want build_metal.sh.
#
# Why a wrapper: the Copilot cloud agent cannot run inside a job `container:` -
# its runtime stages a git-proxy binary on the host and then looks it up at the
# container-remapped RUNNER_TEMP, so the session dies before starting. The agent
# therefore stays on the host and shells into the image for builds instead.
#
# Usage:  .github/scripts/copilot-build.sh [build_metal.sh args...]
# e.g.    .github/scripts/copilot-build.sh --build-metal-tests
#
set -euo pipefail

IMAGE="${TT_CI_BUILD_IMAGE:-ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-ci-build-amd64:latest}"
GARAGE_ENDPOINT="${TT_GARAGE_ENDPOINT:-http://garage.garage.svc.cluster.local:3900}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CCACHE_HOST_DIR="${HOME}/.cache/tt-agent-ccache"

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: docker is not available. This script only works on a runner with" >&2
  echo "       docker access - see .github/workflows/copilot-setup-steps.yml." >&2
  exit 1
fi

mkdir -p "${CCACHE_HOST_DIR}"

DOCKER_ARGS=(
  --rm
  --user "$(id -u):$(id -g)"
  -v "${REPO_ROOT}:/work" -w /work
  -v "${CCACHE_HOST_DIR}:/ccache"
  -e HOME=/tmp
  -e CCACHE_DIR=/ccache
  -e CCACHE_COMPRESS=true
  -e CCACHE_COMPRESSLEVEL=3
)

# Credentials arrive as *Agents* secrets, which the platform exposes to the agent
# as plain environment variables. Copilot has no access to Actions secrets.
if [[ -n "${GARAGE_S3_ACCESS_KEY:-}" && -n "${GARAGE_S3_SECRET_KEY:-}" ]]; then
  # read-only: the agent reads CI's warm cache but must never write to it. A
  # writable shared cache would let agent-executed code plant objects that
  # trusted CI builds later consume. Within a session, repeated builds are
  # served by the local cache mounted at /ccache above.
  #
  # NOTE: this flag is a guardrail, not a boundary - it is client-side, and the
  # agent holds the credentials in its own environment. A genuinely read-only
  # Garage key is the real fix; none exists yet (only one read-write pair is
  # provisioned). Tracked in the PR discussion.
  export CCACHE_REMOTE_STORAGE="s3://ccache|region=garage|prefix=tt-metal|endpoint_url=${GARAGE_ENDPOINT}|read-only=true"
  export AWS_ACCESS_KEY_ID="${GARAGE_S3_ACCESS_KEY}"
  export AWS_SECRET_ACCESS_KEY="${GARAGE_S3_SECRET_KEY}"
  export AWS_DEFAULT_REGION=garage
  export AWS_ENDPOINT_URL_S3="${GARAGE_ENDPOINT}"
  # Passed by name, not value: `-e KEY=value` would put the secret in docker's
  # argv, visible to process inspection and diagnostics on the runner.
  DOCKER_ARGS+=(
    -e CCACHE_REMOTE_STORAGE
    -e AWS_ACCESS_KEY_ID
    -e AWS_SECRET_ACCESS_KEY
    -e AWS_DEFAULT_REGION
    -e AWS_ENDPOINT_URL_S3
  )
  echo "[copilot-build] remote ccache enabled (read-only)"
else
  echo "[copilot-build] WARNING: GARAGE_S3_ACCESS_KEY / GARAGE_S3_SECRET_KEY not set." >&2
  echo "[copilot-build]          Building with a COLD cache. A full cold build takes" >&2
  echo "[copilot-build]          over an hour and will most likely not finish." >&2
  echo "[copilot-build]          They must be repository *Agents* secrets, not Actions secrets." >&2
fi

echo "[copilot-build] image: ${IMAGE}"
echo "[copilot-build] args : ${*:-<none>}"

exec docker run "${DOCKER_ARGS[@]}" "${IMAGE}" bash -lc '
  set -euo pipefail
  git config --global --add safe.directory /work
  git config --global --add safe.directory "*"
  ./build_metal.sh --enable-ccache "$@"
  echo "--- ccache summary ---"
  ccache -sv 2>/dev/null | sed -n "/[Rr]emote storage/,/^$/p" || true
' _ "$@"
