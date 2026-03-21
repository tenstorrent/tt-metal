#!/usr/bin/env bash
# docker_run.sh - CLI wrapper around lib/docker.sh docker_run()
# Usage: docker_run.sh --image IMAGE --command "CMDS" [OPTIONS]
#
# Equivalent to .github/actions/docker-run/action.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib docker

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

IMAGE=""
COMMAND=""
DEVICE=""
INSTALL_WHEEL=0
EXTRA_DOCKER_ARGS=()

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") --image IMAGE --command "CMDS" [OPTIONS]

Run commands inside a Docker container with standard TT-Metal configuration.

Required:
  --image IMAGE         Docker image (or set DOCKER_IMAGE env var)
  --command "CMDS"      Commands to execute inside the container

Options:
  --device PATH         Device path to mount (e.g. /dev/tenstorrent).
                        Detected automatically if not specified.
  --install-wheel       Install .whl from workspace before running commands
  --docker-opts "OPTS"  Additional docker run arguments (space-separated)
  -h, --help            Show this help message

Environment:
  WORKSPACE             Host path mounted as /work (default: cwd)
  DOCKER_IMAGE          Fallback image when --image is omitted
  DOCKER_EXTRA_ENV      Newline-separated KEY=VALUE pairs forwarded to container
  DOCKER_EXTRA_VOLUMES  Newline-separated host:container[:opts] mounts
  DOCKER_EXTRA_OPTS     Space-separated extra docker run options
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)         IMAGE="$2";              shift 2 ;;
        --command)       COMMAND="$2";             shift 2 ;;
        --device)        DEVICE="$2";              shift 2 ;;
        --install-wheel) INSTALL_WHEEL=1;          shift   ;;
        --docker-opts)   EXTRA_DOCKER_ARGS+=($2);  shift 2 ;;
        -h|--help)       usage 0 ;;
        *)               log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ -z "$COMMAND" ]]; then
    log_error "--command is required"
    usage 1
fi

# Resolve image: explicit arg > resolve_image helper
if [[ -n "$IMAGE" ]]; then
    RESOLVED_IMAGE="$IMAGE"
else
    RESOLVED_IMAGE="$(resolve_image)"
fi

# ---------------------------------------------------------------------------
# Build the command string
# ---------------------------------------------------------------------------

FULL_COMMAND=""

if [[ "$INSTALL_WHEEL" == "1" ]]; then
    read -r -d '' WHEEL_SNIPPET <<'WHEEL_EOF' || true
WHEEL_FILES=(*.whl)
if [ ! -e "${WHEEL_FILES[0]}" ]; then
    echo "WARN: No .whl file found in workspace. Skipping wheel installation."
elif [ ${#WHEEL_FILES[@]} -gt 1 ]; then
    echo "ERROR: Multiple .whl files found: ${WHEEL_FILES[*]}"
    exit 1
elif uv pip install "${WHEEL_FILES[0]}"; then
    echo "INFO: Installed wheel '${WHEEL_FILES[0]}'"
else
    echo "WARN: Failed to install wheel '${WHEEL_FILES[0]}'. Continuing."
fi
WHEEL_EOF
    FULL_COMMAND="${WHEEL_SNIPPET}"$'\n'"${COMMAND}"
else
    FULL_COMMAND="$COMMAND"
fi

# ---------------------------------------------------------------------------
# Optional explicit device override
# ---------------------------------------------------------------------------

if [[ -n "$DEVICE" ]]; then
    EXTRA_DOCKER_ARGS+=( --device "$DEVICE" )
fi

# ---------------------------------------------------------------------------
# Login, pull, run
# ---------------------------------------------------------------------------

docker_login
docker_pull_with_retry "$RESOLVED_IMAGE"
docker_run "$RESOLVED_IMAGE" "$FULL_COMMAND" "${EXTRA_DOCKER_ARGS[@]+"${EXTRA_DOCKER_ARGS[@]}"}"
