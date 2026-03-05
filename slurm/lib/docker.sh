#!/usr/bin/env bash
# docker.sh - Docker container execution for Slurm CI jobs
# Assumes slurm/lib/common.sh is already sourced.

# Guard against double-sourcing
[[ -n "${_SLURM_CI_DOCKER_LOADED:-}" ]] && return 0
_SLURM_CI_DOCKER_LOADED=1

require_cmd docker

# ---------------------------------------------------------------------------
# docker_login [registry]
# ---------------------------------------------------------------------------
# Login to a container registry. Defaults to GHCR.
# Auth precedence: DOCKER_USERNAME/DOCKER_PASSWORD > GITHUB_TOKEN
docker_login() {
    local registry="${1:-${GHCR_REGISTRY}}"

    local username="${DOCKER_USERNAME:-}"
    local password="${DOCKER_PASSWORD:-}"

    if [[ -z "$username" || -z "$password" ]]; then
        if [[ -n "${GITHUB_TOKEN:-}" ]]; then
            username="oauth2"
            password="$GITHUB_TOKEN"
        else
            log_error "No Docker credentials available (need DOCKER_USERNAME/DOCKER_PASSWORD or GITHUB_TOKEN)"
            return 1
        fi
    fi

    log_info "Logging in to registry: $registry"
    if echo "$password" | docker login "$registry" --username "$username" --password-stdin 2>&1; then
        log_info "Docker login successful"
    else
        log_error "Docker login failed for registry: $registry"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# docker_pull_with_retry IMAGE [max_retries] [backoff_seconds]
# ---------------------------------------------------------------------------
# Pull a Docker image with exponential backoff for transient failures.
docker_pull_with_retry() {
    local image="$1"
    local max_retries="${2:-${DOCKER_PULL_RETRIES:-3}}"
    local backoff="${3:-10}"
    local attempt=1

    log_info "Pulling image: $image (max $max_retries attempts)"

    while (( attempt <= max_retries )); do
        if timeout "${DOCKER_PULL_TIMEOUT:-300}" docker pull "$image" 2>&1; then
            log_info "Successfully pulled: $image"
            return 0
        fi

        if (( attempt == max_retries )); then
            log_error "Failed to pull image after $max_retries attempts: $image"
            return 1
        fi

        local wait=$(( backoff * attempt ))
        log_warn "Pull attempt $attempt/$max_retries failed, retrying in ${wait}s..."
        sleep "$wait"
        (( attempt++ ))
    done
}

# ---------------------------------------------------------------------------
# docker_run IMAGE COMMANDS [extra_docker_args...]
# ---------------------------------------------------------------------------
# Run commands inside a Docker container with standard TT-Metal configuration.
# Any arguments after COMMANDS are passed directly to docker run.
docker_run() {
    if [[ $# -lt 2 ]]; then
        log_error "Usage: docker_run IMAGE COMMANDS [extra_docker_args...]"
        return 1
    fi

    local image="$1"; shift
    local commands="$1"; shift
    local -a extra_args=("$@")

    local -a docker_args=(
        --rm
        --network host
        --log-driver local
        --log-opt max-size=500m
        --log-opt max-file=5
        -w /work
    )

    # UID/GID mapping
    docker_args+=( -u "$(id -u):$(id -g)" )

    # Device access
    if [[ -e /dev/tenstorrent ]]; then
        docker_args+=( --device /dev/tenstorrent )
    fi

    # Hugepages
    if [[ -d /dev/hugepages-1G ]]; then
        docker_args+=( -v /dev/hugepages-1G:/dev/hugepages-1G )
    fi

    # Core volume mounts
    docker_args+=(
        -v "${WORKSPACE:-.}:/work"
        -v /etc/passwd:/etc/passwd:ro
        -v /etc/shadow:/etc/shadow:ro
        -v /etc/bashrc:/etc/bashrc:ro
    )

    # Artifact directory
    if [[ -n "${ARTIFACT_DIR:-}" ]]; then
        mkdir -p "$ARTIFACT_DIR"
        docker_args+=( -v "${ARTIFACT_DIR}:/artifacts" )
    fi

    # Core environment variables
    docker_args+=(
        -e TT_METAL_HOME=/work
        -e PYTHONPATH=/work
        -e HOME=/work
        -e "ARCH_NAME=${ARCH_NAME:-wormhole_b0}"
        -e "LOGURU_LEVEL=${LOGURU_LEVEL:-INFO}"
    )

    # Forward all TT_METAL_* env vars
    _forward_env_prefix "TT_METAL_" docker_args

    # Forward all SLURM_* env vars
    _forward_env_prefix "SLURM_" docker_args

    # Extra env vars from DOCKER_EXTRA_ENV (newline-separated KEY=VALUE)
    if [[ -n "${DOCKER_EXTRA_ENV:-}" ]]; then
        local line
        while IFS= read -r line; do
            [[ -n "$line" ]] && docker_args+=( -e "$line" )
        done <<< "$DOCKER_EXTRA_ENV"
    fi

    # Extra volumes from DOCKER_EXTRA_VOLUMES (newline-separated host:container[:opts])
    if [[ -n "${DOCKER_EXTRA_VOLUMES:-}" ]]; then
        local vol
        while IFS= read -r vol; do
            [[ -n "$vol" ]] && docker_args+=( -v "$vol" )
        done <<< "$DOCKER_EXTRA_VOLUMES"
    fi

    # Extra options from DOCKER_EXTRA_OPTS (space-separated)
    if [[ -n "${DOCKER_EXTRA_OPTS:-}" ]]; then
        # Word-split intentionally for multiple options
        # shellcheck disable=SC2206
        docker_args+=( ${DOCKER_EXTRA_OPTS} )
    fi

    # Caller-provided extra args
    docker_args+=( "${extra_args[@]}" )

    log_info "Running in container: $image"
    log_debug "docker run ${docker_args[*]} $image bash -c ..."

    docker run "${docker_args[@]}" "$image" bash -c "set -euo pipefail; $commands"
    local rc=$?

    if [[ $rc -ne 0 ]]; then
        log_error "Container exited with code $rc"
    fi
    return $rc
}

# ---------------------------------------------------------------------------
# _forward_env_prefix PREFIX ARRAY_NAME
# ---------------------------------------------------------------------------
# Append -e flags for all exported env vars matching PREFIX to the named array.
_forward_env_prefix() {
    local prefix="$1"
    local -n arr="$2"
    local var
    while IFS='=' read -r var _; do
        if [[ "$var" == ${prefix}* ]]; then
            arr+=( -e "${var}=${!var}" )
        fi
    done < <(env)
}

# ---------------------------------------------------------------------------
# resolve_image [image_override]
# ---------------------------------------------------------------------------
# Resolve the Docker image to use.
# Priority:
#   1. Explicit image_override argument
#   2. DOCKER_IMAGE env var
#   3. Constructed from DOCKER_IMAGE_TAG + DOCKER_IMAGE_ARCH
#   4. DEFAULT_DOCKER_IMAGE from env.sh
# When using GHCR, falls back to Harbor if harbor.ci.tenstorrent.net is reachable.
resolve_image() {
    local image_override="${1:-}"

    if [[ -n "$image_override" ]]; then
        echo "$image_override"
        return 0
    fi

    if [[ -n "${DOCKER_IMAGE:-}" ]]; then
        echo "$DOCKER_IMAGE"
        return 0
    fi

    local image=""

    if [[ -n "${DOCKER_IMAGE_TAG:-}" ]]; then
        local arch="${DOCKER_IMAGE_ARCH:-tt-metalium/ubuntu-22.04-dev-amd64}"
        image="${GHCR_REPO}/${arch}:${DOCKER_IMAGE_TAG}"
    else
        image="${DEFAULT_DOCKER_IMAGE}"
    fi

    # Try Harbor as a mirror when available — faster pulls from on-prem
    if _harbor_is_reachable && [[ "$image" == "${GHCR_REGISTRY}"* ]]; then
        local harbor_image="${image/${GHCR_REGISTRY}/${HARBOR_REGISTRY}}"
        log_info "Harbor reachable, attempting mirror: $harbor_image"
        if docker pull "$harbor_image" &>/dev/null; then
            echo "$harbor_image"
            return 0
        fi
        log_warn "Harbor pull failed, falling back to GHCR: $image"
    fi

    echo "$image"
}

# ---------------------------------------------------------------------------
# _harbor_is_reachable
# ---------------------------------------------------------------------------
_harbor_is_reachable() {
    timeout 3 bash -c "echo >/dev/tcp/${HARBOR_REGISTRY}/443" 2>/dev/null
}

# ---------------------------------------------------------------------------
# docker_cleanup
# ---------------------------------------------------------------------------
# Remove stopped containers and dangling images older than 24 hours.
docker_cleanup() {
    log_info "Cleaning up Docker resources"

    local count
    count=$(docker ps -aq --filter status=exited | wc -l)
    if (( count > 0 )); then
        log_info "Removing $count stopped containers"
        docker container prune -f 2>&1 | while IFS= read -r line; do log_debug "$line"; done
    fi

    log_info "Removing dangling images older than 24h"
    docker image prune -f --filter "until=24h" 2>&1 | while IFS= read -r line; do log_debug "$line"; done

    log_info "Docker cleanup complete"
}
