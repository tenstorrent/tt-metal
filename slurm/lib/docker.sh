#!/usr/bin/env bash
# docker.sh - Docker container execution for Slurm CI jobs
# Assumes slurm/lib/common.sh is already sourced.

# Guard against double-sourcing
[[ -n "${_SLURM_CI_DOCKER_LOADED:-}" ]] && return 0
_SLURM_CI_DOCKER_LOADED=1

if [[ "${NO_DOCKER:-0}" != "1" ]]; then
    require_cmd docker
fi

# ---------------------------------------------------------------------------
# native_run COMMANDS
# ---------------------------------------------------------------------------
# Run commands directly on the host using the local python_env venv.
# Used when NO_DOCKER=1 — the workspace is a pre-built checkout on NFS with
# all deps already installed by create_venv.sh.
native_run() {
    local commands="$1"
    local ws="${WORKSPACE:-.}"
    local venv_dir="${ws}/python_env"

    log_info "Running natively (NO_DOCKER=1)"
    log_info "  WORKSPACE=${ws}"

    if [[ -d "${venv_dir}" ]]; then
        # --- Diagnostics: venv state before activation ---
        log_info "Venv directory: ${venv_dir}"
        if [[ -f "${venv_dir}/pyvenv.cfg" ]]; then
            log_info "pyvenv.cfg contents:"
            while IFS= read -r line; do log_info "  ${line}"; done < "${venv_dir}/pyvenv.cfg"
        else
            log_warn "pyvenv.cfg NOT FOUND in ${venv_dir}"
        fi

        local py_bin="${venv_dir}/bin/python3"
        if [[ -e "${py_bin}" ]]; then
            if [[ -L "${py_bin}" ]]; then
                log_info "python3 binary: symlink -> $(readlink -f "${py_bin}" 2>/dev/null || readlink "${py_bin}")"
            else
                log_info "python3 binary: regular file ($(stat -c '%s bytes, inode %i' "${py_bin}" 2>/dev/null || stat -f '%z bytes' "${py_bin}"))"
            fi
        else
            log_warn "python3 binary NOT FOUND at ${py_bin}"
        fi

        # Check stdlib presence
        local stdlib_encodings
        stdlib_encodings=$(ls -d "${venv_dir}"/lib/python3*/encodings 2>/dev/null | head -1)
        if [[ -n "${stdlib_encodings}" ]]; then
            log_info "stdlib encodings found: ${stdlib_encodings}"
        else
            log_warn "stdlib encodings NOT found under ${venv_dir}/lib/python3*/"
            log_info "Contents of ${venv_dir}/lib/:"
            ls -la "${venv_dir}/lib/" 2>/dev/null | while IFS= read -r line; do log_info "  ${line}"; done
        fi

        # Ensure the venv's Python stdlib is bundled (self-contained).
        # uv's --managed-python copies the binary but not the stdlib;
        # without bundling, Python can't find 'encodings' on compute nodes.
        if ! ls "${venv_dir}"/lib/python3*/encodings/__init__.py &>/dev/null; then
            local bundle_script="${ws}/scripts/bundle_python_into_venv.sh"
            if [[ -x "${bundle_script}" ]]; then
                log_info "Bundling Python stdlib into venv for multi-host portability..."
                "${bundle_script}" "${venv_dir}" --force
            else
                log_warn "Python stdlib not bundled in venv and bundle script not found at ${bundle_script}"
                log_warn "Run: scripts/bundle_python_into_venv.sh python_env --force"
            fi
        fi

        # Fix pyvenv.cfg 'home' if it points to a path that doesn't exist on
        # this node.  uv --managed-python sets home to the uv cpython install
        # dir (e.g. /usr/local/share/uv/python/cpython-.../install/bin) which
        # won't exist on NFS compute nodes.  When Python can't resolve the
        # home path it falls back to its compiled-in prefix (/install) and
        # fails to find the stdlib.  Rewriting home to the venv's own bin/
        # makes Python find the bundled stdlib in python_env/lib/python3.X/.
        if [[ -f "${venv_dir}/pyvenv.cfg" ]]; then
            local cfg_home
            cfg_home=$(grep -E '^home\s*=' "${venv_dir}/pyvenv.cfg" | head -1 | sed 's/^home\s*=\s*//')
            if [[ -n "${cfg_home}" && ! -d "${cfg_home}" ]]; then
                local new_home
                new_home="$(cd "${venv_dir}/bin" && pwd)"
                log_info "pyvenv.cfg home '${cfg_home}' does not exist on this node"
                log_info "Rewriting pyvenv.cfg home -> ${new_home}"
                sed -i "s|^home\s*=.*|home = ${new_home}|" "${venv_dir}/pyvenv.cfg"
                log_info "Updated pyvenv.cfg:"
                while IFS= read -r line; do log_info "  ${line}"; done < "${venv_dir}/pyvenv.cfg"
            fi
        fi

        log_info "Activating venv: ${venv_dir}"
        # shellcheck disable=SC1091
        source "${venv_dir}/bin/activate"

        # --- Diagnostics: state after activation ---
        log_info "Post-activation:"
        log_info "  VIRTUAL_ENV=${VIRTUAL_ENV:-<not set>}"
        log_info "  PATH (first 3)=$(echo "$PATH" | tr ':' '\n' | head -3 | tr '\n' ':')"
        log_info "  which python3: $(which python3 2>/dev/null || echo 'not found')"
        log_info "  which pytest: $(which pytest 2>/dev/null || echo 'not found')"
    else
        log_warn "Venv directory not found: ${venv_dir}"
    fi

    export TT_METAL_HOME="${ws}"
    export PYTHONPATH="${ws}:${ws}/ttnn:${ws}/tools"
    export LD_LIBRARY_PATH="${ws}/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
    export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

    log_info "Final environment:"
    log_info "  TT_METAL_HOME=${TT_METAL_HOME}"
    log_info "  PYTHONPATH=${PYTHONPATH}"
    log_info "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    log_info "Running: ${commands}"

    (cd "${ws}" && bash -c "set -euo pipefail; $commands")
    local rc=$?

    if [[ $rc -ne 0 ]]; then
        log_error "Native run exited with code $rc"
    fi
    return $rc
}

# ---------------------------------------------------------------------------
# run_test COMMANDS [extra_docker_args...]
# ---------------------------------------------------------------------------
# Dispatch to native_run or docker_run based on NO_DOCKER.
run_test() {
    local commands="$1"; shift
    if [[ "${NO_DOCKER:-0}" == "1" ]]; then
        native_run "$commands"
    else
        docker_run "${DOCKER_IMAGE:?DOCKER_IMAGE not set}" "$commands" "$@"
    fi
}

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
        -w "${CONTAINER_WORKDIR:-/work}"
    )

    # UID/GID mapping
    docker_args+=( -u "$(id -u):$(id -g)" )

    # Device access
    if [[ -e "${TT_DEVICE_PATH:-/dev/tenstorrent}" ]]; then
        docker_args+=( --device "${TT_DEVICE_PATH:-/dev/tenstorrent}" )
    fi

    # Hugepages
    local _hp="${HUGEPAGES_PATH:-/dev/hugepages-1G}"
    if [[ -d "$_hp" ]]; then
        docker_args+=( -v "${_hp}:${_hp}" )
    fi

    # Core volume mounts
    docker_args+=(
        -v "${WORKSPACE:-.}:${CONTAINER_WORKDIR:-/work}"
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
        -e "TT_METAL_HOME=${CONTAINER_WORKDIR:-/work}"
        -e "PYTHONPATH=${CONTAINER_WORKDIR:-/work}"
        -e "HOME=${CONTAINER_WORKDIR:-/work}"
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

    # Skip image entrypoint (e.g. dev image starts sshd which fails as
    # non-root).  Slurm handles multi-node orchestration natively.
    docker_args+=( --entrypoint "" )

    # Local build detection: when the workspace contains a python_env from
    # create_venv.sh, the project has been built locally.  Rather than
    # activating the host venv (whose Python binary cannot find its stdlib
    # inside the container), extend PYTHONPATH to replicate the three paths
    # that the editable install's ttnn-custom.pth would add.
    #   LOCAL_VENV=1  force on
    #   LOCAL_VENV=0  force off
    #   unset         auto-detect from workspace
    local local_venv="${LOCAL_VENV:-auto}"
    if [[ "$local_venv" == "auto" ]]; then
        [[ -f "${WORKSPACE:-.}/python_env/bin/activate" ]] && local_venv=1 || local_venv=0
    fi

    if [[ "$local_venv" == "1" ]]; then
        local cwd="${CONTAINER_WORKDIR:-/work}"
        docker_args+=( -e "PYTHONPATH=${cwd}:${cwd}/ttnn:${cwd}/tools" )
        docker_args+=( -e "LD_LIBRARY_PATH=${cwd}/build/lib" )
        log_info "Local build detected, extending PYTHONPATH for: ${cwd}"
    fi

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
