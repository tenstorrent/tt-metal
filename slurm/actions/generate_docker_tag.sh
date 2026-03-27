#!/usr/bin/env bash
# generate_docker_tag.sh - Generate a deterministic Docker tag from Dockerfile + context.
# Equivalent to .github/actions/generate-docker-tag/action.yml
#
# Usage: generate_docker_tag.sh [--dockerfile PATH] [--context DIR]
# Outputs the tag to stdout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

DOCKERFILE=""
CONTEXT_DIR="."

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dockerfile) DOCKERFILE="$2"; shift 2 ;;
        --context)    CONTEXT_DIR="$2"; shift 2 ;;
        *)            log_warn "Unknown option: $1"; shift ;;
    esac
done

if [[ -z "${DOCKERFILE}" ]]; then
    DOCKERFILE="${CONTEXT_DIR}/Dockerfile"
fi

if [[ ! -f "${DOCKERFILE}" ]]; then
    log_fatal "Dockerfile not found: ${DOCKERFILE}"
fi

require_cmd sha256sum

HASH_INPUT="$(cat "${DOCKERFILE}")"

if [[ -f "${CONTEXT_DIR}/requirements.txt" ]]; then
    HASH_INPUT+="$(cat "${CONTEXT_DIR}/requirements.txt")"
fi
if [[ -f "${CONTEXT_DIR}/setup.py" ]]; then
    HASH_INPUT+="$(cat "${CONTEXT_DIR}/setup.py")"
fi
if [[ -f "${CONTEXT_DIR}/pyproject.toml" ]]; then
    HASH_INPUT+="$(cat "${CONTEXT_DIR}/pyproject.toml")"
fi

TAG="$(echo -n "${HASH_INPUT}" | sha256sum | cut -c1-12)"

log_info "Generated Docker tag: ${TAG} (from ${DOCKERFILE})"
echo "${TAG}"
