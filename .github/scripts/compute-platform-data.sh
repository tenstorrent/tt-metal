#!/bin/bash
# Compute platform-specific Docker image tags and check existence
#
# Usage: compute-platform-data.sh <version> <repo> [--force-rebuild] [--check-exists]
#
# Arguments:
#   version       Ubuntu version (e.g., "22.04" or "24.04")
#   repo          GitHub repository (e.g., "owner/repo")
#   --force-rebuild  Treat all images as missing (optional)
#   --check-exists   Check if images exist in registry (optional, default: true)
#
# Output: JSON object with tags, existence flags, and metadata
#
# Example:
#   compute-platform-data.sh "22.04" "tenstorrent/tt-metal" --check-exists

set -euo pipefail

usage() {
    echo "Usage: $0 <version> <repo> [--force-rebuild] [--check-exists]" >&2
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

VERSION="$1"
REPO="$2"
shift 2

FORCE_REBUILD=false
CHECK_EXISTS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --check-exists)
            CHECK_EXISTS=true
            shift
            ;;
        --no-check-exists)
            CHECK_EXISTS=false
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

DISTRO="ubuntu"
ARCH="amd64"

# Determine Python version based on Ubuntu version
if [ "$VERSION" = "24.04" ]; then
    PYTHON_VERSION="3.12"
else
    PYTHON_VERSION="3.10"
fi

VERSION_NODOT="${VERSION//.}"

# Image names
CI_BUILD_LIGHT_NAME="${DISTRO}-${VERSION}-ci-build-light-${ARCH}"
CI_BUILD_NAME="${DISTRO}-${VERSION}-ci-build-${ARCH}"
CI_TEST_LIGHT_NAME="${DISTRO}-${VERSION}-ci-test-light-${ARCH}"
CI_TEST_NAME="${DISTRO}-${VERSION}-ci-test-${ARCH}"
DEV_LIGHT_NAME="${DISTRO}-${VERSION}-dev-light-${ARCH}"
DEV_NAME="${DISTRO}-${VERSION}-dev-${ARCH}"
BASIC_DEV_NAME="${DISTRO}-${VERSION}-basic-dev-${ARCH}"
BASIC_TTNN_NAME="${DISTRO}-${VERSION}-basic-ttnn-runtime-${ARCH}"

# Extra files for hash computation.
# Tool consumers include Dockerfile.tools so tool-layer build behavior remains
# part of their tag material. Bake target metadata is generated per image below.
TOOL_EXTRA_FILE="dockerfile/Dockerfile.tools"

combine_hashes() {
    printf '%s\n' "$@" | sha1sum | cut -d' ' -f1
}

manifest_exists() {
    docker manifest inspect "$1" > /dev/null 2>&1
}

bake_print() {
    local target="$1"
    local -a bake_cmd

    if docker buildx version > /dev/null 2>&1; then
        bake_cmd=(docker buildx)
    elif command -v docker-buildx > /dev/null 2>&1; then
        bake_cmd=(docker-buildx)
    else
        echo "ERROR: Docker Buildx is required to compute Bake target hash inputs." >&2
        exit 1
    fi

    UBUNTU_VERSION="$VERSION" PYTHON_VERSION="$PYTHON_VERSION" \
        "${bake_cmd[@]}" bake -f dockerfile/docker-bake.hcl --progress=quiet --print "$target"
}

write_bake_target_hash_input() {
    local target="$1"
    local output="$2"

    bake_print "$target" | jq -S -c --arg target "$target" '
        .target[$target] // error("Bake target not found: " + $target)
        | del(.tags, .output, ."cache-from", ."cache-to")
        | {bake_target: $target, build: .}
    ' > "$output"

    if [ ! -s "$output" ]; then
        echo "ERROR: Failed to compute Bake hash input for target: $target" >&2
        exit 1
    fi
}

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# Strip comment-only lines from Dockerfile content before hashing so pure
# comment edits (e.g. updating the BUILD INSTRUCTIONS header) don't invalidate
# image tags. Only whole-line comments (optional leading whitespace then '#')
# are removed; lines like `ARG FOO=bar  # note` keep their inline comment since
# the leading token is not '#'. BuildKit parser directives (`# syntax=`,
# `# escape=`) are preserved because they are build-significant.
strip_dockerfile_comments() {
    awk '
        /^#[[:space:]]*syntax=/ { print; next }
        /^#[[:space:]]*escape=/ { print; next }
        /^[[:space:]]*#/ { next }
        { print }
    '
}

write_dockerfile_prefix() {
    local stop_pattern="$1"
    local output="$2"
    awk -v stop_pattern="$stop_pattern" '
        $0 ~ stop_pattern { exit }
        { print }
    ' dockerfile/Dockerfile | strip_dockerfile_comments > "$output"
    if [ ! -s "$output" ]; then
        echo "ERROR: Dockerfile split produced an empty file for stop pattern: $stop_pattern" >&2
        exit 1
    fi
}

write_dockerfile_stage() {
    local start_pattern="$1"
    local stop_pattern="$2"
    local output="$3"
    awk -v start_pattern="$start_pattern" -v stop_pattern="$stop_pattern" '
        $0 ~ start_pattern { in_stage = 1 }
        in_stage && $0 ~ stop_pattern { exit }
        in_stage { print }
    ' dockerfile/Dockerfile | strip_dockerfile_comments > "$output"
    if [ ! -s "$output" ]; then
        echo "ERROR: Dockerfile stage extraction produced an empty file for start pattern: $start_pattern" >&2
        exit 1
    fi
}

combine_dockerfile_parts() {
    local output="$1"
    shift
    : > "$output"
    for part in "$@"; do
        cat "$part" >> "$output"
        printf '\n' >> "$output"
    done
}

CI_BUILD_LIGHT_DOCKERFILE="$TMP_DIR/Dockerfile.ci-build-light"
CI_BUILD_DOCKERFILE="$TMP_DIR/Dockerfile.ci-build"
CI_TEST_LIGHT_DOCKERFILE="$TMP_DIR/Dockerfile.ci-test-light"
CI_TEST_DOCKERFILE="$TMP_DIR/Dockerfile.ci-test"
DEV_LIGHT_DOCKERFILE="$TMP_DIR/Dockerfile.dev-light"
DEV_DOCKERFILE="$TMP_DIR/Dockerfile.dev"
CI_TEST_LIGHT_STAGE="$TMP_DIR/stage.ci-test-light"
CI_TEST_STAGE="$TMP_DIR/stage.ci-test"
DEV_LIGHT_STAGE="$TMP_DIR/stage.dev-light"
DEV_STAGE="$TMP_DIR/stage.dev"
CI_BUILD_LIGHT_BAKE_INPUT="$TMP_DIR/bake.ci-build-light.json"
CI_BUILD_BAKE_INPUT="$TMP_DIR/bake.ci-build.json"
CI_TEST_LIGHT_BAKE_INPUT="$TMP_DIR/bake.ci-test-light.json"
CI_TEST_BAKE_INPUT="$TMP_DIR/bake.ci-test.json"
DEV_LIGHT_BAKE_INPUT="$TMP_DIR/bake.dev-light.json"
DEV_BAKE_INPUT="$TMP_DIR/bake.dev.json"
BASIC_DEV_BAKE_INPUT="$TMP_DIR/bake.basic-dev.json"
BASIC_TTNN_BAKE_INPUT="$TMP_DIR/bake.basic-ttnn-runtime.json"
MANYLINUX_BAKE_INPUT="$TMP_DIR/bake.manylinux.json"

# Hash only the Dockerfile stages needed to build each target. This keeps
# unrelated non-light venv stages from changing light image tags.
write_dockerfile_prefix '^FROM ci-build-light AS ci-build$' "$CI_BUILD_LIGHT_DOCKERFILE"
write_dockerfile_prefix '^FROM ci-build-light AS ci-test-light$' "$CI_BUILD_DOCKERFILE"
write_dockerfile_stage '^FROM ci-build-light AS ci-test-light$' '^FROM ci-test-light AS ci-test$' "$CI_TEST_LIGHT_STAGE"
write_dockerfile_stage '^FROM ci-test-light AS ci-test$' '^FROM ci-test-light AS dev-light$' "$CI_TEST_STAGE"
write_dockerfile_stage '^FROM ci-test-light AS dev-light$' '^FROM dev-light AS dev$' "$DEV_LIGHT_STAGE"
write_dockerfile_stage '^FROM dev-light AS dev$' '^FROM base AS release$' "$DEV_STAGE"

combine_dockerfile_parts "$CI_TEST_LIGHT_DOCKERFILE" "$CI_BUILD_LIGHT_DOCKERFILE" "$CI_TEST_LIGHT_STAGE"
combine_dockerfile_parts "$CI_TEST_DOCKERFILE" "$CI_TEST_LIGHT_DOCKERFILE" "$CI_TEST_STAGE"
combine_dockerfile_parts "$DEV_LIGHT_DOCKERFILE" "$CI_TEST_LIGHT_DOCKERFILE" "$DEV_LIGHT_STAGE"
combine_dockerfile_parts "$DEV_DOCKERFILE" "$DEV_LIGHT_DOCKERFILE" "$DEV_STAGE"

# Compute hashes
write_bake_target_hash_input "ci-build-light" "$CI_BUILD_LIGHT_BAKE_INPUT"
write_bake_target_hash_input "ci-build" "$CI_BUILD_BAKE_INPUT"
write_bake_target_hash_input "ci-test-light" "$CI_TEST_LIGHT_BAKE_INPUT"
write_bake_target_hash_input "ci-test" "$CI_TEST_BAKE_INPUT"
write_bake_target_hash_input "dev-light" "$DEV_LIGHT_BAKE_INPUT"
write_bake_target_hash_input "dev" "$DEV_BAKE_INPUT"
write_bake_target_hash_input "basic-dev" "$BASIC_DEV_BAKE_INPUT"
write_bake_target_hash_input "basic-ttnn-runtime" "$BASIC_TTNN_BAKE_INPUT"
write_bake_target_hash_input "manylinux" "$MANYLINUX_BAKE_INPUT"

CI_BUILD_LIGHT_DOCKERFILE_HASH=$(.github/scripts/dockerfile-hash.sh "$CI_BUILD_LIGHT_DOCKERFILE" "$CI_BUILD_LIGHT_BAKE_INPUT" "$TOOL_EXTRA_FILE")
CI_BUILD_DOCKERFILE_HASH=$(.github/scripts/dockerfile-hash.sh "$CI_BUILD_DOCKERFILE" "$CI_BUILD_BAKE_INPUT" "$TOOL_EXTRA_FILE")
CI_TEST_LIGHT_DOCKERFILE_HASH=$(.github/scripts/dockerfile-hash.sh "$CI_TEST_LIGHT_DOCKERFILE" "$CI_TEST_LIGHT_BAKE_INPUT" "$TOOL_EXTRA_FILE")
CI_TEST_DOCKERFILE_HASH=$(.github/scripts/dockerfile-hash.sh "$CI_TEST_DOCKERFILE" "$CI_TEST_BAKE_INPUT" "$TOOL_EXTRA_FILE")
DEV_LIGHT_DOCKERFILE_HASH=$(.github/scripts/dockerfile-hash.sh "$DEV_LIGHT_DOCKERFILE" "$DEV_LIGHT_BAKE_INPUT" "$TOOL_EXTRA_FILE")
DEV_DOCKERFILE_HASH=$(.github/scripts/dockerfile-hash.sh "$DEV_DOCKERFILE" "$DEV_BAKE_INPUT" "$TOOL_EXTRA_FILE")
BASIC_DEV_HASH=$(.github/scripts/dockerfile-hash.sh dockerfile/Dockerfile.basic-dev "$BASIC_DEV_BAKE_INPUT" "$TOOL_EXTRA_FILE")
BASIC_TTNN_HASH=$(.github/scripts/dockerfile-hash.sh dockerfile/Dockerfile.basic-dev "$BASIC_TTNN_BAKE_INPUT" "$TOOL_EXTRA_FILE")
MANYLINUX_HASH=$(.github/scripts/dockerfile-hash.sh dockerfile/Dockerfile.manylinux "$MANYLINUX_BAKE_INPUT" "$TOOL_EXTRA_FILE")

# Compute separate hashes for the two venv images so ci-build-venv is reusable
# across ci-test-only dependency changes. Hash the resolved Bake target metadata
# for each image instead of the whole docker-bake.hcl, so unrelated Bake targets
# do not invalidate venv image tags.

CI_BUILD_VENV_DOCKERFILE="$TMP_DIR/Dockerfile.python.ci-build"
CI_BUILD_VENV_BAKE_INPUT="$TMP_DIR/bake.ci-build-venv.json"
CI_TEST_VENV_BAKE_INPUT="$TMP_DIR/bake.ci-test-venv.json"
# Split Dockerfile.python at the ci-test-venv stage boundary to hash only the
# ci-build-venv portion. The awk pattern is keyed on the exact stage name line
# "FROM ci-build-venv-builder AS ci-test-venv-builder" in Dockerfile.python.
# If those stage names are ever renamed, this pattern must be updated to match.
awk '
    /^FROM ci-build-venv-builder AS ci-test-venv-builder$/ { exit }
    { print }
' dockerfile/Dockerfile.python > "$CI_BUILD_VENV_DOCKERFILE"

if [ ! -s "$CI_BUILD_VENV_DOCKERFILE" ]; then
    echo "ERROR: awk split of Dockerfile.python produced an empty file." >&2
    echo "The stage pattern 'FROM ci-build-venv-builder AS ci-test-venv-builder' was not found." >&2
    echo "If Dockerfile.python stage names changed, update the awk pattern above." >&2
    exit 1
fi

write_bake_target_hash_input "ci-build-venv" "$CI_BUILD_VENV_BAKE_INPUT"
write_bake_target_hash_input "ci-test-venv" "$CI_TEST_VENV_BAKE_INPUT"

CI_BUILD_VENV_HASH=$(.github/scripts/dockerfile-hash.sh "$CI_BUILD_VENV_DOCKERFILE" "$CI_BUILD_VENV_BAKE_INPUT")
CI_TEST_VENV_HASH=$(.github/scripts/dockerfile-hash.sh dockerfile/Dockerfile.python "$CI_TEST_VENV_BAKE_INPUT")

# Compute canonical tool tag material without registry access. Final image hashes
# include this explicitly so a dev-image manifest hit can stand in for the
# upstream tool/venv chain on the common "everything already exists" path.
TOOL_TAGS=$(.github/scripts/compute-tool-tags.sh "$REPO")
TOOL_TAGS_HASH=$(echo "$TOOL_TAGS" | jq -S -c '.' | sha1sum | cut -d' ' -f1)

# Main image tags include the dependency hashes their targets consume. DEV_HASH
# also includes the prior main image hashes so the dev image can be used as a
# canary for the main-image dependency chain before issuing more registry calls.
CI_BUILD_LIGHT_HASH=$(combine_hashes "$CI_BUILD_LIGHT_DOCKERFILE_HASH" "$TOOL_TAGS_HASH")
CI_BUILD_HASH=$(combine_hashes "$CI_BUILD_DOCKERFILE_HASH" "$CI_BUILD_LIGHT_HASH" "$CI_BUILD_VENV_HASH")
CI_TEST_LIGHT_HASH=$(combine_hashes "$CI_TEST_LIGHT_DOCKERFILE_HASH" "$CI_BUILD_LIGHT_HASH" "$TOOL_TAGS_HASH")
CI_TEST_HASH=$(combine_hashes "$CI_TEST_DOCKERFILE_HASH" "$CI_TEST_LIGHT_HASH" "$CI_TEST_VENV_HASH")
DEV_LIGHT_HASH=$(combine_hashes "$DEV_LIGHT_DOCKERFILE_HASH" "$CI_TEST_LIGHT_HASH" "$TOOL_TAGS_HASH")
DEV_HASH=$(combine_hashes "$DEV_DOCKERFILE_HASH" "$CI_BUILD_LIGHT_HASH" "$CI_BUILD_HASH" "$CI_TEST_LIGHT_HASH" "$CI_TEST_HASH" "$DEV_LIGHT_HASH" "$CI_BUILD_VENV_HASH" "$CI_TEST_VENV_HASH")
BASIC_DEV_HASH=$(combine_hashes "$BASIC_DEV_HASH" "$TOOL_TAGS_HASH")
BASIC_TTNN_HASH=$(combine_hashes "$BASIC_TTNN_HASH" "$TOOL_TAGS_HASH")
MANYLINUX_HASH=$(combine_hashes "$MANYLINUX_HASH" "$TOOL_TAGS_HASH")

# Build tags
CI_BUILD_LIGHT_TAG="ghcr.io/${REPO}/tt-metalium/${CI_BUILD_LIGHT_NAME}:${CI_BUILD_LIGHT_HASH}"
CI_BUILD_TAG="ghcr.io/${REPO}/tt-metalium/${CI_BUILD_NAME}:${CI_BUILD_HASH}"
CI_TEST_LIGHT_TAG="ghcr.io/${REPO}/tt-metalium/${CI_TEST_LIGHT_NAME}:${CI_TEST_LIGHT_HASH}"
CI_TEST_TAG="ghcr.io/${REPO}/tt-metalium/${CI_TEST_NAME}:${CI_TEST_HASH}"
DEV_LIGHT_TAG="ghcr.io/${REPO}/tt-metalium/${DEV_LIGHT_NAME}:${DEV_LIGHT_HASH}"
DEV_TAG="ghcr.io/${REPO}/tt-metalium/${DEV_NAME}:${DEV_HASH}"
BASIC_DEV_TAG="ghcr.io/${REPO}/tt-metalium/${BASIC_DEV_NAME}:${BASIC_DEV_HASH}"
BASIC_TTNN_TAG="ghcr.io/${REPO}/tt-metalium/${BASIC_TTNN_NAME}:${BASIC_TTNN_HASH}"
MANYLINUX_TAG="ghcr.io/${REPO}/tt-metalium/manylinux-${ARCH}:${MANYLINUX_HASH}"

BASE_VENV="ghcr.io/${REPO}/tt-metalium/python-venv"
CI_BUILD_VENV_TAG="${BASE_VENV}/ci-build:${VERSION_NODOT}-${CI_BUILD_VENV_HASH}"
CI_TEST_VENV_TAG="${BASE_VENV}/ci-test:${VERSION_NODOT}-${CI_TEST_VENV_HASH}"

# Check existence (or set all to false if force-rebuild)
if [ "$FORCE_REBUILD" = "true" ]; then
    CI_BUILD_LIGHT_EXISTS=false
    CI_BUILD_EXISTS=false
    CI_TEST_LIGHT_EXISTS=false
    CI_TEST_EXISTS=false
    DEV_LIGHT_EXISTS=false
    DEV_EXISTS=false
    BASIC_DEV_EXISTS=false
    BASIC_TTNN_EXISTS=false
    MANYLINUX_EXISTS=false
    CI_BUILD_VENV_EXISTS=false
    CI_TEST_VENV_EXISTS=false
elif [ "$CHECK_EXISTS" = "true" ]; then
    # Check dev first. Its tag includes the main image, tool, and venv hash
    # chain, so a hit lets us avoid the most expensive fan-out on the common
    # path where the platform has already been built.
    TMPDIR_CHECK=$(mktemp -d)
    if manifest_exists "$DEV_TAG"; then
        DEV_EXISTS=true
        ( manifest_exists "$CI_BUILD_LIGHT_TAG" && echo true || echo false ) > "${TMPDIR_CHECK}/ci_build_light" &
        PID_CI_BUILD_LIGHT=$!
        ( manifest_exists "$CI_TEST_LIGHT_TAG" && echo true || echo false ) > "${TMPDIR_CHECK}/ci_test_light" &
        PID_CI_TEST_LIGHT=$!
        ( manifest_exists "$DEV_LIGHT_TAG" && echo true || echo false ) > "${TMPDIR_CHECK}/dev_light" &
        PID_DEV_LIGHT=$!

        # DEV_TAG is a canary: its content hash includes the ci_build and ci_test
        # layer hashes, so a DEV_TAG hit implies ci_build/ci_test were built from
        # identical layers in the same pipeline run — no need to probe their manifests.
        CI_BUILD_EXISTS=true
        CI_TEST_EXISTS=true
        CI_BUILD_VENV_EXISTS=unknown
        CI_TEST_VENV_EXISTS=unknown

        wait $PID_CI_BUILD_LIGHT; CI_BUILD_LIGHT_EXISTS=$(cat "${TMPDIR_CHECK}/ci_build_light")
        wait $PID_CI_TEST_LIGHT; CI_TEST_LIGHT_EXISTS=$(cat "${TMPDIR_CHECK}/ci_test_light")
        wait $PID_DEV_LIGHT; DEV_LIGHT_EXISTS=$(cat "${TMPDIR_CHECK}/dev_light")
    else
        DEV_EXISTS=false
        ( manifest_exists "$CI_BUILD_LIGHT_TAG" && echo true || echo false ) > "${TMPDIR_CHECK}/ci_build_light" &
        PID_CI_BUILD_LIGHT=$!
        ( manifest_exists "$CI_BUILD_TAG" && echo true || echo false ) > "${TMPDIR_CHECK}/ci_build" &
        PID_CI_BUILD=$!
        ( manifest_exists "$CI_TEST_LIGHT_TAG" && echo true || echo false ) > "${TMPDIR_CHECK}/ci_test_light" &
        PID_CI_TEST_LIGHT=$!
        ( manifest_exists "$CI_TEST_TAG" && echo true || echo false ) > "${TMPDIR_CHECK}/ci_test" &
        PID_CI_TEST=$!
        ( manifest_exists "$DEV_LIGHT_TAG" && echo true || echo false ) > "${TMPDIR_CHECK}/dev_light" &
        PID_DEV_LIGHT=$!

        wait $PID_CI_BUILD_LIGHT; CI_BUILD_LIGHT_EXISTS=$(cat "${TMPDIR_CHECK}/ci_build_light")
        wait $PID_CI_BUILD; CI_BUILD_EXISTS=$(cat "${TMPDIR_CHECK}/ci_build")
        wait $PID_CI_TEST_LIGHT; CI_TEST_LIGHT_EXISTS=$(cat "${TMPDIR_CHECK}/ci_test_light")
        wait $PID_CI_TEST; CI_TEST_EXISTS=$(cat "${TMPDIR_CHECK}/ci_test")
        wait $PID_DEV_LIGHT; DEV_LIGHT_EXISTS=$(cat "${TMPDIR_CHECK}/dev_light")

        CI_BUILD_VENV_EXISTS=unknown
        CI_TEST_VENV_EXISTS=unknown
        if [ "$CI_BUILD_EXISTS" != "true" ]; then
            ( manifest_exists "$CI_BUILD_VENV_TAG" && echo true || echo false ) > "${TMPDIR_CHECK}/ci_build_venv" &
            PID_CI_BUILD_VENV=$!
        fi
        if [ "$CI_TEST_EXISTS" != "true" ] || [ "$DEV_EXISTS" != "true" ]; then
            ( manifest_exists "$CI_TEST_VENV_TAG" && echo true || echo false ) > "${TMPDIR_CHECK}/ci_test_venv" &
            PID_CI_TEST_VENV=$!
        fi

        if [ "${PID_CI_BUILD_VENV:-}" ]; then
            wait $PID_CI_BUILD_VENV; CI_BUILD_VENV_EXISTS=$(cat "${TMPDIR_CHECK}/ci_build_venv")
        fi
        if [ "${PID_CI_TEST_VENV:-}" ]; then
            wait $PID_CI_TEST_VENV; CI_TEST_VENV_EXISTS=$(cat "${TMPDIR_CHECK}/ci_test_venv")
        fi
    fi

    ( manifest_exists "$BASIC_DEV_TAG" && echo true || echo false ) > "${TMPDIR_CHECK}/basic_dev" &
    PID_BASIC_DEV=$!
    ( manifest_exists "$BASIC_TTNN_TAG" && echo true || echo false ) > "${TMPDIR_CHECK}/basic_ttnn" &
    PID_BASIC_TTNN=$!
    ( manifest_exists "$MANYLINUX_TAG" && echo true || echo false ) > "${TMPDIR_CHECK}/manylinux" &
    PID_MANYLINUX=$!

    wait $PID_BASIC_DEV; BASIC_DEV_EXISTS=$(cat "${TMPDIR_CHECK}/basic_dev")
    wait $PID_BASIC_TTNN; BASIC_TTNN_EXISTS=$(cat "${TMPDIR_CHECK}/basic_ttnn")
    wait $PID_MANYLINUX; MANYLINUX_EXISTS=$(cat "${TMPDIR_CHECK}/manylinux")
    rm -rf "$TMPDIR_CHECK"
else
    CI_BUILD_LIGHT_EXISTS=unknown
    CI_BUILD_EXISTS=unknown
    CI_TEST_LIGHT_EXISTS=unknown
    CI_TEST_EXISTS=unknown
    DEV_LIGHT_EXISTS=unknown
    DEV_EXISTS=unknown
    BASIC_DEV_EXISTS=unknown
    BASIC_TTNN_EXISTS=unknown
    MANYLINUX_EXISTS=unknown
    CI_BUILD_VENV_EXISTS=unknown
    CI_TEST_VENV_EXISTS=unknown
fi

CI_BUILD_VENV_REQUIRED=false
CI_TEST_VENV_REQUIRED=false
if [ "$CI_BUILD_EXISTS" != "true" ]; then
    CI_BUILD_VENV_REQUIRED=true
fi
if [ "$CI_TEST_EXISTS" != "true" ] || [ "$DEV_EXISTS" != "true" ]; then
    CI_TEST_VENV_REQUIRED=true
fi

FINAL_IMAGES_MISSING=false
if [ "$CI_BUILD_LIGHT_EXISTS" != "true" ] ||
    [ "$CI_BUILD_EXISTS" != "true" ] ||
    [ "$CI_TEST_LIGHT_EXISTS" != "true" ] ||
    [ "$CI_TEST_EXISTS" != "true" ] ||
    [ "$DEV_LIGHT_EXISTS" != "true" ] ||
    [ "$DEV_EXISTS" != "true" ] ||
    [ "$BASIC_DEV_EXISTS" != "true" ] ||
    [ "$BASIC_TTNN_EXISTS" != "true" ] ||
    [ "$MANYLINUX_EXISTS" != "true" ]; then
    FINAL_IMAGES_MISSING=true
fi

VENVS_REQUIRED=false
if [ "$CI_BUILD_VENV_REQUIRED" = "true" ] || [ "$CI_TEST_VENV_REQUIRED" = "true" ]; then
    VENVS_REQUIRED=true
fi

# Output JSON
# Use --arg for existence fields since they may hold "unknown" (not valid JSON for --argjson)
jq -cn \
    --arg distro "$DISTRO" \
    --arg version "$VERSION" \
    --arg python_version "$PYTHON_VERSION" \
    --arg ci_build_light_tag "$CI_BUILD_LIGHT_TAG" \
    --arg ci_build_tag "$CI_BUILD_TAG" \
    --arg ci_test_light_tag "$CI_TEST_LIGHT_TAG" \
    --arg ci_test_tag "$CI_TEST_TAG" \
    --arg dev_light_tag "$DEV_LIGHT_TAG" \
    --arg dev_tag "$DEV_TAG" \
    --arg basic_dev_tag "$BASIC_DEV_TAG" \
    --arg basic_ttnn_tag "$BASIC_TTNN_TAG" \
    --arg manylinux_tag "$MANYLINUX_TAG" \
    --arg ci_build_venv_tag "$CI_BUILD_VENV_TAG" \
    --arg ci_test_venv_tag "$CI_TEST_VENV_TAG" \
    --arg ci_build_venv_required "$CI_BUILD_VENV_REQUIRED" \
    --arg ci_test_venv_required "$CI_TEST_VENV_REQUIRED" \
    --arg ci_build_light_exists "$CI_BUILD_LIGHT_EXISTS" \
    --arg ci_build_exists "$CI_BUILD_EXISTS" \
    --arg ci_test_light_exists "$CI_TEST_LIGHT_EXISTS" \
    --arg ci_test_exists "$CI_TEST_EXISTS" \
    --arg dev_light_exists "$DEV_LIGHT_EXISTS" \
    --arg dev_exists "$DEV_EXISTS" \
    --arg basic_dev_exists "$BASIC_DEV_EXISTS" \
    --arg basic_ttnn_exists "$BASIC_TTNN_EXISTS" \
    --arg manylinux_exists "$MANYLINUX_EXISTS" \
    --arg ci_build_venv_exists "$CI_BUILD_VENV_EXISTS" \
    --arg ci_test_venv_exists "$CI_TEST_VENV_EXISTS" \
    --arg final_images_missing "$FINAL_IMAGES_MISSING" \
    --arg venvs_required "$VENVS_REQUIRED" \
    '{
        distro: $distro,
        version: $version,
        python_version: $python_version,
        ci_build_light_tag: $ci_build_light_tag,
        ci_build_tag: $ci_build_tag,
        ci_test_light_tag: $ci_test_light_tag,
        ci_test_tag: $ci_test_tag,
        dev_light_tag: $dev_light_tag,
        dev_tag: $dev_tag,
        basic_dev_tag: $basic_dev_tag,
        basic_ttnn_tag: $basic_ttnn_tag,
        manylinux_tag: $manylinux_tag,
        ci_build_venv_tag: $ci_build_venv_tag,
        ci_test_venv_tag: $ci_test_venv_tag,
        ci_build_venv_required: $ci_build_venv_required,
        ci_test_venv_required: $ci_test_venv_required,
        ci_build_light_exists: $ci_build_light_exists,
        ci_build_exists: $ci_build_exists,
        ci_test_light_exists: $ci_test_light_exists,
        ci_test_exists: $ci_test_exists,
        dev_light_exists: $dev_light_exists,
        dev_exists: $dev_exists,
        basic_dev_exists: $basic_dev_exists,
        basic_ttnn_exists: $basic_ttnn_exists,
        manylinux_exists: $manylinux_exists,
        ci_build_venv_exists: $ci_build_venv_exists,
        ci_test_venv_exists: $ci_test_venv_exists,
        final_images_missing: $final_images_missing,
        venvs_required: $venvs_required
    }'
