#!/usr/bin/env bash
# artifacts.sh - Artifact management using shared Weka/NFS storage
# Replaces GitHub Actions artifact upload/download with direct filesystem operations.

set -euo pipefail

# Guard against double-sourcing
[[ -n "${_SLURM_CI_ARTIFACTS_SH:-}" ]] && return 0
_SLURM_CI_ARTIFACTS_SH=1

SLURM_CI_LIB_DIR="${SLURM_CI_LIB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SLURM_CI_LIB_DIR}/common.sh"

ARTIFACT_BASE="${SLURM_CI_ARTIFACT_BASE}"

# ---------------------------------------------------------------------------
# get_artifact_dir - Return the artifact directory for a pipeline
# ---------------------------------------------------------------------------
# Usage: get_artifact_dir <pipeline_id>
get_artifact_dir() {
    local pipeline_id="${1:?pipeline_id required}"
    echo "${ARTIFACT_BASE}/${pipeline_id}"
}

# ---------------------------------------------------------------------------
# stage_build_artifact - Copy build tarball to shared storage
# ---------------------------------------------------------------------------
# Usage: stage_build_artifact <pipeline_id> <source_path>
#   source_path: directory containing ttm_any.tar.zst
stage_build_artifact() {
    local pipeline_id="${1:?pipeline_id required}"
    local source_path="${2:?source_path required}"
    local dest_dir
    dest_dir="$(get_artifact_dir "$pipeline_id")/build"

    local tarball="${source_path}/ttm_any.tar.zst"
    if [[ ! -f "$tarball" ]]; then
        log_error "Build tarball not found: ${tarball}"
        return 1
    fi

    mkdir -p "$dest_dir"
    log_info "Staging build artifact: ${tarball} -> ${dest_dir}/"
    cp "$tarball" "$dest_dir/"
    log_info "Build artifact staged ($(du -h "$dest_dir/ttm_any.tar.zst" | cut -f1))"
}

# ---------------------------------------------------------------------------
# fetch_build_artifact - Copy and extract build tarball from shared storage
# ---------------------------------------------------------------------------
# Usage: fetch_build_artifact <pipeline_id> <dest_path>
fetch_build_artifact() {
    local pipeline_id="${1:?pipeline_id required}"
    local dest_path="${2:?dest_path required}"
    local artifact_dir
    artifact_dir="$(get_artifact_dir "$pipeline_id")/build"

    local tarball="${artifact_dir}/ttm_any.tar.zst"
    if [[ ! -f "$tarball" ]]; then
        log_error "Build artifact not found: ${tarball}"
        return 1
    fi

    mkdir -p "$dest_path"
    log_info "Extracting build artifact: ${tarball} -> ${dest_path}/"
    tar --zstd -xf "$tarball" -C "$dest_path"
    log_info "Build artifact extracted to ${dest_path}"
}

# ---------------------------------------------------------------------------
# stage_wheel - Copy Python wheel to shared storage
# ---------------------------------------------------------------------------
# Usage: stage_wheel <pipeline_id> <wheel_path>
stage_wheel() {
    local pipeline_id="${1:?pipeline_id required}"
    local wheel_path="${2:?wheel_path required}"
    local dest_dir
    dest_dir="$(get_artifact_dir "$pipeline_id")/build/tt_metal_wheels"

    if [[ ! -f "$wheel_path" ]]; then
        log_error "Wheel not found: ${wheel_path}"
        return 1
    fi

    mkdir -p "$dest_dir"
    log_info "Staging wheel: ${wheel_path} -> ${dest_dir}/"
    cp "$wheel_path" "$dest_dir/"
    log_info "Wheel staged: $(basename "$wheel_path")"
}

# ---------------------------------------------------------------------------
# fetch_wheel - Find and install wheel from shared storage
# ---------------------------------------------------------------------------
# Usage: fetch_wheel <pipeline_id>
fetch_wheel() {
    local pipeline_id="${1:?pipeline_id required}"
    local wheel_dir
    wheel_dir="$(get_artifact_dir "$pipeline_id")/build/tt_metal_wheels"

    if [[ ! -d "$wheel_dir" ]]; then
        log_error "Wheel directory not found: ${wheel_dir}"
        return 1
    fi

    require_cmd uv

    local -a wheels
    mapfile -t wheels < <(find "$wheel_dir" -maxdepth 1 -name '*.whl' -type f)

    if [[ ${#wheels[@]} -eq 0 ]]; then
        log_error "No .whl files found in ${wheel_dir}"
        return 1
    fi
    if [[ ${#wheels[@]} -gt 1 ]]; then
        log_error "Multiple .whl files found in ${wheel_dir}: ${wheels[*]}"
        return 1
    fi

    log_info "Installing wheel: $(basename "${wheels[0]}")"
    uv pip install "${wheels[0]}"
    log_info "Wheel installed successfully"
}

# ---------------------------------------------------------------------------
# stage_test_report - Copy test reports to shared storage
# ---------------------------------------------------------------------------
# Usage: stage_test_report <pipeline_id> <job_name> <report_dir>
stage_test_report() {
    local pipeline_id="${1:?pipeline_id required}"
    local job_name="${2:?job_name required}"
    local report_dir="${3:?report_dir required}"
    local dest_dir
    dest_dir="$(get_artifact_dir "$pipeline_id")/reports/${job_name}"

    if [[ ! -d "$report_dir" ]]; then
        log_warn "Report directory does not exist: ${report_dir}"
        return 0
    fi

    mkdir -p "$dest_dir"
    log_info "Staging test reports: ${report_dir} -> ${dest_dir}/"
    cp -r "$report_dir"/. "$dest_dir/"
    log_info "Test reports staged for job '${job_name}'"
}

# ---------------------------------------------------------------------------
# stage_docker_tags - Store Docker image tags to shared storage
# ---------------------------------------------------------------------------
# Usage: stage_docker_tags <pipeline_id> <tags_file>
stage_docker_tags() {
    local pipeline_id="${1:?pipeline_id required}"
    local tags_file="${2:?tags_file required}"
    local dest_dir
    dest_dir="$(get_artifact_dir "$pipeline_id")/docker"

    if [[ ! -f "$tags_file" ]]; then
        log_error "Tags file not found: ${tags_file}"
        return 1
    fi

    mkdir -p "$dest_dir"
    log_info "Staging Docker tags: ${tags_file} -> ${dest_dir}/image_tags.env"
    cp "$tags_file" "$dest_dir/image_tags.env"
}

# ---------------------------------------------------------------------------
# fetch_docker_tags - Source Docker image tags from shared storage
# ---------------------------------------------------------------------------
# Usage: fetch_docker_tags <pipeline_id>
fetch_docker_tags() {
    local pipeline_id="${1:?pipeline_id required}"
    local tags_file
    tags_file="$(get_artifact_dir "$pipeline_id")/docker/image_tags.env"

    if [[ ! -f "$tags_file" ]]; then
        log_error "Docker tags file not found: ${tags_file}"
        return 1
    fi

    log_info "Loading Docker tags from ${tags_file}"
    # shellcheck disable=SC1090
    source "$tags_file"
}

# ---------------------------------------------------------------------------
# cleanup_artifacts - Remove all artifacts for a pipeline
# ---------------------------------------------------------------------------
# Usage: cleanup_artifacts <pipeline_id>
cleanup_artifacts() {
    local pipeline_id="${1:?pipeline_id required}"
    local artifact_dir
    artifact_dir="$(get_artifact_dir "$pipeline_id")"

    if [[ ! -d "$artifact_dir" ]]; then
        log_debug "No artifacts to clean up for pipeline ${pipeline_id}"
        return 0
    fi

    log_info "Removing artifacts for pipeline ${pipeline_id}: ${artifact_dir}"
    rm -rf "$artifact_dir"
    log_info "Artifacts cleaned up"
}

# ---------------------------------------------------------------------------
# cleanup_old_artifacts - Remove artifacts older than N days
# ---------------------------------------------------------------------------
# Usage: cleanup_old_artifacts <days>
cleanup_old_artifacts() {
    local days="${1:?days required}"

    log_info "Cleaning up artifacts older than ${days} days in ${ARTIFACT_BASE}"
    local count=0
    while IFS= read -r -d '' dir; do
        log_info "Removing stale artifact dir: ${dir}"
        rm -rf "$dir"
        count=$((count + 1))
    done < <(find "$ARTIFACT_BASE" -mindepth 1 -maxdepth 1 -type d -mtime +"$days" -print0)

    log_info "Removed ${count} stale artifact directories"
}
