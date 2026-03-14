#!/usr/bin/env bash
# upload_artifact.sh - Copy test reports to shared artifact storage on Weka.
# Equivalent to .github/actions/upload-artifact-with-job-uuid/action.yml
#
# Usage: upload_artifact.sh [--path DIR] [--name NAME]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts

REPORT_PATH=""
ARTIFACT_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --path) REPORT_PATH="$2"; shift 2 ;;
        --name) ARTIFACT_NAME="$2"; shift 2 ;;
        *)      log_warn "Unknown option: $1"; shift ;;
    esac
done

require_env PIPELINE_ID

JOB_NAME="${ARTIFACT_NAME:-$(get_job_name)}"
REPORT_PATH="${REPORT_PATH:-generated/test_reports}"

if [[ ! -d "${REPORT_PATH}" ]]; then
    log_warn "Report directory does not exist: ${REPORT_PATH}"
    exit 0
fi

log_info "Uploading artifacts: ${REPORT_PATH} -> reports/${JOB_NAME}/"
stage_test_report "${PIPELINE_ID}" "${JOB_NAME}" "${REPORT_PATH}"
log_info "Artifacts uploaded for job: ${JOB_NAME}"
