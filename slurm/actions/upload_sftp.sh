#!/usr/bin/env bash
# upload_sftp.sh - Upload data to an SFTP server using predefined batchfiles.
# Equivalent to .github/actions/upload-data-via-sftp/action.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") --type TYPE [OPTIONS]

Upload data via SFTP.  The --type flag selects a predefined batchfile that
maps to the corresponding GitHub Actions upload-data-via-sftp batchfile.

Required:
  --type TYPE           Data type: benchmark, cicd, fabric_bw, fabric_lat, or optest

Options:
  --host HOST           SFTP hostname (or set SFTP_HOST)
  --user USER           SFTP username (or set SFTP_USER)
  --key-file FILE       Path to SSH private key file (or set SFTP_KEYFILE)
  --batchfile FILE      Override the auto-selected batchfile
  --work-dir DIR        Working directory for SFTP commands (default: .)
  -h, --help            Show this help message

Environment:
  SFTP_HOST             Fallback for --host
  SFTP_USER             Fallback for --user
  SFTP_KEYFILE          Fallback for --key-file
  SSH_PRIVATE_KEY       Raw SSH key content (written to temp file if SFTP_KEYFILE is unset)
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Type-to-batchfile mapping
# ---------------------------------------------------------------------------

BATCHFILE_DIR="${REPO_ROOT}/.github/actions/upload-data-via-sftp"

resolve_batchfile() {
    local data_type="$1"
    case "${data_type}" in
        benchmark)   echo "${BATCHFILE_DIR}/benchmark_data_batchfile.txt" ;;
        cicd)        echo "${BATCHFILE_DIR}/cicd_data_batchfile.txt" ;;
        fabric_bw)   echo "${BATCHFILE_DIR}/fabric_bandwidth_benchmark_data_batchfile.txt" ;;
        fabric_lat)  echo "${BATCHFILE_DIR}/fabric_latency_benchmark_data_batchfile.txt" ;;
        optest)      echo "${BATCHFILE_DIR}/optest_batchfile.txt" ;;
        *)           echo "" ;;
    esac
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATA_TYPE=""
SFTP_HOST="${SFTP_HOST:-}"
SFTP_USER="${SFTP_USER:-}"
SFTP_KEYFILE="${SFTP_KEYFILE:-}"
BATCHFILE=""
WORK_DIR=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)      DATA_TYPE="$2"; shift 2 ;;
        --host)      SFTP_HOST="$2"; shift 2 ;;
        --user)      SFTP_USER="$2"; shift 2 ;;
        --key-file)  SFTP_KEYFILE="$2"; shift 2 ;;
        --batchfile) BATCHFILE="$2"; shift 2 ;;
        --work-dir)  WORK_DIR="$2"; shift 2 ;;
        -h|--help)   usage 0 ;;
        *)           log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ -z "${DATA_TYPE}" ]]; then
    log_error "--type is required"
    usage 1
fi

case "${DATA_TYPE}" in
    benchmark|cicd|fabric_bw|fabric_lat|optest) ;;
    *) log_error "Invalid type '${DATA_TYPE}'; expected benchmark|cicd|fabric_bw|fabric_lat|optest"; usage 1 ;;
esac

if [[ -z "${SFTP_HOST}" ]]; then
    log_fatal "SFTP host required (--host or SFTP_HOST)"
fi
if [[ -z "${SFTP_USER}" ]]; then
    log_fatal "SFTP user required (--user or SFTP_USER)"
fi

# Resolve batchfile
if [[ -z "${BATCHFILE}" ]]; then
    BATCHFILE="$(resolve_batchfile "${DATA_TYPE}")"
fi
if [[ -z "${BATCHFILE}" || ! -f "${BATCHFILE}" ]]; then
    log_fatal "Batchfile not found: ${BATCHFILE:-<none>}"
fi

require_cmd sftp

# ---------------------------------------------------------------------------
# Key file setup
# ---------------------------------------------------------------------------

TEMP_KEY=""

if [[ -z "${SFTP_KEYFILE}" && -n "${SSH_PRIVATE_KEY:-}" ]]; then
    TEMP_KEY="$(mktemp)"
    printf '%s\n' "${SSH_PRIVATE_KEY}" > "${TEMP_KEY}"
    chmod 600 "${TEMP_KEY}"
    SFTP_KEYFILE="${TEMP_KEY}"
    register_cleanup "rm -f '${TEMP_KEY}'"
fi

if [[ -n "${SFTP_KEYFILE}" && ! -f "${SFTP_KEYFILE}" ]]; then
    log_fatal "SSH key file not found: ${SFTP_KEYFILE}"
fi

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

if [[ -n "${WORK_DIR}" ]]; then
    log_info "Changing to work directory: ${WORK_DIR}"
    cd "${WORK_DIR}"
fi

SFTP_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes)
[[ -n "${SFTP_KEYFILE}" ]] && SFTP_OPTS+=(-i "${SFTP_KEYFILE}")

log_info "Uploading ${DATA_TYPE} data via SFTP"
log_info "  Host:      ${SFTP_USER}@${SFTP_HOST}"
log_info "  Batchfile: ${BATCHFILE}"

sftp "${SFTP_OPTS[@]}" -b "${BATCHFILE}" "${SFTP_USER}@${SFTP_HOST}" || {
    rc=$?
    log_error "SFTP upload failed (rc=${rc})"
    exit "${rc}"
}

log_info "SFTP upload complete (${DATA_TYPE})"
