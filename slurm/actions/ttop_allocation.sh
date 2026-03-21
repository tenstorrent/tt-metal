#!/usr/bin/env bash
# ttop_allocation.sh - Unified TTOP wrapper for allocation lifecycle management.
# Equivalent to .github/actions/ttop-{create,delete,get-data}-allocation,
#               .github/actions/ttop-{create,delete}-environment,
#               .github/actions/ttop-configure-tt-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib ttop

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") <subcommand> [OPTIONS]

Manage TTOP allocations and environments for multi-node jobs.

Subcommands:
  create        Create an allocation and wait for it to reach Allocated state
  delete        Delete an allocation
  get-data      Retrieve node-to-rank mapping as JSON
  create-env    Create SSH environment on allocated nodes
  delete-env    Delete SSH environment
  configure     Generate hostfile, rankfile, and rank_bindings.yaml
  cleanup       Delete environment then allocation (safe for traps)

Options (apply to relevant subcommands):
  --name NAME           Allocation name (required for all subcommands)
  --spec FILE           Spec YAML file (required for create; optional for create-env)
  --timeout SECS        Wait timeout in seconds (default: 300)
  --output-dir DIR      Output directory for get-data and configure (default: .)
  --image IMAGE         Docker image for create-env
  -h, --help            Show this help message

Environment:
  TTOP_KUBECONFIG       Path to kubeconfig (default: ~/.kube/config)
  TTOP_NAMESPACE        Kubernetes namespace (default: ttop)
  TTOP_WAIT_TIMEOUT     Kubectl wait timeout (default: 999m)
  TT_CARDS_PER_HOST     Cards per host for environment (default: 8)
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Subcommand
# ---------------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
    log_error "Subcommand required"
    usage 1
fi

SUBCOMMAND="$1"; shift

case "${SUBCOMMAND}" in
    create|delete|get-data|create-env|delete-env|configure|cleanup) ;;
    -h|--help) usage 0 ;;
    *) log_error "Unknown subcommand: ${SUBCOMMAND}"; usage 1 ;;
esac

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

NAME=""
SPEC_FILE=""
TIMEOUT=300
OUTPUT_DIR="."
IMAGE=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)       NAME="$2"; shift 2 ;;
        --spec)       SPEC_FILE="$2"; shift 2 ;;
        --timeout)    TIMEOUT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --image)      IMAGE="$2"; shift 2 ;;
        -h|--help)    usage 0 ;;
        *)            log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ -z "${NAME}" ]]; then
    log_error "Allocation name required (--name)"
    usage 1
fi

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

case "${SUBCOMMAND}" in
    create)
        if [[ -z "${SPEC_FILE}" ]]; then
            log_fatal "Spec file required for create (--spec)"
        fi
        ttop_create_allocation "${NAME}" "${SPEC_FILE}" "${TIMEOUT}"
        ;;
    delete)
        ttop_delete_allocation "${NAME}"
        ;;
    get-data)
        ttop_get_allocation_data "${NAME}" "${OUTPUT_DIR}"
        ;;
    create-env)
        ttop_create_environment "${NAME}" "${IMAGE}" "${SPEC_FILE}"
        ;;
    delete-env)
        ttop_delete_environment "${NAME}"
        ;;
    configure)
        ttop_configure_tt_run "${NAME}" "${OUTPUT_DIR}"
        ;;
    cleanup)
        ttop_cleanup "${NAME}"
        ;;
esac
