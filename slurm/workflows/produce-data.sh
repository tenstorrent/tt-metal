#!/usr/bin/env bash
#SBATCH --job-name=produce-data
#SBATCH --partition=build
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#
# Data production pipeline: collect benchmark data, test results,
# and CI metadata for analytics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_config env

require_env PIPELINE_ID

DATA_OUTPUT="${ARTIFACT_DIR}/data"
mkdir -p "${DATA_OUTPUT}"

log_info "=== Data production starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"
log_info "  Output:   ${DATA_OUTPUT}"

REPORT_DIR="${ARTIFACT_DIR}/reports"
if [[ -d "${REPORT_DIR}" ]]; then
    log_info "Collecting test reports from ${REPORT_DIR}"

    python3 -c "
import json, os, sys, glob

reports_dir = '${REPORT_DIR}'
output = []

for job_dir in sorted(glob.glob(os.path.join(reports_dir, '*/'))):
    job_name = os.path.basename(os.path.normpath(job_dir))
    failed = os.path.exists(os.path.join(job_dir, 'FAILED'))
    output.append({
        'job_name': job_name,
        'status': 'failed' if failed else 'passed',
        'pipeline_id': '${PIPELINE_ID}',
        'git_sha': '${GIT_SHA}',
        'git_ref': '${GIT_REF}'
    })

with open('${DATA_OUTPUT}/pipeline_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f'Collected {len(output)} job results')
" 2>&1 || log_warn "Failed to collect test report data"
fi

cat > "${DATA_OUTPUT}/pipeline_metadata.json" <<EOJSON
{
  "pipeline_id": "${PIPELINE_ID}",
  "git_sha": "${GIT_SHA}",
  "git_ref": "${GIT_REF}",
  "submitted_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)"
}
EOJSON

log_info "=== Data production complete ==="
