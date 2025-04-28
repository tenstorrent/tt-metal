#!/bin/bash

set -eo pipefail

# Default argument values
default_repo="tenstorrent/tt-metal"

set_up_dirs() {
    local workflow_run_id=$1
    mkdir -p generated/cicd
    rm -rf generated/cicd/$workflow_run_id
    mkdir -p generated/cicd/$workflow_run_id/artifacts
    mkdir -p generated/cicd/$workflow_run_id/logs
}

download_artifacts() {
    local repo=$1
    local workflow_run_id=$2

    echo "[info] Downloading test reports for workflow run $workflow_run_id"
    api_output=$(gh api --paginate /repos/$repo/actions/runs/$workflow_run_id/artifacts | jq -r '.artifacts[] | .name')
    if echo "$api_output" | grep -q "test_reports_"; then
        gh run download --repo $repo -D generated/cicd/$workflow_run_id/artifacts --pattern test_reports_* $workflow_run_id
    else
        echo "[Warning] Test reports not found for workflow run $workflow_run_id"
    fi
}

download_logs_for_all_jobs() {
    local repo=$1
    local workflow_run_id=$2
    local attempt_number=$3

    echo "[info] Downloading logs for workflow with id $workflow_run_id for attempt $attempt_number"
    gh api /repos/$repo/actions/runs/$workflow_run_id/attempts/$attempt_number/jobs --paginate | jq -c '.jobs[] | {id: .id, conclusion: .conclusion}' | while read -r job; do
        job_id=$(echo "$job" | jq -r '.id')
        job_conclusion=$(echo "$job" | jq -r '.conclusion')
        echo "[info] download logs for job with id $job_id, attempt number $attempt_number"
        # https://github.com/tenstorrent/tt-metal/issues/12966
        # We bypass any log download that returned a non-zero exit code so the downloader doesn't crash midway.
        # williamly: We may want to check http status code for robustness in the future again but it may be costly in terms of api calls used.
        gh api /repos/$repo/actions/jobs/$job_id/logs > generated/cicd/$workflow_run_id/logs/$job_id.log || true

        # Only download annotations for failed jobs
        if [[ "$job_conclusion" == "failure" ]]; then
            echo "[info] downloading annotations for failed job $job_id"
            gh api /repos/$repo/check-runs/$job_id/annotations > generated/cicd/$workflow_run_id/logs/${job_id}_annotations.json
        fi
    done

}

main() {
    # Parse the arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --workflow-run-id)
                workflow_run_id=$2
                shift
                ;;
            --attempt-number)
                attempt_number=$2
                shift
                ;;
            --repo)
                repo=$2
                shift
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
        shift
    done

    # Set default values if arguments are not provided
    repo=${repo:-$default_repo}

    if [[ -z "$workflow_run_id" ]]; then
        echo "workflow_run_id is empty"
        exit 1
    fi

    if [[ -z "$attempt_number" ]]; then
        echo "attempt_number is empty"
        exit 1
    fi

    set_up_dirs "$workflow_run_id"
    download_artifacts "$repo" "$workflow_run_id"
    download_logs_for_all_jobs "$repo" "$workflow_run_id" "$attempt_number"
}

main "$@"
