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

    if gh api --paginate /repos/$repo/actions/runs/$workflow_run_id/artifacts | jq '.artifacts[] | .name' | grep -q "test_reports_"; then
        gh run download --repo $repo -D generated/cicd/$workflow_run_id/artifacts --pattern test_reports_* $workflow_run_id
    else
        echo "[Warning] Test reports not found for workflow run $workflow_run_id"
    fi
}

download_logs_for_all_jobs() {
    local repo=$1
    local workflow_run_id=$2
    local max_attempts=$3

    echo "[info] downloading logs for job with id $job_id for all attempts up to $max_attempts"
    for attempt_number in $(seq 1 $max_attempts); do
        echo "[Info] Downloading for attempt $attempt_number"

        gh api /repos/$repo/actions/runs/$workflow_run_id/attempts/$attempt_number/jobs --paginate | jq '.jobs[].id' | while read -r job_id; do
            echo "[info] download logs for job with id $job_id, attempt number $attempt_number"
            gh api /repos/$repo/actions/jobs/$job_id/logs > generated/cicd/$workflow_run_id/logs/$job_id.log
        done
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
