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

# Function to get jobs with pagination fallback
get_jobs_with_pagination_fallback() {
    local repo=$1
    local workflow_run_id=$2
    local attempt_number=$3

    local jobs_json_file="workflow_jobs.json"

    # Try the original --paginate approach first, fall back to manual pagination if it fails
    set +e  # Disable exit on error
    paginated_output=$(gh api /repos/$repo/actions/runs/$workflow_run_id/attempts/$attempt_number/jobs --paginate 2>&1)
    paginate_exit_code=$?
    set -e  # Re-enable exit on error

    if [ $paginate_exit_code -eq 0 ]; then
        echo "Successfully fetched jobs using --paginate" >&2
        echo "$paginated_output" | jq -s '{total_count: .[0].total_count, jobs: map(.jobs) | add}' > "$jobs_json_file"
        cat "$jobs_json_file"
    else
        echo "--paginate failed (exit code: $paginate_exit_code), falling back to manual pagination" >&2
        echo "Error output: $paginated_output" >&2

        # Manual pagination to avoid 502 errors with --paginate
        # Get first page to determine total count
        first_page=$(gh api "/repos/$repo/actions/runs/$workflow_run_id/attempts/$attempt_number/jobs?per_page=50&page=1")
        total_count=$(echo "$first_page" | jq -r '.total_count')
        echo "Total jobs: $total_count" >&2

        # Calculate total pages needed
        per_page=50
        total_pages=$(( (total_count + per_page - 1) / per_page ))
        echo "Total pages: $total_pages" >&2

        # Initialize with first page
        all_jobs=$(echo "$first_page" | jq -r '.jobs')

        # Fetch remaining pages if any
        for page in $(seq 2 $total_pages); do
            echo "Fetching page $page of $total_pages" >&2
            page_data=$(gh api "/repos/$repo/actions/runs/$workflow_run_id/attempts/$attempt_number/jobs?per_page=50&page=$page")
            page_jobs=$(echo "$page_data" | jq -r '.jobs')
            all_jobs=$(jq -s '.[0] + .[1]' <(echo "$all_jobs") <(echo "$page_jobs"))
            sleep 5
        done

        # Return the combined jobs in the expected format
        echo "$all_jobs" | jq -s '{total_count: '$total_count', jobs: .[0]}' > "$jobs_json_file"
        cat "$jobs_json_file"
    fi
}

download_logs_for_all_jobs() {
    local repo=$1
    local workflow_run_id=$2
    local attempt_number=$3

    echo "[info] Downloading logs for workflow with id $workflow_run_id for attempt $attempt_number"

    # Get jobs using the pagination fallback function
    jobs_data=$(get_jobs_with_pagination_fallback "$repo" "$workflow_run_id" "$attempt_number")

    # Process the jobs data
    echo "$jobs_data" | jq -c '.jobs[] | {id: .id, conclusion: .conclusion}' | while read -r job; do
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
