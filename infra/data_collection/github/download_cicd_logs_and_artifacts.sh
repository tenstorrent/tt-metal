#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    # `gh run download --pattern` lists the run's artifacts itself, so the pre-flight
    # `gh api .../artifacts` call this used to make was a second listing of the same data,
    # costing 2 x ceil(artifacts / 100) API calls where 1 x is enough. `gh run download`
    # exits non-zero when no artifact matches the pattern, which is a normal outcome here
    # rather than an error, so it is handled instead of tripping `set -e`.
    # The pattern is quoted so the shell cannot glob-expand it against the working dir.
    if ! gh run download --repo "$repo" -D "generated/cicd/$workflow_run_id/artifacts" --pattern 'test_reports_*' "$workflow_run_id"; then
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

# Fetch every job log for the run in a single API call.
#
# GET /repos/{owner}/{repo}/actions/runs/{run_id}/attempts/{n}/logs returns a ZIP holding
# one entry per job that produced a log. That is 1 API call for the whole run, where
# downloading each job log individually costs 1 call per job -- by far the largest part of
# this workflow's API budget.
#
# Emits the ids of any jobs the archive did not supply, one per line, so the caller can
# download just those individually. If the archive cannot be fetched at all it emits every
# runnable job id, so any failure here degrades to exactly the previous behaviour.
fetch_job_logs_via_run_archive() {
    local repo=$1
    local workflow_run_id=$2
    local attempt_number=$3
    local jobs_json_file=$4

    local logs_dir="generated/cicd/$workflow_run_id/logs"
    local zip_path="generated/cicd/$workflow_run_id/run_logs.zip"

    if ! gh api "/repos/$repo/actions/runs/$workflow_run_id/attempts/$attempt_number/logs" > "$zip_path" 2>/dev/null; then
        echo "[Warning] Could not fetch run log archive; falling back to per-job log downloads" >&2
        rm -f "$zip_path"
        jq -r '.jobs[] | select(.conclusion != null and .conclusion != "skipped") | .id' "$jobs_json_file"
        return 0
    fi

    python3 "$SCRIPT_DIR/split_run_log_archive.py" \
        --archive "$zip_path" \
        --jobs-json "$jobs_json_file" \
        --out-dir "$logs_dir"

    rm -f "$zip_path"
}

download_logs_for_all_jobs() {
    local repo=$1
    local workflow_run_id=$2
    local attempt_number=$3

    echo "[info] Downloading logs for workflow with id $workflow_run_id for attempt $attempt_number"

    # Get jobs using the pagination fallback function
    jobs_data=$(get_jobs_with_pagination_fallback "$repo" "$workflow_run_id" "$attempt_number")

    # One API call for every job log in the run. Anything it could not supply is listed
    # here and downloaded individually in the loop below.
    jobs_needing_individual_download=$(fetch_job_logs_via_run_archive \
        "$repo" "$workflow_run_id" "$attempt_number" "workflow_jobs.json")

    # Process the jobs data
    # is_civ2 is true when any runs-on label starts with tt-ubuntu- (CIv2 runners)
    #
    # Skipped jobs (and jobs with no conclusion) never ran, so they have no log to fetch:
    # the API returns 404 and the `|| true` below quietly wrote the error body into
    # <job_id>.log. Across a 40-run sample, 33% of all analysed jobs were `skipped`, so
    # filtering them out removes a third of the log downloads and loses no data.
    echo "$jobs_data" | jq -c '.jobs[] | select(.conclusion != null and .conclusion != "skipped") | {id: .id, conclusion: .conclusion, is_civ2: ([.labels[]? | select(startswith("tt-ubuntu-"))] | length > 0)}' | while read -r job; do
        job_id=$(echo "$job" | jq -r '.id')
        job_conclusion=$(echo "$job" | jq -r '.conclusion')
        is_civ2=$(echo "$job" | jq -r '.is_civ2')
        # The run log archive above already supplied almost every job log. Only jobs it
        # could not account for are fetched individually here.
        if printf '%s\n' "$jobs_needing_individual_download" | grep -qxF "$job_id"; then
            echo "[info] download logs for job with id $job_id, attempt number $attempt_number"
            # https://github.com/tenstorrent/tt-metal/issues/12966
            # We bypass any log download that returned a non-zero exit code so the downloader doesn't crash midway.
            # williamly: We may want to check http status code for robustness in the future again but it may be costly in terms of api calls used.
            # We output escape sequences, gh cli >= 2.97.0 requires --allow-escape-sequences else it fails. Fall back to regular call for older images.
            gh api --allow-escape-sequences /repos/$repo/actions/jobs/$job_id/logs > generated/cicd/$workflow_run_id/logs/$job_id.log || \
                gh api /repos/$repo/actions/jobs/$job_id/logs > generated/cicd/$workflow_run_id/logs/$job_id.log || true
        fi

        # Download annotations for failed jobs only, to recover the failure reason.
        #
        # CIv2 runners also emit node-name and card-serial notice annotations at job start
        # (see tenstorrent/github-ci-infra#1408), but the runner prints those same strings as
        # plain stdout in the job log's "Set up runner" step, and we download that log anyway.
        # utils.get_job_row_from_github_job already falls back to
        # workflows.get_civ2_node_name_and_serial_from_job_log when no annotations file is
        # present, so fetching annotations for every CIv2 job re-fetched data we already had.
        # In a 40-run sample, 84% of annotation calls were non-failure CIv2 jobs.
        if [[ "$job_conclusion" == "failure" ]]; then
            echo "[info] downloading annotations for job $job_id (conclusion=$job_conclusion, civ2=$is_civ2)"
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
