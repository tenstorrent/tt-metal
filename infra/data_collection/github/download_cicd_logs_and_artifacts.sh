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
    # `gh run download` lists the run's artifacts itself, so the separate
    # /artifacts listing that used to gate this call was a second (paginated) request
    # spent only to decide whether to make the first. A run with no test_reports_*
    # artifact makes this a no-op, which is the same outcome the gate produced.
    #
    # gh's own message is kept and reported: "no artifacts match" and "rate limit
    # exceeded" both land here, and collapsing the second into "not found" is how a
    # throttled fetch gets mistaken for a run that simply had no test reports.
    local download_error
    if ! download_error=$(gh run download --repo $repo -D generated/cicd/$workflow_run_id/artifacts --pattern 'test_reports_*' $workflow_run_id 2>&1 >/dev/null); then
        echo "[Warning] Test reports not downloaded for workflow run $workflow_run_id: ${download_error:-no reason given}"
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

# Fetch every job log for the attempt in ONE request.
#
# GitHub serves the whole attempt's logs as a single zip:
#   GET /repos/{repo}/actions/runs/{id}/attempts/{n}/logs
# The previous implementation instead called /actions/jobs/{job_id}/logs once per job,
# which made this script's cost scale with the size of the run being analyzed. On a
# 129-job merge-gate run that was 129 requests against the repository's shared
# 15,000/hr GITHUB_TOKEN budget, and produce_data runs ~173 times an hour.
#
# The unpacking is a Python helper rather than `unzip` because archive entries carry job
# and step names verbatim, emoji included: `unzip` can fail to create such a name, and
# having no tty to answer its "continue?" prompt it then aborts and leaves a silently
# partial extraction -- observed truncating at 46 of 85 job logs. See
# extract_job_logs_from_archive.py, which also explains the entry-name-to-job-id mapping.
#
# Any job the archive does not cover -- including any whose name does not resolve to
# exactly one job, which the helper leaves unmapped on purpose -- falls through to the
# per-job path in download_logs_for_all_jobs, so an ambiguous or missing entry costs one
# request instead of risking a log filed under the wrong job.
#
# Returns non-zero if the archive could not be fetched or unpacked at all.
download_logs_archive_for_attempt() {
    local repo=$1
    local workflow_run_id=$2
    local attempt_number=$3
    local jobs_data=$4

    local logs_dir="generated/cicd/$workflow_run_id/logs"
    local script_dir
    script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    local tmp_dir
    tmp_dir=$(mktemp -d)
    local archive="$tmp_dir/logs.zip"
    local jobs_file="$tmp_dir/jobs.json"

    printf '%s' "$jobs_data" > "$jobs_file"

    echo "[info] fetching the whole attempt's logs as one archive"
    # Same escape-sequence handling as the per-job path: gh >= 2.97.0 needs the flag,
    # older images in the fleet do not have it. Both errors are surfaced rather than
    # discarded -- a 403 here is the signal that the repository's hourly REST budget is
    # gone, and silencing it turns that into an unexplained fallback.
    local archive_error retry_error
    if ! archive_error=$(gh api --allow-escape-sequences "/repos/$repo/actions/runs/$workflow_run_id/attempts/$attempt_number/logs" 2>&1 >"$archive"); then
        if ! retry_error=$(gh api "/repos/$repo/actions/runs/$workflow_run_id/attempts/$attempt_number/logs" 2>&1 >"$archive"); then
            echo "[Warning] could not download the attempt log archive: ${retry_error:-no reason given}"
            echo "[Warning] first attempt (with --allow-escape-sequences) said: ${archive_error:-no reason given}"
            echo "[Warning] falling back to per-job downloads"
            rm -rf "$tmp_dir"
            return 1
        fi
    fi

    if ! python3 "$script_dir/extract_job_logs_from_archive.py" \
            --archive "$archive" \
            --jobs-json "$jobs_file" \
            --logs-dir "$logs_dir"; then
        echo "[Warning] falling back to per-job log downloads"
        rm -rf "$tmp_dir"
        return 1
    fi

    rm -rf "$tmp_dir"
    return 0
}

download_logs_for_all_jobs() {
    local repo=$1
    local workflow_run_id=$2
    local attempt_number=$3

    echo "[info] Downloading logs for workflow with id $workflow_run_id for attempt $attempt_number"

    # Get jobs using the pagination fallback function
    jobs_data=$(get_jobs_with_pagination_fallback "$repo" "$workflow_run_id" "$attempt_number")

    download_logs_archive_for_attempt "$repo" "$workflow_run_id" "$attempt_number" "$jobs_data" || true

    # Process the jobs data
    echo "$jobs_data" | jq -c '.jobs[] | {id: .id, conclusion: .conclusion}' | while read -r job; do
        job_id=$(echo "$job" | jq -r '.id')
        job_conclusion=$(echo "$job" | jq -r '.conclusion')
        # The attempt archive above normally supplied this already, so only jobs it did
        # not cover cost a request here. Skipped jobs and jobs that never reached a
        # conclusion have no log to serve, and asking anyway cost *two* requests each:
        # the 404 from the first call is a non-zero exit, so the "||" retry below fires
        # and 404s in turn. On the run this was measured against that was 44 of 129 jobs,
        # so 88 guaranteed-wasted requests.
        if [[ ! -s generated/cicd/$workflow_run_id/logs/$job_id.log ]] &&
           [[ "$job_conclusion" != "skipped" && "$job_conclusion" != "null" && -n "$job_conclusion" ]]; then
            echo "[info] download logs for job with id $job_id, attempt number $attempt_number"
            # https://github.com/tenstorrent/tt-metal/issues/12966
            # We bypass any log download that returned a non-zero exit code so the downloader doesn't crash midway.
            # williamly: We may want to check http status code for robustness in the future again but it may be costly in terms of api calls used.
            # We output escape sequences, gh cli >= 2.97.0 requires --allow-escape-sequences else it fails. Fall back to regular call for older images.
            gh api --allow-escape-sequences /repos/$repo/actions/jobs/$job_id/logs > generated/cicd/$workflow_run_id/logs/$job_id.log || \
                gh api /repos/$repo/actions/jobs/$job_id/logs > generated/cicd/$workflow_run_id/logs/$job_id.log || true
        fi

        # Download annotations for failed jobs only (failure reason).
        if [[ "$job_conclusion" == "failure" ]]; then
            echo "[info] downloading annotations for job $job_id (conclusion=$job_conclusion)"
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
