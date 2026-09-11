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

# Artifacts persist across run attempts, so a re-run's analysis would otherwise
# re-download every test_reports_* zip the original attempt produced -- 56 of them on
# run 34609986416, none of them new. Attempts after the first therefore download only
# the reports created since that attempt started; the rest were already collected when
# the earlier attempt was analysed.
#
# First attempts (about two thirds of executions) keep the unconditional path, so the
# common case pays no extra listing.
download_artifacts() {
    local repo=$1
    local workflow_run_id=$2
    local attempt_number=$3

    if [[ -n "$attempt_number" && "$attempt_number" -gt 1 ]]; then
        download_artifacts_created_since_attempt_start "$repo" "$workflow_run_id" "$attempt_number"
        return
    fi

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

# Only the test_reports_* artifacts this attempt produced. One listing call replaces one
# download call per stale artifact, and a re-run in which no job uploaded a fresh report
# costs the listing alone.
download_artifacts_created_since_attempt_start() {
    local repo=$1
    local workflow_run_id=$2
    local attempt_number=$3

    local attempt_started
    if ! attempt_started=$(gh api "/repos/$repo/actions/runs/$workflow_run_id/attempts/$attempt_number" --jq '.run_started_at' 2>&1); then
        echo "[Warning] could not read attempt $attempt_number start time (${attempt_started:-no reason given}); downloading all test reports"
        download_artifacts "$repo" "$workflow_run_id" 1
        return
    fi
    echo "[info] attempt $attempt_number started $attempt_started; only test reports newer than that are this attempt's"

    local listing
    if ! listing=$(gh api --paginate "/repos/$repo/actions/runs/$workflow_run_id/artifacts" 2>&1); then
        echo "[Warning] could not list artifacts for workflow run $workflow_run_id: ${listing:-no reason given}"
        return
    fi

    # jq compares the ISO-8601 timestamps lexicographically, which is ordering-correct
    # for the UTC "...Z" form the API returns.
    local fresh
    fresh=$(printf '%s' "$listing" | jq -r --arg since "$attempt_started" \
        '[.artifacts[]? | select((.name | startswith("test_reports_")) and .created_at > $since) | .name] | unique | .[]')

    if [[ -z "$fresh" ]]; then
        echo "[info] attempt $attempt_number uploaded no new test reports; the earlier attempt's analysis collected the rest"
        return
    fi

    local -a fresh_names=()
    local -a name_args=()
    local name
    while IFS= read -r name; do
        [[ -z "$name" ]] && continue
        fresh_names+=("$name")
        name_args+=(--name "$name")
    done <<< "$fresh"
    echo "[info] downloading ${#fresh_names[@]} new test report artifact(s) for attempt $attempt_number"

    # Every name goes in one `gh run download`: each invocation does its own artifact
    # listing, so one call per artifact would cost two requests per report instead of one.
    #
    # The destination differs by count because gh's layout does. Several --name arguments
    # each get their own <name>/ directory, but a lone --name flattens its files straight
    # into -D -- and get_workflow_run_uuids_to_test_reports_paths_ globs for
    # test_reports_* directories, so a flattened download silently yields no test results
    # at all. Naming the directory ourselves in that case keeps the layout uniform.
    local dest="generated/cicd/$workflow_run_id/artifacts"
    if [[ "${#fresh_names[@]}" -eq 1 ]]; then
        dest="$dest/${fresh_names[0]}"
        mkdir -p "$dest"
    fi

    local download_error
    if ! download_error=$(gh run download --repo $repo -D "$dest" "${name_args[@]}" $workflow_run_id 2>&1 >/dev/null); then
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

# Fetch job logs an archive at a time instead of a job at a time.
#
# GitHub serves a run attempt's logs as a single zip:
#   GET /repos/{repo}/actions/runs/{id}/attempts/{n}/logs
# which replaces calling /actions/jobs/{job_id}/logs once per job -- 129 requests for a
# 129-job merge-gate run, against the repository's shared 15,000/hr GITHUB_TOKEN budget,
# with produce_data running hundreds of times an hour.
#
# An attempt's archive holds only the jobs that ran in THAT attempt, while the jobs list
# returns every job of the run. So a re-run's archive is nearly empty even though its
# jobs list is full: on run 34609986416 the attempt-3 archive had 2 logs and the
# attempt-1 archive had 85, for a jobs list of 97 both times. Fetching only the current
# attempt therefore left ~83 jobs to the per-job fallback and paid the old price for
# them, which is what re-runs were costing (~129 calls each, a third of executions).
#
# Hence the walk: extract attempt n, and while jobs that ran still have no log, extract
# n-1, n-2 ... 1. That is one request per attempt rather than one per job. Walking
# downward also means a job re-run in a later attempt keeps the later log, because the
# extractor never overwrites one it has already written.
#
# The jobs list cannot shortcut this. Its run_attempt field is simply the attempt that
# was queried -- attempt 1 and attempt 3 both return all 97 jobs, stamped 1 and 3
# respectively -- so which attempt a job actually ran in is only knowable by elimination.
#
# The unpacking is a Python helper rather than `unzip` because archive entries carry job
# and step names verbatim, emoji included: `unzip` can fail to create such a name, and
# having no tty to answer its "continue?" prompt it then aborts and leaves a silently
# partial extraction -- observed truncating at 46 of 85 job logs. See
# extract_job_logs_from_archive.py, which also explains the entry-name-to-job-id mapping.
#
# Whatever no archive covers -- including any name that does not resolve to exactly one
# job, which the helper leaves unmapped on purpose -- still falls through to the per-job
# path in download_logs_for_all_jobs.

# Jobs whose conclusion says they ran but which still have no log on disk. Pure local
# bookkeeping: no API calls, so it is cheap to consult between attempts.
count_ran_jobs_missing_logs() {
    local logs_dir=$1
    local jobs_data=$2

    local missing=0
    local job_id
    while IFS= read -r job_id; do
        [[ -z "$job_id" ]] && continue
        [[ -s "$logs_dir/$job_id.log" ]] || missing=$((missing + 1))
    done < <(printf '%s' "$jobs_data" | jq -r '.jobs[]? | select(.conclusion != "skipped" and .conclusion != null) | .id')
    echo "$missing"
}

# Extract one attempt's archive into <job_id>.log. Returns non-zero only if the archive
# could not be fetched or unpacked, which the caller treats as "try the next attempt"
# rather than as fatal.
extract_one_attempt_archive() {
    local repo=$1
    local workflow_run_id=$2
    local attempt=$3
    local jobs_file=$4
    local logs_dir=$5

    local script_dir
    script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    local tmp_dir
    tmp_dir=$(mktemp -d)
    local archive="$tmp_dir/logs.zip"

    # Same escape-sequence handling as the per-job path: gh >= 2.97.0 needs the flag,
    # older images in the fleet do not have it. Both errors are surfaced rather than
    # discarded -- a 403 here is the signal that the repository's hourly REST budget is
    # gone, and silencing it turns that into an unexplained fallback.
    local archive_error retry_error
    if ! archive_error=$(gh api --allow-escape-sequences "/repos/$repo/actions/runs/$workflow_run_id/attempts/$attempt/logs" 2>&1 >"$archive"); then
        if ! retry_error=$(gh api "/repos/$repo/actions/runs/$workflow_run_id/attempts/$attempt/logs" 2>&1 >"$archive"); then
            echo "[Warning] could not download the attempt $attempt log archive: ${retry_error:-no reason given}"
            echo "[Warning] first try (with --allow-escape-sequences) said: ${archive_error:-no reason given}"
            rm -rf "$tmp_dir"
            return 1
        fi
    fi

    if ! python3 "$script_dir/extract_job_logs_from_archive.py" \
            --archive "$archive" \
            --jobs-json "$jobs_file" \
            --logs-dir "$logs_dir"; then
        rm -rf "$tmp_dir"
        return 1
    fi

    rm -rf "$tmp_dir"
    return 0
}

download_logs_archives_walking_attempts() {
    local repo=$1
    local workflow_run_id=$2
    local attempt_number=$3
    local jobs_data=$4

    local logs_dir="generated/cicd/$workflow_run_id/logs"
    local tmp_dir
    tmp_dir=$(mktemp -d)
    local jobs_file="$tmp_dir/jobs.json"
    printf '%s' "$jobs_data" > "$jobs_file"

    local attempt
    for (( attempt = attempt_number; attempt >= 1; attempt-- )); do
        echo "[info] fetching attempt $attempt's logs as one archive"
        extract_one_attempt_archive "$repo" "$workflow_run_id" "$attempt" "$jobs_file" "$logs_dir" || true

        local missing
        missing=$(count_ran_jobs_missing_logs "$logs_dir" "$jobs_data")
        if [[ "$missing" -eq 0 ]]; then
            echo "[info] every job that ran has a log after $(( attempt_number - attempt + 1 )) archive request(s)"
            break
        fi
        echo "[info] $missing job(s) that ran still have no log"
    done

    rm -rf "$tmp_dir"
}

download_logs_for_all_jobs() {
    local repo=$1
    local workflow_run_id=$2
    local attempt_number=$3

    echo "[info] Downloading logs for workflow with id $workflow_run_id for attempt $attempt_number"

    # Get jobs using the pagination fallback function
    jobs_data=$(get_jobs_with_pagination_fallback "$repo" "$workflow_run_id" "$attempt_number")

    download_logs_archives_walking_attempts "$repo" "$workflow_run_id" "$attempt_number" "$jobs_data"

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
    download_artifacts "$repo" "$workflow_run_id" "$attempt_number"
    download_logs_for_all_jobs "$repo" "$workflow_run_id" "$attempt_number"
}

main "$@"
