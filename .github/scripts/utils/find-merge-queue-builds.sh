#!/usr/bin/env bash
# Find the merge-queue run that already built a commit, so that the push run for
# the same commit can take those builds and not build the commit again.
#
# Usage: find-merge-queue-builds.sh <workflow-file> <sha> <key>=<caller-job-id> [...]
#
# For each <key>, the script writes "<key>=<run-id>" to $GITHUB_OUTPUT (stdout
# when GITHUB_OUTPUT is not set). <run-id> is the newest successful merge_group
# run of <workflow-file> for <sha> in which the build jobs of <caller-job-id>
# ran and passed. When no run agrees, the script writes "<key>=" and the caller
# builds as before. That is the path for an admin merge or a direct push.
#
# The script does not fail. An API error gives empty outputs, so a lookup
# problem costs a build and never removes a build.
#
# Env: GH_TOKEN, GITHUB_REPOSITORY, and WAIT_SECONDS (default 300): how long to
# wait for a queue run of <sha> that is not complete yet.
set -uo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <workflow-file> <sha> <key>=<caller-job-id> [...]" >&2
    exit 2
fi
WORKFLOW=$1
SHA=$2
shift 2
REPO=${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is not set}
WAIT_SECONDS=${WAIT_SECONDS:-300}
OUT=${GITHUB_OUTPUT:-/dev/stdout}
SUMMARY=${GITHUB_STEP_SUMMARY:-/dev/null}

KEYS=()
JOB_IDS=()
RESULTS=()
for pair in "$@"; do
    KEYS+=("${pair%%=*}")
    JOB_IDS+=("${pair#*=}")
    RESULTS+=("")
done

emit() {
    local i
    for i in "${!KEYS[@]}"; do
        echo "${KEYS[$i]}=${RESULTS[$i]}" >> "$OUT"
        if [ -n "${RESULTS[$i]}" ]; then
            echo "- \`${KEYS[$i]}\`: reuse the builds of [run ${RESULTS[$i]}](https://github.com/${REPO}/actions/runs/${RESULTS[$i]})" >> "$SUMMARY"
        else
            echo "- \`${KEYS[$i]}\`: no queue build to reuse; build here" >> "$SUMMARY"
        fi
    done
}

echo "### Merge-queue builds for ${SHA}" >> "$SUMMARY"

if ! printf '%s' "$SHA" | grep -qE '^[0-9a-f]{40}$'; then
    echo "::warning::'${SHA}' is not a full 40-character commit sha; build here."
    emit
    exit 0
fi

# Wait only when no completed run is there yet. The 'destroyed' run of the queue
# entry starts at the merge, so it can still run when the push run starts; do not
# wait for it when the 'checks_requested' run is already complete.
deadline=$((SECONDS + WAIT_SECONDS))
while :; do
    if ! RUNS=$(gh api "repos/${REPO}/actions/workflows/${WORKFLOW}/runs?head_sha=${SHA}&event=merge_group&per_page=100" \
            --jq '.workflow_runs[] | "\(.id) \(.status) \(.conclusion)"'); then
        echo "::warning::Could not list the ${WORKFLOW} merge_group runs for ${SHA}; build here."
        emit
        exit 0
    fi
    echo "merge_group runs of ${WORKFLOW} for ${SHA} (id status conclusion):"
    echo "${RUNS:-<none>}"
    CANDIDATES=$(awk '$2 == "completed" && $3 == "success" {print $1}' <<< "$RUNS")
    PENDING=$(awk 'NF && $2 != "completed" {print $1}' <<< "$RUNS")
    if [ -n "$CANDIDATES" ] || [ -z "$PENDING" ] || [ "$SECONDS" -ge "$deadline" ]; then
        break
    fi
    echo "No completed run yet; wait for: ${PENDING//$'\n'/ }"
    sleep 20
done

# The API lists the newest run first.
for run in $CANDIDATES; do
    if ! JOBS=$(gh api --paginate "repos/${REPO}/actions/runs/${run}/jobs?filter=latest&per_page=100" \
            --jq '.jobs[] | "\(.conclusion)\t\(.name)"'); then
        echo "::warning::Could not list the jobs of run ${run}; skip it."
        continue
    fi
    for i in "${!KEYS[@]}"; do
        [ -n "${RESULTS[$i]}" ] && continue
        key=${KEYS[$i]}
        job=${JOB_IDS[$i]}
        # A job of a called workflow is named "<caller-job-id> / <job name>".
        MATCH=$(awk -F'\t' -v p="${job} / " 'index($2, p) == 1' <<< "$JOBS")
        # The compile job of build-artifact.yaml is "<icon> Build <type> <distro> <version>".
        BUILT=$(awk -F'\t' '$1 == "success" && index($2, " Build ") && !index($2, " Build wheel")' <<< "$MATCH" | grep -c . || true)
        BAD=$(awk -F'\t' '$1 != "success" && $1 != "skipped"' <<< "$MATCH" | grep -c . || true)
        if [ "$BUILT" -ge 1 ] && [ "$BAD" -eq 0 ]; then
            echo "${key}: run ${run} built '${job}' (${BUILT} build jobs passed)."
            RESULTS[$i]=$run
        else
            echo "${key}: run ${run} did not build '${job}' (${BUILT} build jobs passed, ${BAD} jobs did not pass)."
        fi
    done
done

emit
