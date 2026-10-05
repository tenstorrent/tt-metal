#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Run the perf gate's compare jobs again for a PR head, without measuring again
# (#57871). The re-run keeps the pull_request event, so its check stays on the PR.
#
# Usage: llk-perf-gate-rerun.sh <pr-head-sha>
set -euo pipefail

SHA="${1:?usage: llk-perf-gate-rerun.sh <pr-head-sha>}"
REPO="${GITHUB_REPOSITORY:?}"
GATE_JOB='startswith("LLK perf regression gate (")'
# A compare job takes about 5 min; wait up to 20 min for each re-run.
POLL_SECONDS=15
MAX_POLLS=80

# The newest run whose gate jobs ran: a label event adds newer runs where every job is skipped.
RUN=""
for ID in $(gh api "repos/${REPO}/actions/runs?head_sha=${SHA}&event=pull_request&per_page=50" \
    --jq '.workflow_runs[] | select(.path == ".github/workflows/llk-perf-gate.yaml") | .id'); do
    JOBS=$(gh api --paginate "repos/${REPO}/actions/runs/${ID}/jobs?per_page=100" \
        --jq ".jobs[] | select((.name | ${GATE_JOB}) and .conclusion != \"skipped\") | \"\(.id) \(.conclusion)\"")
    if [ -n "${JOBS}" ]; then
        RUN="${ID}"
        break
    fi
done
if [ -z "${RUN}" ]; then
    echo "No perf gate run for ${SHA} compared anything; nothing to run again."
    exit 0
fi
STATUS=$(gh api "repos/${REPO}/actions/runs/${RUN}" --jq .status)
if [ "${STATUS}" != "completed" ]; then
    echo "Perf gate run ${RUN} is still ${STATUS}. It reads the table when it compares."
    exit 0
fi

if grep -q ' failure$' <<< "${JOBS}"; then
    gh api -X POST "repos/${REPO}/actions/runs/${RUN}/rerun-failed-jobs" > /dev/null
    echo "Running the failed jobs of run ${RUN} again."
    exit 0
fi

# Report-only: the gate jobs passed, so run each one again, one at a time.
for JOB in $(cut -d' ' -f1 <<< "${JOBS}"); do
    gh api -X POST "repos/${REPO}/actions/jobs/${JOB}/rerun" > /dev/null
    echo "Running job ${JOB} of run ${RUN} again."
    DONE=0
    for _ in $(seq 1 "${MAX_POLLS}"); do
        sleep "${POLL_SECONDS}"
        if [ "$(gh api "repos/${REPO}/actions/runs/${RUN}" --jq .status)" = completed ]; then
            DONE=1
            break
        fi
    done
    if [ "${DONE}" != 1 ]; then
        echo "::error::Run ${RUN} did not finish within $((POLL_SECONDS * MAX_POLLS / 60)) min; the other gate jobs are not run again."
        exit 1
    fi
done
