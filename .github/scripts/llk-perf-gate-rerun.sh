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

read -r RUN STATUS < <(gh api "repos/${REPO}/actions/runs?head_sha=${SHA}&event=pull_request&per_page=50" \
    --jq '[.workflow_runs[] | select(.path == ".github/workflows/llk-perf-gate.yaml")][0] | "\(.id // "") \(.status // "")"')
if [ -z "${RUN}" ]; then
    echo "No perf gate run for ${SHA}; nothing to run again."
    exit 0
fi
if [ "${STATUS}" != "completed" ]; then
    echo "Perf gate run ${RUN} is still ${STATUS}. It reads the table when it compares."
    exit 0
fi

JOBS=$(gh api --paginate "repos/${REPO}/actions/runs/${RUN}/jobs?per_page=100" \
    --jq '.jobs[] | select((.name | startswith("LLK perf regression gate (")) and .conclusion != "skipped") | "\(.id) \(.conclusion)"')
if [ -z "${JOBS}" ]; then
    echo "Run ${RUN} has no gate job that ran; nothing to run again."
    exit 0
fi

if grep -q ' failure$' <<< "${JOBS}"; then
    gh api -X POST "repos/${REPO}/actions/runs/${RUN}/rerun-failed-jobs" > /dev/null
    echo "Running the failed jobs of run ${RUN} again."
    exit 0
fi

# Report-only: the gate jobs passed, so run each one again, one at a time.
for ID in $(cut -d' ' -f1 <<< "${JOBS}"); do
    gh api -X POST "repos/${REPO}/actions/jobs/${ID}/rerun" > /dev/null
    echo "Running job ${ID} of run ${RUN} again."
    for _ in $(seq 1 60); do
        sleep 15
        [ "$(gh api "repos/${REPO}/actions/runs/${RUN}" --jq .status)" = completed ] && break
    done
done
