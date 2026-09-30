#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# CI entry point of the LLK SFPU report: one architecture, on this runner's device.
# Called from tests/pipeline_reorg/llk_sfpu_report_tests.yaml by llk-sfpu-report.yaml,
# with the checkout of the TOOL revision (main) as the working tree.
#
#   PR_NUMBER  HEAD_SHA  BASE_SHA   the PR, its head, and the merge-base (from the plan job)
#   MODE       merge-base | rebase  (rebase: main vs main + the PR's device diff)
#   OPS        comma-separated MathOperation names, or empty to auto-detect
#   ITERATIONS runs per side
#   OUT_DIR    where summary-<arch>.json, report-<arch>.md and run.log end up
set -euo pipefail

ARCH="$1"
: "${PR_NUMBER:?}" "${HEAD_SHA:?}" "${BASE_SHA:?}" "${OUT_DIR:?}"
MODE="${MODE:-merge-base}"
ITERATIONS="${ITERATIONS:-3}"

LLK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LLK"

(cd tests && ./setup_testing_env.sh >/dev/null)

# Only the two commits are needed: their trees, never their history. The PR ref is
# fetched by name so a fork's head is reachable, then pinned to the planned SHA.
git fetch --no-tags --depth=1 origin "+refs/pull/${PR_NUMBER}/head:refs/remotes/pr/head" "$BASE_SHA"
git cat-file -e "${HEAD_SHA}^{commit}" 2>/dev/null ||
    git fetch --no-tags --depth=1 origin "$HEAD_SHA"

args=(--arch "$ARCH" --head "$HEAD_SHA" --base "$BASE_SHA" --mode "$MODE"
    --work "${RUNNER_TEMP:-/tmp}/llk-sfpu-report" --jobs "$(nproc)")
run_args=(--iterations "$ITERATIONS" --pr "$PR_NUMBER")
[ -n "${OPS:-}" ] && run_args+=(--ops "$OPS")
[ -n "${RUN_URL:-}" ] && run_args+=(--run-url "$RUN_URL")
[ -n "${HEAD_MOVED_TO:-}" ] && run_args+=(--head-moved-to "$HEAD_MOVED_TO")

status=0
python3 sfpu_report/cli.py "${args[@]}" run "${run_args[@]}" || status=$?

mkdir -p "$OUT_DIR"
cp "${RUNNER_TEMP:-/tmp}"/llk-sfpu-report/{summary,report}-"$ARCH".* "$OUT_DIR"/ 2>/dev/null || true
cp "${RUNNER_TEMP:-/tmp}/llk-sfpu-report/run.log" "$OUT_DIR/run-$ARCH.log" 2>/dev/null || true
exit "$status"
