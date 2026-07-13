#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Refresh run.json tokens + cost_usd from live session jsonls.
#
# Runs session_cost.py against the active Claude Code session and atomically
# patches $LOG_DIR/run.json. Meant to be invoked at every pipeline step
# boundary by the orchestrator. A shell script (not a bash function) so it
# survives across separate `Bash` tool-call shells — function definitions do
# not persist between Bash tool calls, which caused run.json to never get
# patched on past runs.
#
# Env vars do NOT persist across separate Bash tool-call shells in Claude Code.
# LOG_DIR is recovered via state.py --worktree-dir (this script always runs
# from $WORKTREE_DIR/tt_metal/tt-llk, written by Step 0); START_TIME, MODEL,
# SESSION_ID, and PROJECT_CWD are then read from $LOG_DIR/state.json.
#
# Fail-silent by design: a transient read-during-append must never abort the
# run; the next refresh catches up. Stderr is discarded for the same reason.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${LOG_DIR:-}" ]]; then
    _WORKTREE_DIR="$(cd ../.. && pwd)"
    LOG_DIR=$(python "${SCRIPT_DIR}/state.py" --worktree-dir "${_WORKTREE_DIR}" get LOG_DIR 2>/dev/null || echo "")
fi
: "${LOG_DIR:?LOG_DIR not exported and not found via state.py --worktree-dir}"

if [[ -z "${START_TIME:-}" ]]; then
    START_TIME=$(python "${SCRIPT_DIR}/state.py" --log-dir "${LOG_DIR}" get START_TIME 2>/dev/null || echo "")
fi
: "${START_TIME:?START_TIME not exported and not found in ${LOG_DIR}/state.json}"

MODEL="${MODEL:-$(python "${SCRIPT_DIR}/state.py" --log-dir "${LOG_DIR}" get MODEL 2>/dev/null || echo "")}"
SESSION_ID="${SESSION_ID:-$(python "${SCRIPT_DIR}/state.py" --log-dir "${LOG_DIR}" get SESSION_ID 2>/dev/null || echo "")}"
PROJECT_CWD="${PROJECT_CWD:-$(python "${SCRIPT_DIR}/state.py" --log-dir "${LOG_DIR}" get PROJECT_CWD 2>/dev/null || echo "")}"

_SESSION_ARGS=""
if [[ -n "${SESSION_ID:-}" && -n "${PROJECT_CWD:-}" ]]; then
    _SESSION_ARGS="--session-id ${SESSION_ID} --project-cwd ${PROJECT_CWD}"
fi

python "${SCRIPT_DIR}/session_cost.py" \
    --since "${START_TIME}" \
    ${MODEL:+--model "${MODEL}"} \
    ${_SESSION_ARGS:+${_SESSION_ARGS}} \
    --log-dir "${LOG_DIR}" >/dev/null 2>&1 || true
