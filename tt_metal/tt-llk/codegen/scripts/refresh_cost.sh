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
# Required env vars (exported by the orchestrator):
#   START_TIME  — ISO 8601 timestamp, only usage at/after this is counted
#   LOG_DIR     — run directory containing run.json
# Optional:
#   MODEL       — opus | sonnet | haiku (default: derived per-message from jsonl)
#
# Env vars do NOT persist across separate Bash tool-call shells in Claude Code.
# The orchestrator writes /tmp/codegen_run_state.sh in Step 0; this script
# sources it as a fallback so refresh_cost calls after the first Bash block
# still have the values they need.
#
# Fail-silent by design: a transient read-during-append must never abort the
# run; the next refresh catches up. Stderr is discarded for the same reason.
set -u

# Fall back to state file if env vars were lost across Bash tool-call shells.
if [[ -z "${LOG_DIR:-}" || -z "${START_TIME:-}" ]]; then
    source /tmp/codegen_run_state.sh 2>/dev/null || true
fi

: "${START_TIME:?START_TIME not exported and /tmp/codegen_run_state.sh not found}"
: "${LOG_DIR:?LOG_DIR not exported and /tmp/codegen_run_state.sh not found}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/session_cost.py" \
    --since "${START_TIME}" \
    ${MODEL:+--model "${MODEL}"} \
    --log-dir "${LOG_DIR}" >/dev/null 2>&1 || true
