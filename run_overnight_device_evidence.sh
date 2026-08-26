#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Unsupervised overnight driver for WH Galaxy Milestone A 2D module device evidence.
#
#   1. re-execs itself inside a detached `screen` session so it survives disconnect
#   2. builds tt-metal
#   3. creates the Python environment
#   4. runs the Claude Code agent against tttv2_milestone_a_device_evidence_agent.md
#
# Usage:
#   ./run_overnight_device_evidence.sh                 # build + venv + agent, in screen
#   ./run_overnight_device_evidence.sh --skip-build    # reuse an existing build/
#   ./run_overnight_device_evidence.sh --skip-venv     # reuse an existing python_env/
#   ./run_overnight_device_evidence.sh --no-screen     # run in the foreground
#   ./run_overnight_device_evidence.sh --build-only    # stop after build + venv
#
# Reattach:  screen -r tttv2-evidence
# Detach:    Ctrl-a d
# Tail logs: tail -f <repo>/tttv2_milestone_a_device_evidence/logs/bootstrap.log

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="tttv2-evidence"
EVIDENCE="${REPO}/tttv2_milestone_a_device_evidence"
LOGS="${EVIDENCE}/logs"
BRIEF="${REPO}/tttv2_milestone_a_device_evidence_agent.md"
CLAUDE_BIN="${CLAUDE_BIN:-${HOME}/.local/bin/claude}"

SKIP_BUILD=0
SKIP_VENV=0
USE_SCREEN=1
BUILD_ONLY=0
FORWARD=()   # args to hand to the in-screen re-exec, minus --no-screen

for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1; FORWARD+=("$arg") ;;
        --skip-venv)  SKIP_VENV=1;  FORWARD+=("$arg") ;;
        --build-only) BUILD_ONLY=1; FORWARD+=("$arg") ;;
        --no-screen)  USE_SCREEN=0 ;;
        -h|--help)    sed -n '3,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "ERROR: unknown argument '$arg'" >&2; exit 2 ;;
    esac
done

mkdir -p "$LOGS"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# --------------------------------------------------------------------------
# Stage 0: re-exec inside screen
# --------------------------------------------------------------------------
if [[ "$USE_SCREEN" -eq 1 && -z "${STY:-}" ]]; then
    if ! command -v screen >/dev/null 2>&1; then
        log "screen not installed; installing via apt (requires passwordless sudo)"
        if sudo -n apt-get update -qq && sudo -n apt-get install -y -qq screen; then
            log "screen installed"
        else
            echo "ERROR: could not install screen." >&2
            echo "       Install it manually (sudo apt-get install screen) or re-run with --no-screen." >&2
            echo "       With --no-screen, use nohup/setsid yourself if you need disconnect survival." >&2
            exit 1
        fi
    fi

    # `screen -ls` exits non-zero in some builds even on success; never let it trip set -e.
    if screen -ls 2>/dev/null | grep -q "\.${SESSION}[[:space:]]"; then
        echo "ERROR: screen session '${SESSION}' already exists. Reattach with: screen -r ${SESSION}" >&2
        exit 1
    fi

    log "launching detached screen session '${SESSION}'"
    SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    screen -dmS "$SESSION" -L -Logfile "${LOGS}/screen.log" \
        bash -lc "$(printf '%q ' "$SELF" --no-screen "${FORWARD[@]+${FORWARD[@]}}")"
    sleep 1
    { screen -ls 2>/dev/null || true; } | sed -n "/${SESSION}/p"
    cat <<EOF

Started. The session survives disconnect.

  reattach : screen -r ${SESSION}
  detach   : Ctrl-a d
  raw log  : tail -f ${LOGS}/bootstrap.log
  screenlog: tail -f ${LOGS}/screen.log
  evidence : ${EVIDENCE}/REPORT.md   (written by the agent when it finishes)
EOF
    exit 0
fi

# --------------------------------------------------------------------------
# Everything below runs inside the screen session (or --no-screen foreground).
# Mirror all output to bootstrap.log without piping the build/agent themselves.
# --------------------------------------------------------------------------
exec > >(tee -a "${LOGS}/bootstrap.log") 2>&1

log "repo         : ${REPO}"
log "commit       : $(git -C "$REPO" rev-parse HEAD)"
log "branch       : $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
log "evidence dir : ${EVIDENCE}"

cd "$REPO"

# --- preflight -------------------------------------------------------------
if [[ ! -f "$BRIEF" ]]; then
    log "FATAL: agent brief not found at ${BRIEF}"
    exit 1
fi
if [[ ! -x "$CLAUDE_BIN" ]]; then
    log "FATAL: claude CLI not found/executable at ${CLAUDE_BIN}. Set CLAUDE_BIN=<path>."
    exit 1
fi

device_count="$(ls /dev/tenstorrent 2>/dev/null | wc -l)"
log "tenstorrent device nodes: ${device_count}"
if [[ "$device_count" -ne 32 ]]; then
    log "FATAL: expected 32 device nodes for a WH Galaxy (8,4) mesh, found ${device_count}."
    log "       Refusing to produce partial-mesh evidence."
    exit 1
fi

log "submodule status:"
git -C "$REPO" submodule status --recursive
if git -C "$REPO" submodule status --recursive | grep -qE '^[-+]'; then
    log "FATAL: submodules are not initialized/synced. Run: git submodule update --init --recursive"
    exit 1
fi

tt-smi -ls > "${LOGS}/01_tt_smi_before.log" 2>&1 || log "WARN: tt-smi -ls failed (continuing)"

# --- build -----------------------------------------------------------------
if [[ "$SKIP_BUILD" -eq 0 ]]; then
    log "building tt-metal (Release, ccache) -> ${LOGS}/02_build.log"
    if ! ./build_metal.sh --enable-ccache > "${LOGS}/02_build.log" 2>&1; then
        log "FATAL: build failed. Tail of ${LOGS}/02_build.log:"
        tail -60 "${LOGS}/02_build.log"
        exit 1
    fi
    log "build OK"
else
    log "skipping build (--skip-build)"
fi

# --- python env ------------------------------------------------------------
if [[ "$SKIP_VENV" -eq 0 ]]; then
    log "creating python environment -> ${LOGS}/03_create_venv.log"
    if ! ./create_venv.sh > "${LOGS}/03_create_venv.log" 2>&1; then
        log "FATAL: create_venv.sh failed. Tail of ${LOGS}/03_create_venv.log:"
        tail -60 "${LOGS}/03_create_venv.log"
        exit 1
    fi
    log "python env OK"
else
    log "skipping venv (--skip-venv)"
fi

VENV="${PYTHON_ENV_DIR:-${REPO}/python_env}"
if [[ ! -f "${VENV}/bin/activate" ]]; then
    log "FATAL: no virtualenv at ${VENV}"
    exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export TT_METAL_HOME="$REPO"
export PYTHONPATH="$REPO"
# Serialize any accidental concurrent device use and give the lock a long fuse.
export TT_DEVICE_LOCK_PATH="${TT_DEVICE_LOCK_PATH:-/tmp/tt_device.lock}"
export TT_DEVICE_LOCK_TIMEOUT="${TT_DEVICE_LOCK_TIMEOUT:-3600}"

log "python       : $(python --version 2>&1) at $(command -v python)"
python -c 'import ttnn; print("ttnn import OK")' || { log "FATAL: cannot import ttnn"; exit 1; }

if [[ "$BUILD_ONLY" -eq 1 ]]; then
    log "--build-only requested; stopping before the agent."
    exit 0
fi

# --- agent -----------------------------------------------------------------
PROMPT=$(cat <<EOF
Read ${BRIEF} in full and execute it exactly as written, start to finish.

You are unsupervised. Nobody will answer questions, so never stop to ask one — every decision is
already made in the brief. If something is genuinely blocked, record it as BLOCKED with its logs and
continue with the remaining work.

Fixed parameters for this run:
- Repository root : ${REPO}
- Commit under test: $(git -C "$REPO" rev-parse HEAD) on branch $(git -C "$REPO" rev-parse --abbrev-ref HEAD)
- EVIDENCE        : ${EVIDENCE}     (this is the \$EVIDENCE the brief refers to; logs go in \$EVIDENCE/logs)
- Python env      : already built and activated; do NOT rebuild tt-metal or recreate the venv
- Deliverables    : ${EVIDENCE}/REPORT.md and ${EVIDENCE}/ENVIRONMENT.md

Honor the brief's hard prohibitions without exception. In particular: modify no source file and no
test file, make no git commit/push/checkout, and never run two device pytest processes at once.

Report failures truthfully. A red result with good logs is a successful run of this task; a green
result obtained by editing or skipping tests is a failed one.

End your final message with the absolute path to REPORT.md.
EOF
)

log "starting Claude Code agent -> ${LOGS}/04_agent.log"
log "agent brief : ${BRIEF}"

set +e
# --dangerously-skip-permissions: required, the agent must use every tool unattended.
# stream-json (not text) so a crash mid-run still leaves a readable partial transcript.
"$CLAUDE_BIN" \
    --dangerously-skip-permissions \
    --verbose \
    --print \
    --output-format stream-json \
    "$PROMPT" > "${LOGS}/04_agent.jsonl" 2> "${LOGS}/04_agent.stderr.log"
agent_rc=$?
set -e

log "agent exited with rc=${agent_rc}"

# Render a human-readable transcript from the stream-json log.
python - "$LOGS" <<'PY' || log "WARN: could not render agent transcript"
import json, pathlib, sys
logs = pathlib.Path(sys.argv[1])
src, dst = logs / "04_agent.jsonl", logs / "04_agent.log"
lines = []
for raw in src.read_text(errors="replace").splitlines():
    raw = raw.strip()
    if not raw:
        continue
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        continue
    message = event.get("message") or {}
    for block in message.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            lines.append(f"[{event.get('type')}] {block['text']}")
        elif block.get("type") == "tool_use":
            lines.append(f"[tool_use] {block.get('name')} {json.dumps(block.get('input'))[:2000]}")
    if event.get("type") == "result":
        lines.append(f"[result] {event.get('subtype')} :: {str(event.get('result'))[:4000]}")
dst.write_text("\n".join(lines) + "\n")
print(f"transcript -> {dst}")
PY

tt-smi -ls > "${LOGS}/99_tt_smi_after.log" 2>&1 || log "WARN: final tt-smi -ls failed"

echo
log "=========================== DONE ==========================="
log "agent rc      : ${agent_rc}"
log "report        : ${EVIDENCE}/REPORT.md"
log "environment   : ${EVIDENCE}/ENVIRONMENT.md"
log "all logs      : ${LOGS}"
if [[ -f "${EVIDENCE}/REPORT.md" ]]; then
    echo
    echo "--- REPORT.md (first 60 lines) ---"
    head -60 "${EVIDENCE}/REPORT.md"
else
    log "WARN: REPORT.md was not produced. Inspect ${LOGS}/04_agent.log"
fi

exit "$agent_rc"
