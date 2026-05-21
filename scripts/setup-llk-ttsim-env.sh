#!/usr/bin/env bash
# Prepare tt_metal/tt-llk/tests for craq-sim LLK pytest sweeps on metal2.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO_ROOT}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

HARNESS="$REPO_ROOT/tt_metal/tt-llk/tests"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/python_env/bin/python3}"

if [ ! -d "$HARNESS/python_tests" ]; then
    echo "ERROR: missing LLK harness: $HARNESS/python_tests" >&2
    exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: no python3 found (set PYTHON_BIN)" >&2
    exit 1
fi

log() { echo "[setup-llk] $*"; }

ensure_venv() {
    local venv="$HARNESS/.venv"
    local venv_python="$venv/bin/python"
    local metal2_pyenv="$REPO_ROOT/python_env"

    # Prefer metal2's main python_env: its tt-exalens supports craq-sim .so paths
    # (fork's LLK venv pins older tt-umd that only accepts RTL sim directories).
    if [ -x "$metal2_pyenv/bin/python" ]; then
        if [ -L "$venv" ] || [ -d "$venv" ]; then
            rm -rf "$venv"
        fi
        log "linking LLK venv -> metal2 python_env ($metal2_pyenv)"
        ln -sfn "$metal2_pyenv" "$venv"
        if [ -x "$venv_python" ]; then
            if ! "$venv_python" -m pip --version >/dev/null 2>&1; then
                "$venv_python" -m ensurepip --upgrade >/dev/null 2>&1 || true
            fi
            "$venv_python" -m pip install -q pytest-forked pytest-xdist pytest-split 2>/dev/null || true
            return 0
        fi
    fi

    if [ -x "$venv_python" ]; then
        log "venv ok: $venv_python"
        return 0
    fi

    log "creating venv at $venv"
    "$PYTHON_BIN" -m venv "$venv"
    # shellcheck source=/dev/null
    source "$venv/bin/activate"
    "$venv_python" -m pip install -q --upgrade pip
    if command -v uv >/dev/null 2>&1; then
        uv pip install -q --index-strategy unsafe-best-match -r "$HARNESS/requirements.txt"
    else
        "$venv_python" -m pip install -q -r "$HARNESS/requirements.txt"
    fi
}

ensure_sfpi() {
    local gxx="$HARNESS/sfpi/compiler/bin/riscv-tt-elf-g++"
    if [ -x "$gxx" ]; then
        log "sfpi ok: $gxx"
        return 0
    fi

    log "installing SFPI via setup_testing_env.sh"
    (cd "$HARNESS" && ./setup_testing_env.sh)
    if [ ! -x "$gxx" ]; then
        echo "ERROR: SFPI install failed; missing $gxx" >&2
        exit 1
    fi
}

ensure_venv
ensure_sfpi
log "LLK ttsim env ready under $HARNESS"
