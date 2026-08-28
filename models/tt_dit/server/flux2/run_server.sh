#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Launch the standalone FLUX.2 text-to-image server (uvicorn + FastAPI).
#
# Usage:
#   bash models/tt_dit/server/flux2/run_server.sh
#   FLUX2_MESH_SHAPE=2x2 FLUX2_STEPS=28 bash models/tt_dit/server/flux2/run_server.sh
#   bash models/tt_dit/server/flux2/run_server.sh --skip-venv
#
set -euo pipefail

SKIP_VENV=0
for arg in "$@"; do
  case "$arg" in
    --skip-venv) SKIP_VENV=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# TT_METAL_WATCHER overflows the fabric-router kernel-config buffer on multi-chip
# fabric. Never run the multi-chip server with it set.
unset TT_METAL_WATCHER

# --- Tenstorrent / repo env ---------------------------------------------------
: "${TT_METAL_HOME:=$(git rev-parse --show-toplevel)}"
export TT_METAL_HOME
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH:-}"
export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-${HOME}/.cache/tt_dit}"

# --- Model / mesh selection ---------------------------------------------------
# FLUX.2 requires BOTH parallel factors > 1, so 2x2 is the only geometry that runs
# on a 4-chip box. config.py rejects anything that would resolve sp or tp to 1.
export FLUX2_MESH_SHAPE="${FLUX2_MESH_SHAPE:-2x2}"
export FLUX2_TOPOLOGY="${FLUX2_TOPOLOGY:-linear}"
export FLUX2_HEIGHT="${FLUX2_HEIGHT:-1024}"
export FLUX2_WIDTH="${FLUX2_WIDTH:-1024}"
export FLUX2_STEPS="${FLUX2_STEPS:-50}"

# The 32B transformer, the 24B text encoder and the VAE cannot be co-resident on
# four chips, so both of these stay on by default.
export FLUX2_FSDP="${FLUX2_FSDP:-1}"
export FLUX2_DYNAMIC_LOAD="${FLUX2_DYNAMIC_LOAD:-1}"
export FLUX2_TRACED="${FLUX2_TRACED:-1}"

export FLUX2_PORT="${FLUX2_PORT:-${SERVICE_PORT:-8000}}"

# --- Python env ---------------------------------------------------------------
if [ "${SKIP_VENV}" -eq 0 ]; then
  if [ -f "${TT_METAL_HOME}/python_env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${TT_METAL_HOME}/python_env/bin/activate"
  else
    echo "WARNING: ${TT_METAL_HOME}/python_env not found; pass --skip-venv to silence." >&2
  fi
fi

# --- Dependency preflight -----------------------------------------------------
# Fail fast with an actionable message rather than an opaque import error minutes in.
if ! python -c "import fastapi, uvicorn, pydantic" 2>/dev/null; then
  echo "ERROR: fastapi / uvicorn / pydantic not importable by '$(command -v python)'." >&2
  echo "       pip install fastapi uvicorn 'pydantic>=2'." >&2
  exit 1
fi

echo "Launching FLUX.2 server: mesh=${FLUX2_MESH_SHAPE} topology=${FLUX2_TOPOLOGY} \
shape=${FLUX2_WIDTH}x${FLUX2_HEIGHT} steps=${FLUX2_STEPS} port=${FLUX2_PORT}"

exec python -m uvicorn \
  --host "${FLUX2_HOST:-0.0.0.0}" \
  --port "${FLUX2_PORT}" \
  --lifespan on \
  models.tt_dit.server.flux2.app:app
