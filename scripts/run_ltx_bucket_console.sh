#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TT_METAL_SHM_TRACKING_DISABLED="${TT_METAL_SHM_TRACKING_DISABLED:-1}"
export TT_METAL_INSPECTOR="${TT_METAL_INSPECTOR:-1}"
export TT_METAL_INSPECTOR_RPC="${TT_METAL_INSPECTOR_RPC:-1}"
export TT_METAL_LOGS_PATH="${TT_METAL_LOGS_PATH:-/tmp/tt-logs}"
export LTX_VOC_TRACE="${LTX_VOC_TRACE:-0}"
export LTX_BWE_TRACE="${LTX_BWE_TRACE:-0}"
export LTX_VAE_TRACE="${LTX_VAE_TRACE:-0}"
export LTX_VAE_TEMPORAL_CHUNK_LATENTS="${LTX_VAE_TEMPORAL_CHUNK_LATENTS:-0}"

exec python_env/bin/python scripts/ltx_bucket_console.py "$@"
