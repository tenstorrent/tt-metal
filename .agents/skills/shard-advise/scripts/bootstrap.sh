#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Activate the pre-built tt-mlir L1 shard advisor environment for use inside an
# agentic-research experiment. Source this (do not execute):
#
#     source .agents/skills/shard-advise/scripts/bootstrap.sh
#
# It does NOT build tt-mlir. Building is one-time operator setup (see the
# integration README). This only locates + activates that env and makes sure a
# system descriptor exists so the optimizer's OpModel queries have a device.

# --- locate the advisor checkout -------------------------------------------
if [ -z "${TTMLIR_ADVISOR_HOME:-}" ]; then
  echo "shard-advise: TTMLIR_ADVISOR_HOME is not set." >&2
  echo "  Point it at a tt-mlir checkout built with -DTTMLIR_ENABLE_OPMODEL=ON" >&2
  echo "  -DTTMLIR_ENABLE_TTNN_JIT=ON (one-time; see integration README)." >&2
  return 1 2>/dev/null || exit 1
fi
if [ ! -d "$TTMLIR_ADVISOR_HOME" ]; then
  echo "shard-advise: TTMLIR_ADVISOR_HOME=$TTMLIR_ADVISOR_HOME does not exist." >&2
  return 1 2>/dev/null || exit 1
fi

# --- activate its venv ------------------------------------------------------
# shellcheck disable=SC1091
source "$TTMLIR_ADVISOR_HOME/env/activate"

if ! command -v ttnn-advise >/dev/null 2>&1; then
  echo "shard-advise: 'ttnn-advise' not on PATH after activating the advisor env." >&2
  echo "  The tt-mlir build may lack -DTTMLIR_ENABLE_TTNN_JIT=ON, or was not" >&2
  echo "  reinstalled after a code change (run: cmake --build build)." >&2
  return 1 2>/dev/null || exit 1
fi

# --- tracing needs these (see tt-mlir ttnn-jit build notes) -----------------
export LIBRARY_PATH="$TTMLIR_ADVISOR_HOME/.local/libnsl-shim:${LIBRARY_PATH:-}"
if [ -n "${TT_METAL_HOME:-}" ]; then
  export PYTHONPATH="$TT_METAL_HOME/tools:${PYTHONPATH:-}"
fi

# --- ensure a system descriptor (the mock device the optimizer queries) -----
if [ -z "${SYSTEM_DESC_PATH:-}" ]; then
  DEFAULT_SD="$TTMLIR_ADVISOR_HOME/ttrt-artifacts/system_desc.ttsys"
  if [ ! -f "$DEFAULT_SD" ]; then
    echo "shard-advise: generating system descriptor via ttrt query ..." >&2
    ( cd "$TTMLIR_ADVISOR_HOME" && ttrt query --save-artifacts ) || {
      echo "shard-advise: ttrt query failed; set SYSTEM_DESC_PATH manually." >&2
      return 1 2>/dev/null || exit 1
    }
  fi
  export SYSTEM_DESC_PATH="$DEFAULT_SD"
fi

echo "shard-advise: ready. ttnn-advise=$(command -v ttnn-advise)"
echo "shard-advise: SYSTEM_DESC_PATH=$SYSTEM_DESC_PATH"
