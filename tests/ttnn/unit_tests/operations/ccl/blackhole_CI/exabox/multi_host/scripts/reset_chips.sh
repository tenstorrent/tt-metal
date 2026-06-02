#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Resets chips for the multi-host CCL exabox tests.
# Runs `tt-smi -glx_reset_auto` in parallel on every host in HOSTS (or
# locally for SINGLE_BH). Wallclock ≈ 60 s.

set -uo pipefail

DEFAULT_QUAD_HOSTS="bh-glx-b06u02,bh-glx-b06u08,bh-glx-b07u02,bh-glx-b07u08"
DEFAULT_DUAL_HOSTS="bh-glx-b06u02,bh-glx-b06u08"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-h|--help]

Resets chips on all hosts targeted by the multi-host CCL tests by running
'tt-smi -glx_reset_auto' on each in parallel. Wallclock ≈ 60 s.

Required environment:
  TT_METAL_HOME    Repo root (used to find tt-smi). Default: /data/llong/tt-metal

Optional environment:
  MESH_DEVICE      One of SINGLE_BH | DUAL_BH | QUAD_BH. Default: QUAD_BH.
                     SINGLE_BH → reset locally only (no ssh).
                     DUAL_BH   → reset on 2 hosts in parallel.
                     QUAD_BH   → reset on 4 hosts in parallel.

  HOSTS            Comma- or space-separated host list (per MESH_DEVICE).
                   Default for QUAD_BH: ${DEFAULT_QUAD_HOSTS}
                   Default for DUAL_BH: ${DEFAULT_DUAL_HOSTS}
                   *** OVERRIDE THIS for a different cluster. ***
                   Ignored for SINGLE_BH.

When to run:
  - Before the first test of a session.
  - After any hung run.
  - Whenever you see 'Device N init: failed to initialize FW' in test logs.
  - Whenever you see 'Waiting for lock CHIP_IN_USE_*_PCIe' in the next
    run's startup.

Examples:
  bash $0
  MESH_DEVICE=DUAL_BH HOSTS="h1,h2" bash $0
  MESH_DEVICE=SINGLE_BH bash $0
EOF
}

# --- arg parsing ----------------------------------------------------------

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") ;;
    *)  echo "[error] unexpected argument: $1" >&2
        echo "Run with --help for usage." >&2
        exit 2 ;;
esac

# --- environment ----------------------------------------------------------

TT_METAL_HOME="${TT_METAL_HOME:-/data/llong/tt-metal}"
MESH_DEVICE="${MESH_DEVICE:-QUAD_BH}"
TT_SMI="${TT_METAL_HOME}/python_env/bin/tt-smi"

if [ ! -x "$TT_SMI" ]; then
    echo "[reset_chips] ERROR: tt-smi not found at $TT_SMI" >&2
    echo "[reset_chips]   make sure \$TT_METAL_HOME is correct and python_env is set up" >&2
    exit 1
fi

case "${MESH_DEVICE}" in
    SINGLE_BH) HOSTS="" ;;
    DUAL_BH)   HOSTS="${HOSTS:-${DEFAULT_DUAL_HOSTS}}" ;;
    QUAD_BH)   HOSTS="${HOSTS:-${DEFAULT_QUAD_HOSTS}}" ;;
    *)
        echo "[reset_chips] ERROR: unsupported MESH_DEVICE='${MESH_DEVICE}'" >&2
        echo "[reset_chips]   use SINGLE_BH, DUAL_BH, or QUAD_BH" >&2
        exit 2 ;;
esac

# Normalize HOSTS: accept comma- or space-separated, emit space-separated
# (so `for h in $HOSTS` iterates correctly).
if [ -n "${HOSTS}" ]; then
    HOSTS="$(echo "${HOSTS}" | tr ',' ' ' | tr -s ' ' | sed 's/^ *//;s/ *$//')"
fi

# --- pre-flight banner ----------------------------------------------------

echo "[reset_chips] MESH_DEVICE = ${MESH_DEVICE}"
if [ -z "${HOSTS}" ]; then
    echo "[reset_chips] mode        = local (SINGLE_BH)"
else
    echo "[reset_chips] hosts       = ${HOSTS}"
fi

# --- reset ---------------------------------------------------------------

LOG_DIR="${LOG_DIR:-/tmp}"
mkdir -p "$LOG_DIR"

if [ -z "${HOSTS}" ]; then
    # SINGLE_BH: no ssh.
    echo "[reset_chips] resetting locally..."
    "$TT_SMI" -glx_reset_auto
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[reset_chips] FAIL (rc=$rc)" >&2
        exit $rc
    fi
    echo "[reset_chips] done."
    exit 0
fi

# Multi-host: parallel ssh.
echo "[reset_chips] resetting in parallel (~60s)..."
PIDS=()
for h in $HOSTS; do
    ssh -o BatchMode=yes "$h" "$TT_SMI -glx_reset_auto" \
        > "$LOG_DIR/reset_${h}.log" 2>&1 &
    PIDS+=($!)
done

# Wait for everyone, collect status.
fail=0
i=0
for h in $HOSTS; do
    pid="${PIDS[$i]}"
    if wait "$pid"; then
        if grep -q "Re-initialized 32 boards" "$LOG_DIR/reset_${h}.log"; then
            echo "[reset_chips] $h: OK (32 boards re-initialized)"
        else
            echo "[reset_chips] $h: completed but no 're-initialized' marker — see $LOG_DIR/reset_${h}.log"
            fail=1
        fi
    else
        echo "[reset_chips] $h: FAILED — see $LOG_DIR/reset_${h}.log" >&2
        tail -3 "$LOG_DIR/reset_${h}.log" 2>/dev/null | sed 's/^/    /' >&2
        fail=1
    fi
    i=$((i + 1))
done

if [ $fail -eq 0 ]; then
    echo "[reset_chips] done."
    exit 0
else
    echo "[reset_chips] one or more hosts failed; see $LOG_DIR/reset_*.log" >&2
    exit 1
fi
