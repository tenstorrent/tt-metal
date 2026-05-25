#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Verify tracy-capture / tracy-csvexport run on this host (no SIGILL from -march mismatch).
# Used in profiler CI on T3K VMs and after profiler artifact builds.

set -euo pipefail

TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$(dirname "$0")/../../.." && pwd)}"
TRACY_BIN_DIR="${TT_METAL_HOME}/build/tools/profiler/bin"

check_tool() {
    local tool="$1"
    local path="${TRACY_BIN_DIR}/${tool}"

    if [[ ! -x "${path}" ]]; then
        echo "ERROR: Missing executable ${path}"
        exit 1
    fi

    set +e
    local output
    output="$("${path}" 2>&1)"
    local rc=$?
    set -e

    # Bash reports SIGILL as exit code 132 (128 + 4).
    if [[ ${rc} -eq 132 ]]; then
        echo "ERROR: ${tool} illegal instruction (SIGILL) on this CPU"
        exit 1
    fi

    if ! grep -q "tracy-" <<<"${output}"; then
        echo "ERROR: ${tool} unexpected output (rc=${rc}): ${output}"
        exit 1
    fi

    echo "OK: ${tool} usage smoke (exit ${rc})"
}

check_capture_worker_path() {
    local capture="${TRACY_BIN_DIR}/tracy-capture"
    local tmp
    tmp="$(mktemp /tmp/tracy_smoke.XXXXXX.tracy)"

    set +e
    timeout 3s "${capture}" -o "${tmp}" -f -p 65534 >/dev/null 2>&1
    local rc=$?
    set -e
    rm -f "${tmp}"

    if [[ ${rc} -eq 132 ]]; then
        echo "ERROR: tracy-capture illegal instruction (SIGILL) during Worker startup"
        exit 1
    fi

    echo "OK: tracy-capture Worker startup smoke (exit ${rc}, timeout/connection failure expected)"
}

check_tool tracy-capture
check_tool tracy-csvexport
check_capture_worker_path
