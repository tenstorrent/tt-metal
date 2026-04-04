#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ULFM diagnostics → GitHub Actions annotations, plus optional whole-script wrap.
#
# Promotion (when GITHUB_ACTIONS is set): lines containing
# "ULFM detected a rank failure" → ::error (policy=fast_fail) or ::warning
# (policy=fault_tolerant). That includes remote ULFM failures from
# handle_rank_failure, local std::terminate (operation=std::terminate), and the
# MPI_Finalize watchdog (operation=MPI_Finalize_watchdog).
#
# C++ emits those structured lines from mpi_distributed_context.cpp (terminate
# handler, SIGALRM watchdog, and emit_rank_failure_diagnostics for remote failures).
#
# Sourced mode (same directory as other multihost scripts):
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   # shellcheck source=ulfm_github_workflow_wrappers.sh
#   source "$SCRIPT_DIR/ulfm_github_workflow_wrappers.sh"
#
# Executed mode — stream annotations live, tee full child output once, then do a
# final best-effort tmp-file scan on exit (multi-host-physical jobs):
#   ./tests/scripts/multihost/ulfm_github_workflow_wrappers.sh run ./tests/scripts/multihost/run_dual_t3k_tests.sh
#
# Emission rules when GITHUB_ACTIONS is set:
#   - policy=fast_fail      → ::error (multihost two-host title when failed_hostname
#                             and detecting_hostname differ and are not unknown-hostname)
#   - policy=fault_tolerant → ::warning
#
# Requires: bash. When GITHUB_ACTIONS is unset or empty, annotation emission is a no-op.
#
# Self-test (galaxy-style ULFM line + encoding assertions):
#   ./tests/scripts/multihost/test_ulfm_gha_annotations_sample_log.sh
#
# Debugging:
#   ULFM_GHA_DEBUG=1 ./tests/scripts/multihost/ulfm_github_workflow_wrappers.sh run ...
#   Emits lifecycle markers before child launch, before final emit, and after final emit.
#
# Workflow commands: the FIRST "::" in the line ends the property list and starts the
# message body. A raw "::" inside title=... (for example "std::terminate") therefore
# breaks parsing. GitHub requires different escaping rules for properties vs message:
# properties escape "%" / CR / LF / ":" / "," while the message body only escapes
# "%" / CR / LF so user-facing text still renders with normal ":" characters.

_ULFM_GHA_CLEANUP_DONE=0
_ULFM_GHA_TMP_OUTPUT=""
_ULFM_GHA_EMITTED_LINES_FILE=""
_ULFM_GHA_STREAM_FIFO=""
_ULFM_GHA_STREAM_PID=""
#ULFM_GHA_DEBUG=1 # for verbose debugging

_gha_escape_workflow_message() {
    local s=$1
    s=${s//'%'/'%25'}
    s=${s//$'\r'/'%0D'}
    s=${s//$'\n'/'%0A'}
    printf '%s' "$s"
}

_gha_escape_workflow_property() {
    local s=$1
    s=${s//'%'/'%25'}
    s=${s//':'/'%3A'}
    s=${s//','/'%2C'}
    s=${s//$'\r'/'%0D'}
    s=${s//$'\n'/'%0A'}
    printf '%s' "$s"
}

_ulfm_gha_debug() {
    [[ -n "${ULFM_GHA_DEBUG:-}" ]] || return 0
    printf '[ULFM_GHA_DEBUG] %s\n' "$*" >&2
}

# Extract value for KEY from a semicolon-separated ULFM diagnostic line (trimmed).
_ulfm_field_value() {
    local line=$1 key=$2
    echo "$line" | sed -n "s/.*${key}=\\([^;]*\\).*/\\1/p" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

_ulfm_github_annotation_title_for_line() {
    local line=$1
    local failed_host="" detecting_host=""

    if [[ "$line" == *"policy=fast_fail"* ]]; then
        if [[ "$line" == *"operation=std::terminate"* ]]; then
            # Avoid "std::" in title — raw "::" splits the workflow command before the message.
            echo "MPI terminate (local rank fatal)"
            return
        fi
        if [[ "$line" == *"operation=MPI_Finalize_watchdog"* ]]; then
            echo "MPI Finalize watchdog (finalize timeout)"
            return
        fi

        failed_host=$(_ulfm_field_value "$line" "failed_hostname")
        detecting_host=$(_ulfm_field_value "$line" "detecting_hostname")
        if [[ -n "$failed_host" && -n "$detecting_host" &&
            "$failed_host" != "$detecting_host" &&
            "$failed_host" != "unknown-hostname" &&
            "$detecting_host" != "unknown-hostname" ]]; then
            echo "ULFM FAST_FAIL multihost (survivor ${detecting_host} vs failed ${failed_host})"
        elif [[ "$failed_host" == "unknown-hostname" || "$line" == *"failed_rank_name=unknown-world-rank"* ]]; then
            echo "ULFM FAST_FAIL (expected test path; failed host unresolved)"
        else
            echo "ULFM FAST_FAIL (expected test path)"
        fi
        return
    fi

    echo "ULFM FAULT_TOLERANT (expected in recovery tests)"
}

_emit_ulfm_github_annotation_for_line() {
    local line=$1
    local emitted_lines_file=${2:-}
    local level="warning"
    local title="" title_esc="" esc=""

    [[ -n "${GITHUB_ACTIONS:-}" ]] || return 0
    [[ "$line" == *"ULFM detected a rank failure"* ]] || return 0

    if [[ -n "$emitted_lines_file" && -f "$emitted_lines_file" ]] &&
        grep -F -x -q -- "$line" "$emitted_lines_file" 2>/dev/null; then
        return 0
    fi

    if [[ "$line" == *"policy=fast_fail"* ]]; then
        level="error"
    fi

    title=$(_ulfm_github_annotation_title_for_line "$line")
    title_esc=$(_gha_escape_workflow_property "$title")
    esc=$(_gha_escape_workflow_message "$line")
    echo "::${level} title=${title_esc}::$esc"

    if [[ -n "$emitted_lines_file" ]]; then
        printf '%s\n' "$line" >> "$emitted_lines_file"
    fi
}

_ulfm_gha_match_count_in_file() {
    local tmpout=$1
    [[ -s "$tmpout" ]] || {
        echo 0
        return 0
    }
    grep -F -c "ULFM detected a rank failure" "$tmpout" 2>/dev/null || true
}

_emit_ulfm_github_annotations_from_stream() {
    local emitted_lines_file=${1:-}
    local line="" stream_matches=0

    if [[ -z "${GITHUB_ACTIONS:-}" ]]; then
        cat >/dev/null
        return 0
    fi

    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" == *"ULFM detected a rank failure"* ]] || continue
        stream_matches=$((stream_matches + 1))
        _ulfm_gha_debug "phase=stream-match count=${stream_matches}"
        _emit_ulfm_github_annotation_for_line "$line" "$emitted_lines_file"
    done
}

# Emit one workflow command per ULFM structured diagnostic line from a file.
# - policy=fast_fail → ::error (stands out in the Actions UI; used by intentional
#   FAST_FAIL tests such as FastFailEmitsRankFailureDiagnostics).
# - policy=fault_tolerant → ::warning
_emit_ulfm_github_annotations_from_file() {
    local tmpout=$1
    local emitted_lines_file=${2:-}
    [[ -n "${GITHUB_ACTIONS:-}" ]] || return 0
    [[ -s "$tmpout" ]] || return 0

    local line=""
    while IFS= read -r line || [[ -n "$line" ]]; do
        _emit_ulfm_github_annotation_for_line "$line" "$emitted_lines_file"
    done < <(grep -F "ULFM detected a rank failure" "$tmpout" 2>/dev/null || true)
}

_run_ulfm_wrapped_child() {
    if command -v setsid >/dev/null 2>&1; then
        if setsid --help 2>&1 | grep -q -- '--wait'; then
            _ulfm_gha_debug "phase=launch mode=setsid_wait"
            setsid --wait bash "$@"
            return
        fi
        _ulfm_gha_debug "phase=launch mode=setsid"
        setsid bash "$@"
        return
    fi

    _ulfm_gha_debug "phase=launch mode=bash"
    bash "$@"
}

_ulfm_github_wrapper_cleanup() {
    local exit_status=$1
    local reason=$2
    local match_count=0

    if [[ "${_ULFM_GHA_CLEANUP_DONE:-0}" -ne 0 ]]; then
        return 0
    fi
    _ULFM_GHA_CLEANUP_DONE=1

    if [[ -n "${_ULFM_GHA_TMP_OUTPUT:-}" ]]; then
        match_count=$(_ulfm_gha_match_count_in_file "$_ULFM_GHA_TMP_OUTPUT")
    fi
    _ulfm_gha_debug "phase=pre-final-emit reason=${reason} exit_status=${exit_status} matches=${match_count}"
    _emit_ulfm_github_annotations_from_file "$_ULFM_GHA_TMP_OUTPUT" "$_ULFM_GHA_EMITTED_LINES_FILE"
    _ulfm_gha_debug "phase=post-final-emit reason=${reason} exit_status=${exit_status}"

    if [[ -n "${_ULFM_GHA_STREAM_PID:-}" ]]; then
        kill "$_ULFM_GHA_STREAM_PID" 2>/dev/null || true
        wait "$_ULFM_GHA_STREAM_PID" 2>/dev/null || true
    fi
    [[ -n "${_ULFM_GHA_STREAM_FIFO:-}" ]] && rm -f "$_ULFM_GHA_STREAM_FIFO"
    [[ -n "${_ULFM_GHA_TMP_OUTPUT:-}" ]] && rm -f "$_ULFM_GHA_TMP_OUTPUT"
    [[ -n "${_ULFM_GHA_EMITTED_LINES_FILE:-}" ]] && rm -f "$_ULFM_GHA_EMITTED_LINES_FILE"
}

_ulfm_github_wrapper_on_exit() {
    local exit_status=$?
    _ulfm_github_wrapper_cleanup "$exit_status" "EXIT"
}

_ulfm_github_wrapper_on_signal() {
    local signal_name=$1
    local signal_status=$2
    _ulfm_gha_debug "phase=signal name=${signal_name}"
    _ulfm_github_wrapper_cleanup "$signal_status" "SIG${signal_name}"
    trap - "$signal_name"
    exit "$signal_status"
}

# Run a script (and args) with combined stdout/stderr captured; emit ULFM annotations.
# Intended for CI: ./ulfm_github_workflow_wrappers.sh run ./path/to/script.sh [args...]
if [[ "${BASH_SOURCE[0]}" == "${0}" ]] && [[ "${1:-}" == "run" ]]; then
    shift
    if [[ $# -lt 1 ]]; then
        echo "usage: $0 run <script> [args...]" >&2
        exit 2
    fi

    _ULFM_GHA_TMP_OUTPUT=$(mktemp)
    _ULFM_GHA_EMITTED_LINES_FILE=$(mktemp)
    _ULFM_GHA_STREAM_FIFO=$(mktemp -u)
    : > "$_ULFM_GHA_EMITTED_LINES_FILE"
    mkfifo "$_ULFM_GHA_STREAM_FIFO"
    _ULFM_GHA_CLEANUP_DONE=0

    trap '_ulfm_github_wrapper_on_exit' EXIT
    trap '_ulfm_github_wrapper_on_signal TERM 143' TERM
    trap '_ulfm_github_wrapper_on_signal INT 130' INT

    _emit_ulfm_github_annotations_from_stream "$_ULFM_GHA_EMITTED_LINES_FILE" <"$_ULFM_GHA_STREAM_FIFO" &
    _ULFM_GHA_STREAM_PID=$!

    set -o pipefail
    set +e
    _ulfm_gha_debug "phase=before-child argc=$# tmpout=${_ULFM_GHA_TMP_OUTPUT}"
    _run_ulfm_wrapped_child "$@" 2>&1 | tee "$_ULFM_GHA_TMP_OUTPUT" "$_ULFM_GHA_STREAM_FIFO"
    cmd_status=${PIPESTATUS[0]}
    set +o pipefail
    wait "$_ULFM_GHA_STREAM_PID" 2>/dev/null || true
    _ULFM_GHA_STREAM_PID=""
    rm -f "$_ULFM_GHA_STREAM_FIFO"
    _ULFM_GHA_STREAM_FIFO=""
    _ulfm_gha_debug "phase=after-child status=${cmd_status}"
    exit "$cmd_status"
fi
