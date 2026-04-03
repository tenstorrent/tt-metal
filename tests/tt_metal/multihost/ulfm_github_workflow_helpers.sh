#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers to promote ULFM stderr diagnostics to GitHub Actions workflow
# annotations. Intended to be sourced from multihost shell test wrappers:
#
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   # shellcheck source=ulfm_github_workflow_helpers.sh
#   source "$SCRIPT_DIR/ulfm_github_workflow_helpers.sh"
#
# Requires: bash. When GITHUB_ACTIONS is unset or empty, annotation emission is a no-op.

_gha_escape_workflow_message() {
    local s=$1
    s=${s//'%'/'%25'}
    s=${s//$'\r'/'%0D'}
    s=${s//$'\n'/'%0A'}
    printf '%s' "$s"
}

# Emit one ::warning per line containing the ULFM structured diagnostic prefix.
_emit_ulfm_github_annotations_from_file() {
    local tmpout=$1
    [[ -n "${GITHUB_ACTIONS:-}" ]] || return 0
    [[ -s "$tmpout" ]] || return 0
    local line esc
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" == *"ULFM detected a rank failure"* ]] || continue
        esc=$(_gha_escape_workflow_message "$line")
        echo "::warning title=ULFM rank failure::$esc"
    done < <(grep -F "ULFM detected a rank failure" "$tmpout" 2>/dev/null || true)
}
