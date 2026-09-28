#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Print the commits that can hold the merge gate's baseline, newest first: the
# first-parent commits at or before merge-base(<ref>, HEAD) that changed an LLK
# perf input. The merge gate uploads a merge baseline at exactly these commits,
# so the first line is the baseline the gate wants.
#
# Usage: llk-perf-baseline-commits.sh <ref> [max-commits]
set -euo pipefail

REF="${1:?usage: llk-perf-baseline-commits.sh <ref> [max-commits]}"
MAX="${2:-200}"
FIND="$(dirname "${BASH_SOURCE[0]}")/utils/find-changed-files.sh"

START=$(git merge-base "$REF" HEAD)
FOUND=0
for C in $(git rev-list --first-parent -n "$MAX" "$START"); do
    git rev-parse -q --verify "${C}^" >/dev/null || break
    if FIND_CHANGES_BASE="${C}^" FIND_CHANGES_HEAD="$C" GITHUB_OUTPUT="" "$FIND" \
        | grep -qx "llk-perf-inputs-changed=true"; then
        echo "$C"
        FOUND=1
    fi
done
if [ "$FOUND" = 0 ]; then
    echo "No LLK perf change in the last ${MAX} commits before ${START}" >&2
    exit 1
fi
