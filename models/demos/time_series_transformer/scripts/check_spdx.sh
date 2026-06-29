#!/usr/bin/env bash
# scripts/check_spdx.sh
set -euo pipefail
HEADER_LICENSE="# SPDX-License-Identifier: Apache-2.0"
HEADER_COPY="# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc."

cd "$(dirname "$0")/.."
fail=0
for f in $(find tt tests scripts -name "*.py"); do
    if ! grep -q "SPDX-License-Identifier" "$f"; then
        echo "MISSING SPDX: $f"
        fail=1
        if [[ "${1:-}" == "--fix" ]]; then
            tmp=$(mktemp)
            { echo "$HEADER_COPY"; echo "$HEADER_LICENSE"; echo; cat "$f"; } > "$tmp"
            mv "$tmp" "$f"
            echo "  -> fixed"
        fi
    fi
done
exit $fail
