#!/bin/sh
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Patch the activate script for POSIX sh compatibility.
# The --relocatable flag generates an activate script that derives VIRTUAL_ENV
# from $SCRIPT_PATH at runtime, but $SCRIPT_PATH is only set for bash
# ($BASH_SOURCE), zsh (${(%):-%x}), and ksh (${.sh.file}).
# POSIX sh (dash, etc.) leaves SCRIPT_PATH empty, causing "realpath ''" errors.
# This patch adds a fallback to the hardcoded path for shells that cannot
# determine the script path dynamically. Fish and csh have their own activate
# files (activate.fish, activate.csh) and are not affected.
#
# Usage: scripts/patch_activate_posix.sh <venv_dir>

set -e

VENV_DIR="${1:-}"
if [ -z "$VENV_DIR" ] || [ ! -d "$VENV_DIR" ]; then
    echo "Usage: $0 <venv_dir>" >&2
    echo "  <venv_dir> must be an existing directory (venv root)." >&2
    exit 1
fi

ACTIVATE="$VENV_DIR/bin/activate"
if [ ! -f "$ACTIVATE" ]; then
    echo "Not found: $ACTIVATE" >&2
    exit 1
fi

# Only patch if the script looks like uv's relocatable activate (uses SCRIPT_PATH in realpath)
if ! grep -q 'VIRTUAL_ENV=.*dirname.*SCRIPT_PATH' "$ACTIVATE" 2>/dev/null; then
    echo "Skip (not a relocatable activate or already patched): $ACTIVATE" >&2
    exit 0
fi

# Get absolute path of the venv directory for the hardcoded fallback
VENV_ABS_PATH=$(cd "$VENV_DIR" && pwd)

# Replace the VIRTUAL_ENV=... line with POSIX-compatible logic:
# - If SCRIPT_PATH is set (bash/zsh/ksh): use dynamic resolution
# - Otherwise: use hardcoded absolute path (for POSIX sh/dash)
awk -v venv_path="$VENV_ABS_PATH" -v sq="'" '
/^VIRTUAL_ENV=.*dirname.*SCRIPT_PATH/ {
    print "if [ -n \"${SCRIPT_PATH:-}\" ]; then"
    print "    VIRTUAL_ENV=\"$(dirname -- \"$(dirname -- \"$(realpath -- \"$SCRIPT_PATH\")\")\")\""
    print "else"
    printf "    VIRTUAL_ENV=%s%s%s\n", sq, venv_path, sq
    print "fi"
    next
}
{ print }
' "$ACTIVATE" > "${ACTIVATE}.tmp"
chmod --reference="$ACTIVATE" "${ACTIVATE}.tmp" 2>/dev/null || true
chmod +x "${ACTIVATE}.tmp"
mv "${ACTIVATE}.tmp" "$ACTIVATE"
echo "Patched for POSIX sh: $ACTIVATE"
