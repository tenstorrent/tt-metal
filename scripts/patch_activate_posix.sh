#!/bin/sh
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Patch a relocatable venv's bin/activate so it works under POSIX sh (e.g. Docker
# RUN uses /bin/sh). The unpatched script uses $BASH_SOURCE for path resolution,
# which is undefined in sh, causing "realpath: '': No such file or directory"
# and a wrong PATH (e.g. ./bin instead of the absolute venv path).
#
# This script replaces the single VIRTUAL_ENV=... line with logic that:
#   1. Uses optional $1 as venv root when sourced as: . bin/activate /path/to/venv
#   2. Uses pre-set VIRTUAL_ENV when: VIRTUAL_ENV=/path . bin/activate
#   3. Uses SCRIPT_PATH (bash/zsh/ksh) when set
#   4. Tries $0 as script path when it contains / and is a file (some shells)
#   5. Otherwise prints an error and exits
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
if ! grep -q 'realpath -- "\$SCRIPT_PATH"' "$ACTIVATE" 2>/dev/null; then
    echo "Skip (not a relocatable activate or already patched): $ACTIVATE" >&2
    exit 0
fi

# 1) Insert a line to save caller's VIRTUAL_ENV before "deactivate nondestructive" clears it
#    (so "VIRTUAL_ENV=path . activate" works in POSIX sh).
# 2) Replace the single VIRTUAL_ENV=... line with POSIX-compatible logic.
awk '
/^deactivate nondestructive$/ {
    print "[ -n \"\${VIRTUAL_ENV:-}\" ] && _CALLER_VIRTUAL_ENV=\"\$VIRTUAL_ENV\""
    print ""
    print
    next
}
/realpath -- "\$SCRIPT_PATH"/ {
    print "# POSIX sh compatibility: SCRIPT_PATH is only set in bash/zsh/ksh."
    print "# When sourced under /bin/sh, use: . bin/activate /path/to/venv  or  VIRTUAL_ENV=/path . bin/activate"
    print "if [ -n \"\${1:-}\" ] && [ -d \"\$1\" ]; then"
    print "    VIRTUAL_ENV=\"\$1\""
    print "elif [ -n \"\${SCRIPT_PATH:-}\" ]; then"
    print "    VIRTUAL_ENV=\"$(dirname -- \"$(dirname -- \"$(realpath -- \"\$SCRIPT_PATH\")\")\")\""
    print "elif [ -n \"\${_CALLER_VIRTUAL_ENV:-}\" ]; then"
    print "    VIRTUAL_ENV=\"\$_CALLER_VIRTUAL_ENV\""
    print "    unset _CALLER_VIRTUAL_ENV"
    print "elif [ -n \"\$0\" ] && case \"\$0\" in */*) true ;; *) false ;; esac && [ -f \"\$0\" ]; then"
    print "    VIRTUAL_ENV=\"$(dirname -- \"$(dirname -- \"$(realpath -- \"\$0\")\")\")\""
    print "else"
    print "    echo \"activate (relocatable): Could not determine venv path. Under POSIX sh use: . bin/activate /path/to/venv  or  VIRTUAL_ENV=/path . bin/activate\" >&2"
    print "    exit 1"
    print "fi"
    next
}
{ print }
' "$ACTIVATE" > "${ACTIVATE}.tmp"
mv "${ACTIVATE}.tmp" "$ACTIVATE"
chmod +x "$ACTIVATE" 2>/dev/null || true
echo "Patched for POSIX sh: $ACTIVATE"
