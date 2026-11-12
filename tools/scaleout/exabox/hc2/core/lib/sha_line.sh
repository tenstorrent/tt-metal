#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# sha_line <dir>: prints "tray-test sha <40-hex>[ dirty]". Never fails: no set -e, works without git.
# dirty = uncommitted changes outside tests/. Not a git checkout: COMMIT or VERSION file, else "unknown".
sha_line() {
  local d=${1:-.} s="" dirty="" f
  s=$(git -C "$d" rev-parse HEAD 2> /dev/null) || s=""
  if [[ $s =~ ^[0-9a-f]{40}$ ]]; then
    [ -n "$(git -C "$d" status --porcelain -- . ':(exclude)tests' ':(exclude)traytest/tests' 2> /dev/null | head -1)" ] && dirty=" dirty"
  else
    s=""
    for f in COMMIT VERSION; do
      [ -r "$d/$f" ] && s=$(grep -a -m1 -oE '[0-9a-f]{40}' "$d/$f" 2> /dev/null) && [ -n "$s" ] && break
      s=""
    done
    [ -n "$s" ] || s=unknown
  fi
  echo "tray-test sha $s$dirty"
  return 0
}
