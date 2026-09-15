#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Shared NNG callback setup and orphan ownership for the LLK and Metal runners.

_resolve_nng_channel() {
  local callback_host callback_port
  NNG_HOST="$(hostname 2>/dev/null || uname -n)"
  NNG_LOCAL="${NNG_SOCKET_LOCAL_PORT:-5555}"
  NNG_ADDR="${NNG_SOCKET_ADDR:-}"
  [[ -n "$NNG_ADDR" ]] && return 0

  callback_host="$NNG_HOST"
  callback_port="$NNG_LOCAL"
  if [[ -f /.dockerenv ]]; then
    callback_host="${callback_host%%-special-*}"
    callback_port="${P_USER_DBD_PORT:-}"
    [[ -n "$callback_port" ]] ||
      callback_port="$(bash -lc 'printf "%s" "${P_USER_DBD_PORT:-}"' 2>/dev/null)"
    [[ "$callback_port" =~ ^[0-9]+$ ]] || {
      echo "ERROR: NNG_SOCKET_ADDR is unset and IRD did not provide a valid P_USER_DBD_PORT" >&2
      return 3
    }
  fi
  NNG_ADDR="tcp://${callback_host}:${callback_port}"
}

# Call only while holding LOCKFILE. Keep the previous owner's exact remote host
# and tag in the lock inode so a hard-killed runner can be reaped by its successor.
_reap_previous_nng_job() {
  local current_tag="$1" previous_host previous_tag
  if read -r previous_host previous_tag < "$LOCKFILE" &&
     [[ -n "$previous_host" && -n "$previous_tag" && -x "$REAP" ]]; then
    bash "$REAP" --arch "$ARCH" --emu-host "$previous_host" --lock "$LOCKFILE" \
      --tag "$previous_tag" --force >&2 2>&1 || true
  fi
  printf '%s %s\n' "$EMU_HOST" "$current_tag" > "$LOCKFILE"
}
