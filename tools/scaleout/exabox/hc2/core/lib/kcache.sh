#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# kc_init <py> <src dir> [load user]: tt-metal's JIT kernel cache kept across run.sh's sub-stages and across runs on the
# same tray-test sha Every mesh open of the run (the bus-bar parts, the chunks
# process, the numeric checks) is its own python process; with one persistent cache dir a later open finds the kernels
# and firmware the first one built (ttnn 0.80.0 reads TT_METAL_CACHE and skips a compile whose dependencies are up to
# date) instead of compiling again. Sourced by run.sh once, in every mode, before anything opens a device.
#   TT_METAL_CACHE  set by the caller: kept as it is (logged "caller's"), never keyed or pruned by us
#   TRAY_KCACHE_ROOT  the dir that holds one cache per key (default <load user's home>/.cache/tray-test-kcache;
#                   with UB0_PYTHON set, a stub or test run, \$TMPDIR/tray-test-kcache, so tests leave no home dir
#                   behind); launchers point it at node-local disk. TRAY_KCACHE=0: tt-metal's own default, untouched.
#   key             <tray-test sha>-ttnn<ttnn version>-<host>: never shared across shas. A dirty tree, or no git and no
#                   COMMIT/VERSION file (a copied tree), keys on a hash of the sources instead (src<12 hex>)
#   TRAY_KCACHE_KEEP  key dirs kept under the root (default 4, the newest by mtime); older ones of our naming go
#   TRAY_KCACHE_MIN_MB  free space the root's disk needs (default 2048); less: tt-metal's own default, logged
# Prints one "kernel cache: ..." line; exports TT_METAL_CACHE (with a trailing /, tt-metal appends its own names) and
# sets KC_DIR. Never fails the run: no set -e, any problem leaves tt-metal's own default and says why.
kc_key() {  # <py> <src dir> -> the key on stdout
  local py=$1 d=$2 s v h
  s=$(sha_line "$d"); s=${s#tray-test sha }
  if ! [[ $s =~ ^[0-9a-f]{40}$ ]]; then   # dirty or unknown: what runs is the sources, not a commit
    s=src$(cd "$d" && cat ./*.py ./*.sh lib/*.py lib/*.sh fleet/*.py fleet/*.sh 2> /dev/null | sha1sum | cut -c1-12)
  else s=${s:0:12}; fi
  # the venv's own ttnn dist-info, not a python start (a stub's call log stays as it was); the sha pins setup.sh's version too
  v=$(ls -d -- "$(dirname -- "$py")"/../lib/python*/site-packages/ttnn-*.dist-info 2> /dev/null | head -1)
  v=${v##*/ttnn-}; v=${v%.dist-info}
  [[ $v =~ ^[0-9A-Za-z.+_-]+$ ]] || v=unknown
  h=$(hostname -s 2> /dev/null | tr -c 'A-Za-z0-9.\n-' _); h=${h:-host}
  echo "$s-ttnn$v-$h"
}
kc_prune() {  # <root> <keep> <current key>: older key dirs of this host's naming only (sha12 or src12, -ttnn, -host)
  local root=$1 keep=$2 cur=$3 n=0 k ch
  [[ $keep =~ ^[1-9][0-9]*$ ]] || keep=4
  case $(readlink -f -- "$root" 2> /dev/null) in ""|/|"$(readlink -f -- "${HOME:-/}")") return 0 ;; esac   # never a home or /
  ch=${cur#*-ttnn}; ch=${ch#*-}   # this host's part of the key: on a shared root another host's dirs are never counted or removed
  while IFS= read -r k; do
    [[ $k =~ ^(src)?[0-9a-f]{12}-ttnn[0-9A-Za-z.+_-]+-[A-Za-z0-9._-]+$ ]] || continue
    [[ ${k#*-ttnn} == *-"$ch" ]] || continue
    [ "$k" = "$cur" ] && continue
    n=$((n + 1)); [ "$n" -le "$keep" ] && continue
    rm -rf -- "${root:?}/${k:?}"
  done < <(ls -1t -- "$root" 2> /dev/null)
}
kc_init() {
  local py=$1 d=$2 u=${3:-} home root key free
  KC_DIR=""
  if [ -n "${TT_METAL_CACHE:-}" ]; then KC_DIR=$TT_METAL_CACHE; echo "kernel cache: TT_METAL_CACHE=$TT_METAL_CACHE (caller's)"; return 0; fi
  [ "${TRAY_KCACHE:-1}" = 0 ] && { echo "kernel cache: tt-metal's own default (TRAY_KCACHE=0)"; return 0; }
  home=$HOME; [ -n "$u" ] && home=$(getent passwd "$u" 2> /dev/null | cut -d: -f6)
  root=${home:-/nonexistent}/.cache/tray-test-kcache; [ -n "${UB0_PYTHON:-}" ] && root=${TMPDIR:-/tmp}/tray-test-kcache
  root=${TRAY_KCACHE_ROOT:-$root}
  key=$(kc_key "$py" "$d")
  if ! ${KC_AS[@]+"${KC_AS[@]}"} mkdir -p -- "$root/$key" 2> /dev/null || ! ${KC_AS[@]+"${KC_AS[@]}"} test -w "$root/$key"; then
    echo "kernel cache: tt-metal's own default ($root/$key not writable)"; return 0; fi
  free=$(df -Pk -- "$root" 2> /dev/null | awk 'NR == 2 { print int($4 / 1024) }')
  if [[ $free =~ ^[0-9]+$ ]] && [ "$free" -lt "${TRAY_KCACHE_MIN_MB:-2048}" ]; then
    echo "kernel cache: tt-metal's own default ($root has $free MB free, under TRAY_KCACHE_MIN_MB=${TRAY_KCACHE_MIN_MB:-2048})"; return 0; fi
  ${KC_AS[@]+"${KC_AS[@]}"} touch -- "$root/$key" 2> /dev/null   # the newest by mtime: kept by the prune below
  ${KC_AS[@]+"${KC_AS[@]}"} bash -c "$(declare -f kc_prune); kc_prune \"\$@\"" kc "$root" "${TRAY_KCACHE_KEEP:-4}" "$key"
  KC_DIR=$root/$key/; export TT_METAL_CACHE=$KC_DIR
  echo "kernel cache: TT_METAL_CACHE=$KC_DIR ($(ls -A -- "$KC_DIR" 2> /dev/null | grep -c .) entries, $free MB free; key $key)"
  return 0
}
