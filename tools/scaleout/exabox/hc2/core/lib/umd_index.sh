# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# shellcheck shell=bash
# lib/umd_index.sh: the chip index a per-chip open passes in TT_VISIBLE_DEVICES.
# tt-umd reads each TT_VISIBLE_DEVICES token as an index into its own list of chips sorted by PCI address
# (LC_ALL=C sort of the BDF string; tt-umd enumerate_devices), never as the /dev/tenstorrent number. On every recorded
# 32-chip open that list is [0..15, 24..31, 16..23]: /dev 16-23 = c1-c8:00.0 (tray 3) but UMD index 16-23 = 81-88:00.0
# (tray 4), so an open by /dev number put the other tray's twin chip under the label (notes/runsh-open-map-7023.md).
# The index is computed from sysfs at each open (a chip off the bus shifts every later index), as bin/umd-index does.
# /dev numbers stay for ethmask TTDevice.create and tt-smi labels. Functions only; uses SYS (run.sh: /sys).
#   rk_umd <bdf>...                  comma list of UMD indices, in argument order; exit 1 if a chip has no tt-kmd node
#                                    (UMD_MAP=<log>.map: from that open-time map instead of the live sysfs)
#   umd_snap <log>                   writes the live "<bdf> <devN>" map to <log>.map when an open launches
#   rk_opencheck <log> <what> <bdf>... the open's "PCIe ids" bracket must be exactly those chips' /dev numbers; else a
#                                    FAIL line naming the intended and the opened chips by BDF and tray 1-4 (RK_OPEN_BAD).
#                                    It maps by <log>.map, the open-time map: after a hang the MCE guard or fl_watch's
#                                    loss branch may have removed a chip from sysfs. With no map and an intended chip
#                                    off the bus now, it does not compare and says why (no finding).
umd_live() {  # "<bdf> <devN>" of each chip UMD can open now (it has a tt-kmd node), in UMD's order
  local d n
  for d in "${SYS:-/sys}"/bus/pci/devices/*; do
    for n in "$d"/tenstorrent/tenstorrent!*; do
      [[ ${n##*/tenstorrent!} =~ ^[0-9]+$ ]] && { echo "${d##*/} ${n##*/tenstorrent!}"; break; }; done
  done | LC_ALL=C sort -k1,1
}
umd_snap() { umd_live > "$1.map" 2> /dev/null || :; }
rk_umd() {
  [ $# -gt 0 ] || return 1
  { if [ -n "${UMD_MAP:-}" ] && [ -r "$UMD_MAP" ]; then cat "$UMD_MAP"; else umd_live; fi; } | awk -v want="$*" 'BEGIN { n = split(want, w, " "); for (i = 1; i <= n; i++) if (w[i] !~ /^[0-9a-f]+:[0-9a-f]+:/) w[i] = "0000:" w[i] }
    { idx[$1] = NR - 1 }
    END { s = ""; m = 0; for (i = 1; i <= n; i++) { if (w[i] in idx) s = s (s == "" ? "" : ",") idx[w[i]]; else m = 1 }; print s; exit m }'
}
umd_lk() {  # bdf -> "c1:00.0 (tray 3, /dev 16)"
  local t="?"; declare -F rk_tray > /dev/null && t=$(rk_tray "$1" | cut -d' ' -f1)
  echo "${1#0000:} (tray $t, /dev ${2:-?})"
}
rk_opencheck() {
  local log=$1 what=$2 line ids got want l b n; shift 2
  line=$(grep -aoE 'Opening local chip ids/PCIe ids: \{[0-9, ]*\}/\[[0-9, ]*\]' "$log" 2>/dev/null | head -1)
  [ -n "$line" ] || return 0   # nothing opened (or no UMD line): nothing to compare
  if [ -r "$log.map" ]; then l=$(cat "$log.map"); else l=$(umd_live)
    for b in "$@"; do [[ $b =~ ^[0-9a-f]+:[0-9a-f]+: ]] || b=0000:$b
      awk -v b="$b" '$1 == b { f = 1 } END { exit !f }' <<< "$l" && continue
      l="open check of $what: not compared, $(umd_lk "$b") has left the bus since the open and the log has no open-time map ($log)"
      if declare -F say > /dev/null; then say "$l"; else echo "$l"; fi
      return 0; done; fi
  ids=$(sed -E 's/.*\/\[([0-9, ]*)\]/\1/' <<< "$line" | tr -d ' ' | tr , '\n' | grep . | sort -n | paste -sd,)
  want=$(for b in "$@"; do [[ $b =~ ^[0-9a-f]+:[0-9a-f]+: ]] || b=0000:$b; awk -v b="$b" '$1 == b { print $2 }' <<< "$l"; done | sort -n | paste -sd,)
  [ "$ids" = "$want" ] && return 0
  got=$(tr , '\n' <<< "$ids" | while read -r n; do b=$(awk -v n="$n" '$2 == n { print $1 }' <<< "$l"); umd_lk "${b:-?}" "$n"; done | paste -sd';' | sed 's/;/, /g')
  want=$(for b in "$@"; do [[ $b =~ ^[0-9a-f]+:[0-9a-f]+: ]] || b=0000:$b; umd_lk "$b" "$(awk -v b="$b" '$1 == b { print $2 }' <<< "$l")"; done | paste -sd';' | sed 's/;/, /g')
  l="FAIL $what: run.sh bug: opened $got instead of $want ($log)"
  RK_OPEN_BAD="${RK_OPEN_BAD:+$RK_OPEN_BAD; }$l"
  if declare -F fl_find > /dev/null && [ -n "${FL_OUT:-}" ]; then fl_find "$l"; elif declare -F say > /dev/null; then say "$l"; else echo "$l"; fi
  return 1
}
