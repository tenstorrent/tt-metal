# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/vcore.sh: stage 5 numeric verify, the heavy chunk's VCORE sampler. Read-only hwmon in0_input
# (mV) of each chip every FLEET_VCORE_S s (default 0.1, never below 0.1), its own process, only while the heavy chunk runs.
# No device open, no sync or readback in the load, no power write. Min and mean per chip land next to each numeric hit.
#   vc_start <dir>   starts the sampler, samples in <dir>/vcore.txt ("<epoch> <bdf> <mV>"); vc_stop ends it (the chunk's end,
#                    and a hang before any capture or hold, so a hung chip is never read); <dir>/vcore_skip: BDFs it leaves out
#   vc_report <dir> <report.txt> [<c8_telemetry.txt>]   stage 5 lines: one per chip with a NUMERIC FAIL/WARN, one summary
#                    row for the others; then fleet/vcore_chips.py's per-chip load voltage: one line per tray
#                    (mean [min-max] mV @ AICLK MHz), a box line and VCORE HIGH for a chip FLEET_VCORE_HIGH_MV (default 30) or
#                    more over the box median; also kept in <dir>/vcore_chips.txt for the run summary. Report only.
VC_PID=""
VC_PY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../fleet" && pwd)/vcore_chips.py
vc_interval() { awk -v s="${FLEET_VCORE_S:-0.1}" 'BEGIN { if (s !~ /^[0-9]*\.?[0-9]+$/ || s + 0 < 0.1) s = 0.1; print s }'; }
vc_start() {
  local d=$1 main=$$ iv iso=/dev/null; iv=$(vc_interval); : > "$d/vcore.txt"
  declare -F mg_iso_path > /dev/null && iso=$(mg_iso_path)   # chips the MCE guard cut: never read again
  # no fork per chip per poll: a chip's BDF is resolved once per tt-kmd node and again
  # only when its device link no longer points at it; the skip list, hwmon dir and vcore are read with builtins
  ( trap - EXIT; local -A cb; local -a sk isl; local s
    while kill -0 "$main" 2>/dev/null && [ ! -e "$d/vcore_stop" ]; do
      sk=(); [ -r "$d/vcore_skip" ] && mapfile -t sk < "$d/vcore_skip" 2> /dev/null
      isl=(); [ -r "$iso" ] && mapfile -t isl < "$iso" 2> /dev/null
      for c in "${SYS:-}"/class/tenstorrent/tenstorrent!*; do
        b=${cb[$c]:-}
        if [ -z "$b" ] || ! [ "$c/device" -ef "${SYS:-}/bus/pci/devices/$b" ]; then b=$(basename "$(readlink -f "$c/device")"); cb[$c]=$b; fi
        for s in "${sk[@]}"; do { [ "$s" = "$b" ] || [ "$s" = "${b#0000:}" ]; } && continue 2; done
        for s in "${isl[@]}"; do [ "$s" = "$b" ] && continue 2; done
        h=; for h in "$c"/device/hwmon/hwmon*; do [ -e "$h" ] && break; h=; done
        v=; { read -r v < "$h/in0_input"; } 2> /dev/null
        [[ $v =~ ^[0-9]{1,5}$ ]] && echo "${EPOCHSECONDS} $b $((10#$v))" >> "$d/vcore.txt"
      done; sleep "$iv"
    done ) 2>/dev/null &
  VC_PID=$!
}
vc_stop() { [ -n "$VC_PID" ] || return 0; kill "$VC_PID" 2>/dev/null; wait "$VC_PID" 2>/dev/null; VC_PID=""; }
vc_report() {
  local d=$1 hits; hits=$(grep -aE '^NUMERIC (FAIL|WARN) ' "$2" 2>/dev/null | grep -aoE '[0-9a-f]{2}:00\.0' | sort -u)
  [ -s "$d/vcore.txt" ] || { echo "NUMERIC VCORE: no samples (stage 5 numeric verify: heavy chunk)"; return 0; }
  awk -v hits="$hits" 'BEGIN { n = split(hits, a, "\n"); for (i = 1; i <= n; i++) h[a[i]] = 1 }
    { b = $2; sub(/^0000:/, "", b); k[b]++; s[b] += $3; if (!(b in m) || $3 < m[b]) m[b] = $3; if (!(b in h)) { T++; S += $3; if (!(any++) || $3 < M) M = $3 } }
    END { c = 0; for (b in k) { if (b in h) printf "%s\t%s\t%d\t%.0f\t%d\n", "hit", b, m[b], s[b] / k[b], k[b]; else c++ }
          if (c) printf "clean\t%d\t%d\t%.0f\n", c, M, S / T }' "$d/vcore.txt" | sort |
  while IFS=$'\t' read -r kind a b c e; do
    if [ "$kind" = hit ]; then printf 'NUMERIC VCORE %s (tray %s): min %s mV, mean %s mV over %s samples, heavy LoFi bf8 numeric hit (stage 5 numeric verify)\n' "$a" "$(rk_tray "$a")" "$b" "$c" "$e"
    else printf 'NUMERIC VCORE %s chips with no numeric hit: min %s mV, mean %s mV (stage 5 numeric verify: heavy chunk)\n' "$a" "$b" "$c"; fi
  done
  local hm=${FLEET_VCORE_HIGH_MV:-30} t; [[ $hm =~ ^[0-9]{1,4}$ ]] || hm=30; local -a tm=()
  for t in $(awk '{ print $2 }' "$d/vcore.txt" | sort -u); do tm+=("$t=$(rk_tray "${t#0000:}")"); done
  python3 "$VC_PY" "$d/vcore.txt" "${3:--}" "$hm" "${tm[@]}" 2> /dev/null | tee "$d/vcore_chips.txt"
}
