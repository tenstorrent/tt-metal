#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# fleet/on_hang.sh: TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE hook (and run.sh --mode's beat-watchdog fallback). Runs
# while the hung process is still alive: noc_snapshot.sh (lspci -vvv of each root port and endpoint, dmesg tail,
# tt-triage NOC0 then NOC1), then returns so the op-timeout exception ends the run and run.sh's normal reset still
# runs. Once per FLEET_HANG_DIR (a second timeout during teardown does nothing). Never resets or writes a chip.
# env: FLEET_HANG_DIR (where the snapshot goes), FLEET_HANG_BDFS (endpoints, space separated; default all chips),
#      FLEET_HANG_CAP_S (whole snapshot cap, default 180), TRAY_REPRO_SYS
D=${FLEET_HANG_DIR:?}; SYS=${TRAY_REPRO_SYS:-/sys}; HERE=$(dirname "$(readlink -f "$0")")
mkdir -p "$D" && mkdir "$D/.hook" 2>/dev/null || exit 0
date '+%F %T %Z' > "$D/HANG"
pairs=()
for b in ${FLEET_HANG_BDFS:-$(ls "$SYS"/bus/pci/devices/ 2>/dev/null)}; do
  [ "$(cat "$SYS/bus/pci/devices/$b/vendor" 2>/dev/null)" = 0x1e52 ] || continue
  rp=$(basename "$(dirname "$(readlink -f "$SYS/bus/pci/devices/$b")")"); pairs+=("$rp=$b")
done
cap=${FLEET_HANG_CAP_S:-180}
SNAP_TRIAGE_TIMEOUT_S=${SNAP_TRIAGE_TIMEOUT_S:-$(( (cap - 20) / 2 ))} timeout -k 5 "$cap" bash "$HERE/../noc_snapshot.sh" "$D/snapshot" "${pairs[@]}" > "$D/snapshot.log" 2>&1
echo "rc $? at $(date '+%F %T %Z')" >> "$D/HANG"
exit 0
