#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Remove the GDDR soft harvest on every chip (back to 8 channels), reset the Galaxy, and verify.
# Leaves the TDP override alone: pick the power per run with TT_METAL_TDP_LIMIT_WATTS instead.
#
#   handoff/wan-bin6/gddr/unharvest.sh <snapshot.tsv>            # dry run: print what would change
#   handoff/wan-bin6/gddr/unharvest.sh <snapshot.tsv> --apply    # do it
#
# The snapshot is required so that there is always a way back (restore.sh <snapshot.tsv> --apply).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_bhmod

snap="${1:?usage: unharvest.sh <snapshot.tsv> [--apply]}"
apply=0; [[ "${2:-}" == "--apply" ]] && apply=1
read_snapshot "$snap" > /dev/null
grep -q "host=$(hostname) " "$snap" || die "snapshot $snap was not taken on $(hostname)"

if [[ $apply -eq 0 ]]; then
    log "DRY RUN (add --apply to change the board)"
    while read -r asic bdf dev; do
        echo "$SUDO $BHMOD -d $dev res dram_table.soft_harvest_dram_mask    # ASIC $asic ($bdf), currently mask $(read_overrides "$dev" | cut -d' ' -f1)"
    done < <(list_chips)
    echo "$SUDO tt-smi -glx_reset"
    echo "$GDDR_DIR/verify.sh unharvested"
    exit 0
fi

require_idle
while read -r asic bdf dev; do
    log "ASIC $asic ($bdf, $dev): reset soft_harvest_dram_mask to default"
    $SUDO "$BHMOD" -d "$dev" res dram_table.soft_harvest_dram_mask
done < <(list_chips)
glx_reset
"$GDDR_DIR/verify.sh" unharvested
log "Done. To go back to Bin6: $GDDR_DIR/restore.sh $snap --apply"
