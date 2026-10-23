#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Put every chip's GDDR soft-harvest mask back to the value in a snapshot (matched by ASIC ID), reset the
# Galaxy, and verify. Masks of 0 in the snapshot are reset to the default instead of set.
#
#   handoff/wan-bin6/gddr/restore.sh <snapshot.tsv>            # dry run: bh-mod `set -n` on every chip
#   handoff/wan-bin6/gddr/restore.sh <snapshot.tsv> --apply    # do it
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_bhmod

snap="${1:?usage: restore.sh <snapshot.tsv> [--apply]}"
apply=0; [[ "${2:-}" == "--apply" ]] && apply=1
grep -q "host=$(hostname) " "$snap" || die "snapshot $snap was not taken on $(hostname)"

# Check the whole snapshot before touching anything.
n=0
while IFS=$'\t' read -r asic bdf _ mask _; do
    [[ "$mask" == 0 ]] || is_one_channel_mask "$mask" || die "ASIC $asic: mask $mask is not a single channel"
    dev_for_asic "$asic" > /dev/null
    n=$((n + 1))
done < <(read_snapshot "$snap")
[[ "$n" -eq "$EXPECTED_CHIPS" ]] || die "snapshot has $n chips, expected $EXPECTED_CHIPS"

[[ $apply -eq 1 ]] && require_idle || log "DRY RUN (add --apply to change the board)"
while IFS=$'\t' read -r asic bdf _ mask _; do
    dev=$(dev_for_asic "$asic")
    if [[ "$mask" == 0 ]]; then
        log "ASIC $asic ($bdf, $dev): reset mask to default"
        [[ $apply -eq 1 ]] && $SUDO "$BHMOD" -d "$dev" res dram_table.soft_harvest_dram_mask
    elif [[ $apply -eq 1 ]]; then
        log "ASIC $asic ($bdf, $dev): set mask $mask"
        $SUDO "$BHMOD" -d "$dev" set "dram_table.soft_harvest_dram_mask=$mask"
    else
        log "ASIC $asic ($bdf, $dev): set -n mask $mask"
        $SUDO "$BHMOD" -d "$dev" set -n "dram_table.soft_harvest_dram_mask=$mask"
    fi
done < <(read_snapshot "$snap")

if [[ $apply -eq 0 ]]; then
    echo "then: $SUDO tt-smi -glx_reset && $GDDR_DIR/verify.sh $snap"
    exit 0
fi
glx_reset
"$GDDR_DIR/verify.sh" "$snap"
