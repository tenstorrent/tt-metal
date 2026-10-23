#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Read-only. Record every chip's GDDR soft-harvest mask and TDP override, plus the evidence the Exabox page
# asks for. Take one BEFORE changing anything: restore.sh needs it to put each mask back on the right chip.
#
#   handoff/wan-bin6/gddr/snapshot.sh [out_dir]      # default: ~/gddr-state/<host>-<timestamp>/
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_bhmod

out="${1:-$HOME/gddr-state/$(hostname)-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$out"
snap="$out/snapshot.tsv"

log "writing $snap"
{
    echo "# host=$(hostname) date=$(date -Is) bh-mod=$($SUDO "$BHMOD" --version 2>/dev/null)"
    echo "# asic_id	pci_bdf	dev_at_snapshot	soft_harvest_dram_mask	tdp_limit"
    while read -r asic bdf dev; do
        read -r mask tdp < <(read_overrides "$dev")
        printf '%s\t%s\t%s\t%s\t%s\n' "$asic" "$bdf" "$dev" "$mask" "$tdp"
    done < <(list_chips)
} > "$snap"

n=$(read_snapshot "$snap" | wc -l)
[[ "$n" -eq "$EXPECTED_CHIPS" ]] || echo "WARNING: found $n chips, expected $EXPECTED_CHIPS" >&2

# Evidence, as listed on the Exabox page.
$SUDO "$BHMOD" --version > "$out/bhmod-version.txt" 2>&1 || true
$SUDO tt-smi -ls > "$out/tt-smi-list.txt" 2>&1 || true
lspci -d 1e52: > "$out/lspci.txt" 2>&1 || true
$SUDO tt-smi -s -f "$out/tt-smi.json" > /dev/null 2>&1 || true

echo
column -t -s $'\t' < <(grep -v '^# host' "$snap")
echo
echo "masks: $(read_snapshot "$snap" | cut -f4 | sort -n | uniq -c | awk '{printf "%s x%s  ", $2, $1}')"
echo "tdp:   $(read_snapshot "$snap" | cut -f5 | sort | uniq -c | awk '{printf "%s x%s  ", $2, $1}')"
echo "PCIe devices: $(wc -l < "$out/lspci.txt")"
echo
echo "Saved to $out"
