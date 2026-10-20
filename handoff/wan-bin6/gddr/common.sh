# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for the GDDR soft-harvest scripts. Source it; don't run it.
# Command shapes follow the Exabox page "GDDR Harvesting — Blackhole Galaxy Bin6 Command Reference".
#
# Chips are keyed by ASIC ID (/sys/class/tenstorrent/*/tt_asic_id), not by /dev/tenstorrent/N, because
# the device numbering is not guaranteed to survive a reset. The current device node is looked up through
# /dev/tenstorrent/by-id/blackhole-<asic_id> every time it is needed.

BHMOD="${BHMOD:-/data/local-syseng/dc-deploy-pkgs/bh-mod}"
SUDO="${SUDO-sudo}"   # set SUDO= (empty) to run bh-mod without sudo
EXPECTED_CHIPS="${EXPECTED_CHIPS:-32}"
GDDR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$GDDR_DIR/../../.." && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date +%T)] $*"; }

require_bhmod() {
    [[ -x "$BHMOD" ]] || die "bh-mod not found at $BHMOD (set BHMOD=...)"
}

# Refuse to touch the board while anything else is using it.
require_idle() {
    local busy
    busy=$(ps -eo user,pid,cmd | grep -E "[p]ytest|[t]t-smi|[t]t-triage|[b]h-mod " | grep -v "$$" || true)
    [[ -z "$busy" ]] || die "board looks busy, stop these first:
$busy"
}

# One line per chip: "<asic_id> <pci_bdf> <dev_node>", sorted by PCI address.
list_chips() {
    local d asic bdf
    for d in /sys/class/tenstorrent/tenstorrent\!*; do
        asic=$(cat "$d/tt_asic_id")
        bdf=$(basename "$(readlink -f "$d/device")")
        echo "$asic $bdf $(dev_for_asic "$asic")"
    done | sort -k2
}

dev_for_asic() {
    local link="/dev/tenstorrent/by-id/blackhole-$1"
    [[ -e "$link" ]] || die "no device node for ASIC $1 (missing $link)"
    readlink -f "$link"
}

# Print "<soft_harvest_dram_mask> <tdp_limit>" for one device node. A field without an override is at its
# firmware default (mask 0, TDP 130 W on this firmware), so it doesn't appear in `get --delta`.
read_overrides() {
    local out mask tdp
    out=$($SUDO "$BHMOD" -d "$1" get --delta 2>/dev/null | sed 's/\xe2\x94\x82/|/g')
    mask=$(awk -F'|' '$2 ~ /soft_harvest_dram_mask/ {gsub(/ /, "", $4); print $4}' <<<"$out")
    tdp=$(awk -F'|' '$2 ~ /chip_limits.tdp_limit/ {gsub(/ /, "", $4); print $4}' <<<"$out")
    echo "${mask:-0} ${tdp:-default}"
}

is_one_channel_mask() {
    case "$1" in 1|2|4|8|16|32|64|128) return 0 ;; *) return 1 ;; esac
}

glx_reset() {
    log "tt-smi -glx_reset (IPMI power reset of all chips, 1-2 min)"
    $SUDO tt-smi -glx_reset
}

# The snapshot is TSV: asic_id, pci_bdf, dev_at_snapshot, soft_harvest_dram_mask, tdp_limit.
read_snapshot() {
    [[ -f "$1" ]] || die "snapshot not found: $1"
    grep -v '^#' "$1"
}
