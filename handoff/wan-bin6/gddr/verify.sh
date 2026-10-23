#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Read-only. Check the board after a harvest change and a reset:
#   1. bh-mod masks match what is expected (per ASIC ID),
#   2. all chips are on PCIe,
#   3. tt-metal sees the expected DRAM bank count (dram_grid_size().x: 8 unharvested, 7 harvested),
#   4. the ethernet links still form the 4x8 torus.
#
#   handoff/wan-bin6/gddr/verify.sh unharvested        # expect mask 0 everywhere, 8 banks
#   handoff/wan-bin6/gddr/verify.sh <snapshot.tsv>     # expect the snapshot's masks
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_bhmod

want="${1:?usage: verify.sh unharvested|<snapshot.tsv>}"
declare -A expect
if [[ "$want" == unharvested ]]; then
    while read -r asic _ _; do expect[$asic]=0; done < <(list_chips)
else
    while IFS=$'\t' read -r asic _ _ mask _; do expect[$asic]=$mask; done < <(read_snapshot "$want")
fi

fail=0
log "1/4 bh-mod soft_harvest_dram_mask per chip"
n=0
while read -r asic bdf dev; do
    read -r mask tdp < <(read_overrides "$dev")
    n=$((n + 1))
    if [[ "${expect[$asic]:-missing}" != "$mask" ]]; then
        echo "  MISMATCH ASIC $asic ($bdf): mask $mask, expected ${expect[$asic]:-<not in snapshot>}"
        fail=1
    fi
done < <(list_chips)
echo "  $n chips checked"

log "2/4 PCIe enumeration"
pcie=$(lspci -d 1e52: | wc -l)
echo "  $pcie Tenstorrent PCIe devices"
[[ "$pcie" -eq "$EXPECTED_CHIPS" ]] || { echo "  expected $EXPECTED_CHIPS"; fail=1; }

log "3/4 DRAM banks seen by tt-metal"
expected_banks=8
if [[ "$want" != unharvested ]] && [[ $(read_snapshot "$want" | cut -f4 | grep -cv '^0$' || true) -gt 0 ]]; then
    expected_banks=7
fi
banks=$(cd "$REPO" && source python_env/bin/activate && TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" python3 -c "
import ttnn
d = ttnn.open_mesh_device(ttnn.MeshShape(4, 8))
print('DRAM_BANKS=%d' % d.dram_grid_size().x)
ttnn.close_mesh_device(d)" 2>/dev/null | sed -n 's/^DRAM_BANKS=//p')
echo "  dram_grid_size().x = $banks (expected $expected_banks)"
[[ "$banks" == "$expected_banks" ]] || fail=1

log "4/4 ethernet links"
links=$(cd "$REPO" && source python_env/bin/activate && python3 -I "$GDDR_DIR/../check_links.py" 2>/dev/null | grep "^T" || true)
if [[ -n "$links" ]]; then
    echo "$links" | sed 's/^/  /'
    # Ports 8-9 are not cabled on this Galaxy; anything else down is a problem.
    down=$(grep -cv "inactive ports: \[8, 9\] active-but-unconnected: \[\]$" <<<"$links" || true)
    [[ "$down" -eq 0 ]] || { echo "  $down chips have links down: reset again"; fail=1; }
else
    echo "  all ports trained"
fi

[[ $fail -eq 0 ]] && log "PASS" || { log "FAIL (see above). The Exabox page says: stop and escalate if ASICs are missing or the topology does not return."; exit 1; }
