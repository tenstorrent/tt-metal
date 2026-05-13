#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Configure 2MB hugepages for test_external_cmac_smoke_mlx (DPDK app).
# Run once per boot, with sudo.
#
# Usage: sudo ./setup_hugepages_mlx_smoke.sh [num_pages]
#   num_pages defaults to 1024 (2 GiB).

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Must be run as root (sudo)." >&2
    exit 1
fi

NUM_PAGES="${1:-1024}"

echo "Reserving $NUM_PAGES × 2MB hugepages on node 0..."
echo "$NUM_PAGES" > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages

if ! mount | grep -q 'hugetlbfs on /mnt/huge'; then
    mkdir -p /mnt/huge
    mount -t hugetlbfs nodev /mnt/huge
    echo "Mounted hugetlbfs at /mnt/huge"
fi

echo
echo "Current state:"
grep -E '^(HugePages_Total|HugePages_Free|Hugepagesize):' /proc/meminfo

echo
echo "DPDK test_external_cmac_smoke_mlx is ready to run."
echo "Make sure the Mellanox interface is up:"
echo "  sudo ip link set enp2s0np0 up"
