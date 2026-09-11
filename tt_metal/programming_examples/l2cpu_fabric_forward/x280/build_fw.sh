#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build the x280 fabric-worker firmware: build/fw_fabric.bin (from fw_fabric.c +
# l2cpu_fabric.h). Reuses the sibling example's verified clang recipe, start.S and
# dram.ld (cached-GDDR execution at 0x4000_3000_0000).
set -euo pipefail
cd "$(dirname "$0")"
SIB=../../l2cpu_noc_transfer/x280
exec "${SIB}/build_fw.sh" "$(pwd)/fw_fabric.c" "$(pwd)/build/fw_fabric" -I"$(pwd)"
