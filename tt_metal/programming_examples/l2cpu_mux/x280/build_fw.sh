#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build the L2CPU mux firmware: build/fw_mux.bin (fw_mux.c + l2cpu_fabric.h + fabric_mbox.h
# from the fabric_forward example + l2cpu_mux_layout.h). Reuses the sibling clang recipe.
set -euo pipefail
cd "$(dirname "$0")"
NOC=../../l2cpu_noc_transfer/x280
FF=../../l2cpu_fabric_forward/x280
exec "${NOC}/build_fw.sh" "$(pwd)/fw_mux.c" "$(pwd)/build/fw_mux" -I"$(pwd)/.." -I"$(realpath "${FF}")"
