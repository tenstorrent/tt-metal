// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The [gate | up] tile layout the swiglu_packed kernels share: a packed row is 2*Wt tiles, gate in
// [0, Wt) and up in [Wt, 2*Wt), and work is handed out in blocks of block_size tiles along a row.

#pragma once

#include <cstdint>

inline uint32_t swiglu_packed_gate_tile(uint32_t b, uint32_t Wt, uint32_t block_size) {
    const uint32_t blocks_per_row = Wt / block_size;
    return (b / blocks_per_row) * (2U * Wt) + (b % blocks_per_row) * block_size;
}
