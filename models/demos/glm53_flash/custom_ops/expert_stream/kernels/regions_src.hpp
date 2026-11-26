// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// REGIONS_SRC: the stream / dispatch kernels reading the regions prefill path's per-expert DRAM-interleaved weight
// tensors instead of the stacked bank-sharded copy (custom_ops/expert_stream/regions.py, doc/mesh/regions_decode.md).
//
// A DRAM-interleaved TILE tensor puts tile page p in bank p % banks at byte offset (p / banks) * tile_bytes, so for a
// [K, I] gate / up tensor with nt = I / 32 tile columns and nt % banks == 0 (ppb = nt / banks pairs per bank) tile
// (k, n = bank + banks * q) sits in `bank` at (k * ppb + q) * tile_bytes: this bank's column q is a K run at stride
// ppb tiles (contiguous at ppb = 1, i.e. exactly the stacked layout's column). For a [I, H] down tensor with
// ht = H / 32 output columns and ht % banks == 0 (C = ht / banks columns per bank) tile (i, h = bank + banks * m) sits
// in `bank` at (i * C + m) * tile_bytes: a core's `cols` columns of one K tile are one contiguous packet (k-major).
// Every expert's buffer address comes from the common runtime args (gate 0..E-1, up E..2E-1; down 0..E-1), the
// bank's NOC base from get_noc_addr_from_bank_id<true>(bank, 0) (the bank_to_dram_offset the address generator adds).
//
// The column bookkeeping below is shared by the writers (where a (bank, q) pair lands in `act`, which output tile
// column a down core's c-th column is); the host mirrors are RegionsStreamPlan.act_tile_column / out_tile_column.

#pragma once

#include <stdint.h>

// act tile column of this bank's pair q under the regions-sourced layout (the stacked layout: bank * ppb + q).
inline uint32_t regions_act_col(uint32_t bank, uint32_t q, uint32_t banks) { return bank + banks * q; }

// output tile column of a down core's c-th column from today's col0 = bank * C + r * cols (the stacked layout:
// col0 + c): bank + banks * (r * cols + c).
inline uint32_t regions_out_col(uint32_t col0, uint32_t C, uint32_t banks, uint32_t c) {
    const uint32_t bank = col0 / C;
    const uint32_t m0 = col0 - bank * C;
    return bank + banks * (m0 + c);
}
