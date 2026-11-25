// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Combine consume microbench, reader (BRISC): the accumulator reader's per-row work, from landed rows in L1.
//
// At the start (not timed) the P wire rows of the input tensor are read into a pool in L1 (the landing slots). Then
// n_rows rows are handed to the compute, row n from slot n % P, for token n % tokens, as the accumulator reader does:
//   * bf16 (fp8 = 0): the row's data (row_tiles pages of 2048 B) into cb_row; its gate (trailer word 1, bf16 bits)
//     into element 0 of a cb_w tile (the compute broadcasts it).
//   * fp8 (fp8 = 1): the row's codes (row_tiles pages of 1024 B) into cb_row; for each flat tile p (1024 codes) an FP16
//     factor tile into cb_fac (FP16: fp8 unpacks to FP16 in srcA and the FPU takes one exponent family), only its
//     column 0 used (the compute broadcasts each of its 64 face rows' column 0 over the row): face row r's factor =
//     the scale of codes 16r .. 16r + 15 (its e8m0 byte) x 2^10 (kFactorShift: FP16 range; the snap undoes it). The
//     gate goes with the meta.
//   * meta (cb_meta, a page a row): word 0 the token, word 1 the gate's bf16 bits.
//
// Wire row (fp8): [data_bytes codes][cells: 16 B a down core: 12 exponent bytes (its 192 columns / 16) + 4 spare]
//                 [32-byte trailer: word 1 the gate's bf16 bits]
//
// Compile-time args: fp8, n_rows, pool_slots, slot_bytes, data_bytes, row_tiles, tokens, cells_per_row, build_factors,
// then the input's TensorAccessorArgs. Runtime args: the input's address.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr bool fp8 = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t n_rows = get_compile_time_arg_val(1);
    constexpr uint32_t pool_slots = get_compile_time_arg_val(2);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t data_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t tokens = get_compile_time_arg_val(6);
    constexpr uint32_t cells = get_compile_time_arg_val(7);
    constexpr uint32_t cb_row = 0, cb_w = 1, cb_fac = 2, cb_meta = 3, cb_pool = 10;
    constexpr uint32_t kFacTileBytes = 2048;  // an FP16 tile
    constexpr uint32_t kCellBytes = 12;       // a cell's exponent bytes (192 codes / 16)
    constexpr int32_t kFactorShift = 10;      // the factor is the scale x 2^10 (in FP16 range; the snap undoes it)
    static_assert(!fp8 || cells * kCellBytes == row_tiles * 64, "a cell's exponents per 192 codes");

    // (perf only: 0 keeps the factor tiles as they are - the compute alone, the output not checked)
    constexpr bool build_factors = get_compile_time_arg_val(8) != 0;
    constexpr auto in_args = TensorAccessorArgs<9>();
    const auto in = TensorAccessor(in_args, get_arg_val<uint32_t>(0), slot_bytes);
    // e8m0 byte -> the FP16 factor's bits: 2^(E - 127) x 2^kFactorShift, exponent field clamped to FP16's normal range
    // (exact for the snap: the bench's docstring)
    uint16_t lut[256];
    for (int32_t b = 0; b < 256; ++b) {
        int32_t e = b - 127 + kFactorShift + 15;
        e = e < 1 ? 1 : (e > 30 ? 30 : e);
        lut[b] = static_cast<uint16_t>(e << 10);
    }
    // the landing slots
    const uint32_t pool = get_write_ptr(cb_pool);
    for (uint32_t s = 0; s < pool_slots; ++s) {
        noc_async_read(in.get_noc_addr(s), pool + s * slot_bytes, slot_bytes);
    }
    noc_async_read_barrier();
    if constexpr (fp8) {  // the factor tiles' unused columns: 0 (finite), once, over the whole CB
        constexpr uint32_t fac_pages = 2 * row_tiles;
        cb_reserve_back(cb_fac, fac_pages);
        volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_fac));
        for (uint32_t w = 0; w < fac_pages * kFacTileBytes / 4; ++w) {
            f[w] = 0;
        }
    }
    {
        DeviceZoneScopedN("CMB-RD");
        for (uint32_t n = 0; n < n_rows; ++n) {
            const uint32_t slot = pool + (n % pool_slots) * slot_bytes;
            volatile tt_l1_ptr uint32_t* trailer =  // (after the data and the cells)
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + data_bytes + (fp8 ? cells * 16 : 0));
            const uint32_t gate = trailer[1] & 0xFFFFu;
            cb_reserve_back(cb_row, row_tiles);
            noc_async_read(get_noc_addr(slot), get_write_ptr(cb_row), data_bytes);
            if constexpr (fp8) {
                cb_reserve_back(cb_fac, row_tiles);
                // exponent byte k (codes 16k .. 16k + 15) = cell k / 12, byte k % 12; its factor: flat tile k / 64,
                // face row k % 64, column 0 = halfword 16k of the row's factor tiles (a tile is 64 face rows x 32 B)
                if constexpr (build_factors) {
                    volatile tt_l1_ptr uint32_t* cell =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + data_bytes);
                    volatile tt_l1_ptr uint16_t* f =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_fac));
                    for (uint32_t c = 0; c < cells; ++c, cell += 4) {
#pragma GCC unroll 3
                        for (uint32_t j = 0; j < kCellBytes / 4; ++j, f += 64) {
                            const uint32_t w = cell[j];
                            f[0] = lut[w & 0xFF];
                            f[16] = lut[(w >> 8) & 0xFF];
                            f[32] = lut[(w >> 16) & 0xFF];
                            f[48] = lut[w >> 24];
                        }
                    }
                }
                cb_push_back(cb_fac, row_tiles);
            }
            {  // (the gate tile: the bf16 path's srcB; the fp8 path's debug mode 2)
                cb_reserve_back(cb_w, 1);
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w))[0] = gate;
                cb_push_back(cb_w, 1);
            }
            cb_reserve_back(cb_meta, 1);
            volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta));
            meta[0] = n % tokens;
            meta[1] = gate;
            noc_async_read_barrier();
            cb_push_back(cb_row, row_tiles);
            cb_push_back(cb_meta, 1);
        }
    }
}
