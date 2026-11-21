// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// LocalRoutedUntilize reader: reads this chip's per-expert counts / region offsets (indexed by global id) and the
// local slot -> global id table, enumerates the tile-rows of the local expert regions that hold real rows (expert
// order, ceil(count / 32) per expert) and takes every N_CORES-th of them, starting at `me`. It first hands the compute
// kernel its unit count (one uint32 in cb_n), then streams each unit's tile-row of the source (bf8 TILE [M, H]) into
// cb_in, BLOCK_CT tiles at a time. The writer walks the same unit list.
//
// compile-time args: [cb_in, cb_n, cb_meta, E, E_LOC, H_T, BLOCK_CT, N_CORES, VEC_PAGE, SRC_PAGE]
//                    + TensorAccessorArgs(src, counts, offsets, idx)
// runtime args: [src, counts, offsets, idx addresses, me]
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_n = get_compile_time_arg_val(1);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(2);
    constexpr uint32_t E = get_compile_time_arg_val(3);
    constexpr uint32_t E_LOC = get_compile_time_arg_val(4);
    constexpr uint32_t H_T = get_compile_time_arg_val(5);
    constexpr uint32_t BLOCK_CT = get_compile_time_arg_val(6);
    constexpr uint32_t N_CORES = get_compile_time_arg_val(7);
    constexpr uint32_t VEC_PAGE = get_compile_time_arg_val(8);
    constexpr uint32_t SRC_PAGE = get_compile_time_arg_val(9);
    constexpr auto src_args = TensorAccessorArgs<10>();
    constexpr auto counts_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr uint32_t NB = H_T / BLOCK_CT;
    static_assert(H_T % BLOCK_CT == 0, "pack_untilize needs equal blocks");

    const auto s_src = TensorAccessor(src_args, get_arg_val<uint32_t>(0), SRC_PAGE);
    const auto s_counts = TensorAccessor(counts_args, get_arg_val<uint32_t>(1), VEC_PAGE);
    const auto s_offsets = TensorAccessor(offsets_args, get_arg_val<uint32_t>(2), VEC_PAGE);
    const auto s_idx = TensorAccessor(idx_args, get_arg_val<uint32_t>(3), E_LOC * 4);
    const uint32_t me = get_arg_val<uint32_t>(4);

    // meta scratch: counts page | offsets page | idx table (64-byte aligned regions)
    const uint32_t meta = get_write_ptr(cb_meta);
    constexpr uint32_t OFF_OFS = (VEC_PAGE + 63) & ~63u;
    constexpr uint32_t OFF_IDX = 2 * OFF_OFS;
    noc_async_read(s_counts.get_noc_addr(0), meta, VEC_PAGE);
    noc_async_read(s_offsets.get_noc_addr(0), meta + OFF_OFS, VEC_PAGE);
    noc_async_read(s_idx.get_noc_addr(0), meta + OFF_IDX, E_LOC * 4);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);
    volatile tt_l1_ptr uint32_t* offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta + OFF_OFS);
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta + OFF_IDX);

    uint32_t n_my = 0, u = 0;
    for (uint32_t e = 0; e < E_LOC; ++e) {
        const uint32_t g = idx[e];
        const uint32_t units = g < E ? (counts[g] + 31) >> 5 : 0;
        for (uint32_t j = 0; j < units; ++j, ++u) {
            n_my += (u % N_CORES) == me ? 1 : 0;
        }
    }
    cb_reserve_back(cb_n, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_n))[0] = n_my;
    cb_push_back(cb_n, 1);

    constexpr uint32_t tile_bytes = SRC_PAGE;
    u = 0;
    for (uint32_t e = 0; e < E_LOC; ++e) {
        const uint32_t g = idx[e];
        const uint32_t units = g < E ? (counts[g] + 31) >> 5 : 0;
        const uint32_t tr0 = g < E ? offsets[g] >> 5 : 0;
        for (uint32_t j = 0; j < units; ++j, ++u) {
            if ((u % N_CORES) != me) {
                continue;
            }
            const uint32_t page0 = (tr0 + j) * H_T;
            for (uint32_t b = 0; b < NB; ++b) {
                cb_reserve_back(cb_in, BLOCK_CT);
                const uint32_t l1 = get_write_ptr(cb_in);
                for (uint32_t i = 0; i < BLOCK_CT; ++i) {
                    noc_async_read(s_src.get_noc_addr(page0 + b * BLOCK_CT + i), l1 + i * tile_bytes, tile_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_in, BLOCK_CT);
            }
        }
    }
}
