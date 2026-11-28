// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// LocalRoutedUntilize writer: walks the reader's unit list (its own copy of the counts / offsets / idx pages) and
// writes the real rows of every unit compute untilized (min(32, count - 32 j) of the tile-row's 32) to the same rows of
// the ROW_MAJOR destination [M, H] bf16; padding rows of a region and rows outside the regions are not written.
//
// compile-time args: [cb_out, cb_meta, E, E_LOC, N_CORES, VEC_PAGE, ROW_BYTES, DST_PAGE]
//                    + TensorAccessorArgs(dst, counts, offsets, idx)
// runtime args: [dst, counts, offsets, idx addresses, me]
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t E = get_compile_time_arg_val(2);
    constexpr uint32_t E_LOC = get_compile_time_arg_val(3);
    constexpr uint32_t N_CORES = get_compile_time_arg_val(4);
    constexpr uint32_t VEC_PAGE = get_compile_time_arg_val(5);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(6);
    constexpr uint32_t DST_PAGE = get_compile_time_arg_val(7);
    constexpr auto dst_args = TensorAccessorArgs<8>();
    constexpr auto counts_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();

    const auto s_dst = TensorAccessor(dst_args, get_arg_val<uint32_t>(0), DST_PAGE);
    const auto s_counts = TensorAccessor(counts_args, get_arg_val<uint32_t>(1), VEC_PAGE);
    const auto s_offsets = TensorAccessor(offsets_args, get_arg_val<uint32_t>(2), VEC_PAGE);
    const auto s_idx = TensorAccessor(idx_args, get_arg_val<uint32_t>(3), E_LOC * 4);
    const uint32_t me = get_arg_val<uint32_t>(4);

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

    uint32_t u = 0;
    for (uint32_t e = 0; e < E_LOC; ++e) {
        const uint32_t g = idx[e];
        const uint32_t count = g < E ? counts[g] : 0;
        const uint32_t units = (count + 31) >> 5;
        const uint32_t row0 = g < E ? offsets[g] : 0;
        for (uint32_t j = 0; j < units; ++j, ++u) {
            if ((u % N_CORES) != me) {
                continue;
            }
            const uint32_t rows = (count - 32 * j) < 32 ? (count - 32 * j) : 32;
            cb_wait_front(cb_out, 32);
            const uint32_t l1 = get_read_ptr(cb_out);
            for (uint32_t r = 0; r < rows; ++r) {
                noc_async_write(l1 + r * ROW_BYTES, s_dst.get_noc_addr(row0 + 32 * j + r), ROW_BYTES);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, 32);
        }
    }
}
