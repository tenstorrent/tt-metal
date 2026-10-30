// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for both add_rmsnorm_fw phases.
//   phase A: the slice's sum tiles, then one fp32 partial tile (page r*S + s)
//   phase B (APPLY): the row's rms tile (page r, written by slice 0 only), then the slice's output tiles

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/add_rmsnorm_fw_allcore/device/kernels/add_rmsnorm_fw_allcore_cbs.hpp"

namespace cb = add_rmsnorm_fw_cb;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t S = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);

// Waits for `wait` tiles and writes the first `n`: the compute kernel emits full `block` pushes (the tail of a
// slice padded with unused tiles), so the pops here never straddle the end of the CB.
template <typename Accessor>
FORCE_INLINE void write_block_async(
    uint32_t cb_id, const Accessor& acc, uint32_t first_page, uint32_t n, uint32_t tile_bytes, uint32_t wait) {
    cb_wait_front(cb_id, wait);
    uint32_t l1 = get_read_ptr(cb_id);
    for (uint32_t j = 0; j < n; ++j) {
        noc_async_write_page(first_page + j, acc, l1);
        l1 += tile_bytes;
    }
}

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t out0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t out1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    constexpr auto out0_args = TensorAccessorArgs<4>();
    constexpr auto out1_args = TensorAccessorArgs<out0_args.next_compile_time_args_offset()>();
    const auto out0_acc = TensorAccessor(out0_args, out0_addr);  // sum (A) / out (B)
    const auto out1_acc = TensorAccessor(out1_args, out1_addr);  // partials (A) / rms (B)
#ifdef APPLY
    constexpr uint32_t cb_main = cb::out;
#else
    constexpr uint32_t cb_main = cb::sum_out;
#endif
    const uint32_t tile_bytes = get_tile_size(cb_main);

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t r = work / S;
        const uint32_t s = work - r * S;
        const uint32_t col0 = s * St;
        const uint32_t ncols = (col0 + St <= Wt) ? St : (Wt - col0);

#ifdef APPLY
        // Every slice of the row computes the same rms tile; slice 0 writes it.
        cb_wait_front(cb::rms_out, 1);
        if (s == 0) {
            noc_async_write_page(r, out1_acc, get_read_ptr(cb::rms_out));
            noc_async_write_barrier();
        }
        cb_pop_front(cb::rms_out, 1);
#endif

        for (uint32_t c = 0; c < ncols; c += block) {
            const uint32_t n = (c + block <= ncols) ? block : (ncols - c);
            const uint32_t page = r * Wt + col0 + c;
            write_block_async(cb_main, out0_acc, page, n, tile_bytes, block);
            noc_async_write_barrier();
            cb_pop_front(cb_main, block);
        }

#ifndef APPLY
        write_block_async(cb::partial_out, out1_acc, r * S + s, 1, get_tile_size(cb::partial_out), 1);
        noc_async_write_barrier();
        cb_pop_front(cb::partial_out, 1);
#endif
    }
}
