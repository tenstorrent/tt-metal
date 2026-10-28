// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for both rmsnorm_fw phases. Phase A writes one fp32 partial tile per work item (page r*S + s);
// phase B (APPLY) streams y back to the slice's pages and, with RETURN_RMS, writes the row's rms tile
// (page r of the [B, N, S, 1] rms tensor) from the slice-0 item.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/rmsnorm_fw/device/kernels/rmsnorm_fw_cbs.hpp"

namespace cb = rmsnorm_fw_cb;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t S = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);

// Waits for `avail` tiles (a whole `block` for streamed outputs; a CB only wraps cleanly when every push has
// the same size) and writes the first `n` of them.
template <typename Accessor>
FORCE_INLINE void write_tiles_async(
    uint32_t cb_id, const Accessor& acc, uint32_t first_page, uint32_t n, uint32_t avail, uint32_t tile_bytes) {
    cb_wait_front(cb_id, avail);
    uint32_t l1 = get_read_ptr(cb_id);
    for (uint32_t j = 0; j < n; ++j) {
        noc_async_write_page(first_page + j, acc, l1);
        l1 += tile_bytes;
    }
}

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t out0_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t out1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    constexpr auto out0_args = TensorAccessorArgs<4>();
    const auto out0_acc = TensorAccessor(out0_args, out0_addr);
#if defined(APPLY) && defined(RETURN_RMS)
    constexpr auto out1_args = TensorAccessorArgs<out0_args.next_compile_time_args_offset()>();
    const auto out1_acc = TensorAccessor(out1_args, out1_addr);
#endif

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t r = work / S;
        const uint32_t s = work - r * S;

#ifdef APPLY
        const uint32_t tile_bytes = get_tile_size(cb::y);
#ifdef RETURN_RMS
        if (s == 0) {
            write_tiles_async(cb::rms_out, out1_acc, r, 1, 1, tile_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb::rms_out, 1);
        }
#endif
        const uint32_t col0 = s * St;
        const uint32_t ncols = (col0 + St <= Wt) ? St : (Wt - col0);
        for (uint32_t c = 0; c < ncols; c += block) {
            const uint32_t n = (c + block <= ncols) ? block : (ncols - c);
            const uint32_t page = r * Wt + col0 + c;
            write_tiles_async(cb::y, out0_acc, page, n, block, tile_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb::y, block);
        }
#else
        const uint32_t tile_bytes = get_tile_size(cb::partial_out);
        write_tiles_async(cb::partial_out, out0_acc, r * S + s, 1, 1, tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb::partial_out, 1);
#endif
    }
}
