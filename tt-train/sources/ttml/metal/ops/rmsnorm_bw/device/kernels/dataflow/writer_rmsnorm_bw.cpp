// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for both rmsnorm_bw phases. Phase A writes one fp32 partial tile per work item (page r*S + s);
// phase B (APPLY) streams dL_da (and dL_dgamma components with COMPUTE_DGAMMA) back to the slice's pages.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/rmsnorm_bw/device/kernels/rmsnorm_bw_cbs.hpp"

namespace cb = rmsnorm_bw_cb;

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t S = get_compile_time_arg_val(1);
constexpr uint32_t St = get_compile_time_arg_val(2);
constexpr uint32_t block = get_compile_time_arg_val(3);

template <typename Accessor>
FORCE_INLINE void write_block_async(
    uint32_t cb_id, const Accessor& acc, uint32_t first_page, uint32_t n, uint32_t tile_bytes) {
    cb_wait_front(cb_id, n);
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
#if defined(APPLY) && defined(COMPUTE_DGAMMA)
    constexpr auto out1_args = TensorAccessorArgs<out0_args.next_compile_time_args_offset()>();
    const auto out1_acc = TensorAccessor(out1_args, out1_addr);
#endif

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t r = work / S;
        const uint32_t s = work - r * S;

#ifdef APPLY
        const uint32_t tile_bytes = get_tile_size(cb::dx);
        const uint32_t col0 = s * St;
        const uint32_t ncols = (col0 + St <= Wt) ? St : (Wt - col0);
        for (uint32_t c = 0; c < ncols; c += block) {
            const uint32_t n = (c + block <= ncols) ? block : (ncols - c);
            const uint32_t page = r * Wt + col0 + c;
            write_block_async(cb::dx, out0_acc, page, n, tile_bytes);
#ifdef COMPUTE_DGAMMA
            write_block_async(cb::dgamma, out1_acc, page, n, tile_bytes);
#endif
            noc_async_write_barrier();
            cb_pop_front(cb::dx, n);
#ifdef COMPUTE_DGAMMA
            cb_pop_front(cb::dgamma, n);
#endif
        }
#else
        const uint32_t tile_bytes = get_tile_size(cb::partial_out);
        write_block_async(cb::partial_out, out0_acc, r * S + s, 1, tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb::partial_out, 1);
#endif
    }
}
