// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for gated_rmsnorm fw/bw: streams the Gt output tiles of each (tile-row, group) work item back to
// pages r*Wt + h*Gt + j. Forward writes `out`; backward writes dx (cb::out), dgate and, with
// COMPUTE_DGAMMA, the unreduced dgamma components.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/gated_rmsnorm/device/kernels/gated_rmsnorm_cbs.hpp"

namespace cb = gated_rmsnorm_cb;

constexpr uint32_t Gt = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t num_groups = get_compile_time_arg_val(2);

template <typename Accessor>
FORCE_INLINE void write_group(uint32_t cb_id, const Accessor& acc, uint32_t first_page, uint32_t tile_bytes) {
    cb_wait_front(cb_id, Gt);
    uint32_t l1 = get_read_ptr(cb_id);
    for (uint32_t j = 0; j < Gt; ++j) {
        noc_async_write_page(first_page + j, acc, l1);
        l1 += tile_bytes;
    }
}

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t dgate_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t dgamma_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    constexpr auto out_args = TensorAccessorArgs<3>();
    const auto out_acc = TensorAccessor(out_args, out_addr);
#ifdef BACKWARD
    constexpr auto dgate_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto dgate_acc = TensorAccessor(dgate_args, dgate_addr);
#ifdef COMPUTE_DGAMMA
    constexpr auto dgamma_args = TensorAccessorArgs<dgate_args.next_compile_time_args_offset()>();
    const auto dgamma_acc = TensorAccessor(dgamma_args, dgamma_addr);
#endif
#endif

    const uint32_t tile_bytes = get_tile_size(cb::out);

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t r = work / num_groups;
        const uint32_t h = work - r * num_groups;
        const uint32_t first_page = r * Wt + h * Gt;

        write_group(cb::out, out_acc, first_page, tile_bytes);
#ifdef BACKWARD
        write_group(cb::dgate, dgate_acc, first_page, tile_bytes);
#ifdef COMPUTE_DGAMMA
        write_group(cb::dgamma, dgamma_acc, first_page, tile_bytes);
#endif
#endif
        noc_async_write_barrier();
        cb_pop_front(cb::out, Gt);
#ifdef BACKWARD
        cb_pop_front(cb::dgate, Gt);
#ifdef COMPUTE_DGAMMA
        cb_pop_front(cb::dgamma, Gt);
#endif
#endif
    }
}
