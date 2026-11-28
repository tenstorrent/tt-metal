// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer for delta_solve_compute.cpp: per matrix m of this core, the transition's 16 tiles (HAS_ALPHA), then the
// injection's, to pages 16 m .. 16 m + 15 of each output.
//
// Compile-time args: has_alpha, then the tensor accessor args of transition, injection.
// Runtime args: transition, injection addresses, first matrix, number of matrices.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_tr = 16;
constexpr uint32_t cb_inj = 17;
constexpr uint32_t TILES = 16;
}  // namespace

void kernel_main() {
    const uint32_t tr_addr = get_arg_val<uint32_t>(0);
    const uint32_t inj_addr = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    constexpr bool has_alpha = get_compile_time_arg_val(0) != 0;
    constexpr auto tr_args = TensorAccessorArgs<1>();
    constexpr auto inj_args = TensorAccessorArgs<tr_args.next_compile_time_args_offset()>();

    Noc noc;
    DataflowBuffer tr_buf(cb_tr);
    DataflowBuffer inj_buf(cb_inj);
    const uint32_t tile_bytes = inj_buf.get_entry_size();
    const auto tr = TensorAccessor(tr_args, tr_addr);
    const auto inj = TensorAccessor(inj_args, inj_addr);

    const auto drain = [&](DataflowBuffer& buf, const auto& dst, uint32_t first_page) {
        for (uint32_t t = 0; t < TILES; ++t) {
            buf.wait_front(1);
            noc.async_write(buf, dst, tile_bytes, {}, {.page_id = first_page + t});
            noc.async_writes_flushed();
            buf.pop_front(1);
        }
    };

    for (uint32_t m = first; m < first + count; ++m) {
        if constexpr (has_alpha) {
            drain(tr_buf, tr, m * TILES);
        }
        drain(inj_buf, inj, m * TILES);
    }
    noc.async_write_barrier();
}
