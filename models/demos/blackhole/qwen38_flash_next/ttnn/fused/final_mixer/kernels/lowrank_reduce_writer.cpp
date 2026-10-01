// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Final mixer stage 3a writer, one core: the T bf16 low-rank tiles (CB 16) -> the output pages 0..T-1.
// Compile-time args: 0 T, then TensorAccessorArgs(out).  Runtime args: 0 out addr.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t T = get_compile_time_arg_val(0);
constexpr uint32_t BF16_TILE_BYTES = 2048;
constexpr uint32_t c_lr = 16;

void kernel_main() {
    constexpr auto a_o = TensorAccessorArgs<1>();
    const auto out = TensorAccessor(a_o, get_arg_val<uint32_t>(0));
    Noc noc;
    DataflowBuffer lr(c_lr);
    {
        FUSED_ZONE("fz_fm_lr_w_main");
        lr.wait_front(T);
        for (uint32_t t = 0; t < T; ++t) {
            noc.async_write(
                lr, out, BF16_TILE_BYTES, {.offset_bytes = t * BF16_TILE_BYTES}, {.page_id = t, .offset_bytes = 0});
        }
        noc.async_write_barrier();
        lr.pop_front(T);
    }
}
