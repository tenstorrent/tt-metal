// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE stage 3 writer, one core: the Vt gated tiles (CB 16, bf16) -> pages 0..Vt-1 of the output; with debug, the
// fp32 sum tile (CB 17) and coefficient tile (CB 18) -> page 0 of two fp32 tile tensors.
// Compile-time args: 0 Vt, 1 debug, then TensorAccessorArgs(out), (debug sum), (debug coefficient).
// Runtime args: 0 out addr, 1 debug sum addr, 2 debug coefficient addr, 3 the first output tile (the lane's Vt block; 0
// for the 1-row form).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t Vt = get_compile_time_arg_val(0);
constexpr uint32_t DEBUG = get_compile_time_arg_val(1);
constexpr uint32_t BF16_TILE = 2048, FP32_TILE = 4096;
constexpr uint32_t c_out = 16, c_dbg_sum = 17, c_dbg_coef = 18;

void kernel_main() {
    constexpr auto a_out = TensorAccessorArgs<2>();
    constexpr auto a_sum = TensorAccessorArgs<a_out.next_compile_time_args_offset()>();
    constexpr auto a_coef = TensorAccessorArgs<a_sum.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(a_out, get_arg_val<uint32_t>(0));
    const uint32_t first = get_arg_val<uint32_t>(3);
    Noc noc;
    if constexpr (DEBUG) {
        const auto dsum = TensorAccessor(a_sum, get_arg_val<uint32_t>(1));
        const auto dcoef = TensorAccessor(a_coef, get_arg_val<uint32_t>(2));
        DataflowBuffer s(c_dbg_sum), c(c_dbg_coef);
        s.wait_front(1);
        noc.async_write(s, dsum, FP32_TILE, {.offset_bytes = 0}, {.page_id = 0, .offset_bytes = 0});
        c.wait_front(1);
        noc.async_write(c, dcoef, FP32_TILE, {.offset_bytes = 0}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
        s.pop_front(1);
        c.pop_front(1);
    }
    {
        FUSED_ZONE("fz_pl_gate_w_main");
        DataflowBuffer o(c_out);
        o.wait_front(Vt);
        for (uint32_t t = 0; t < Vt; ++t) {
            noc.async_write(
                o, out, BF16_TILE, {.offset_bytes = t * BF16_TILE}, {.page_id = first + t, .offset_bytes = 0});
        }
        noc.async_write_barrier();
        o.pop_front(Vt);
    }
}
