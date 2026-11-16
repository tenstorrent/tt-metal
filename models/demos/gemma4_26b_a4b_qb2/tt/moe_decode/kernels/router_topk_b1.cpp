// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// moe_decode batch-1 router tail on one core (data movement only): the E router logits of row 0 (FP32 TILE
// [1, 1, 1, E], E a multiple of 32) -> top-k, softmax over the k picks (equal to softmax over all E, then top-k,
// then renormalized), times per_expert_scale[id], written as the packed routing row RouterTopk.op_packed produces:
// [w_0 .. w_{k-1} | id_0 .. id_{k-1}] bf16 (ids exact below 256). Order: descending logit; equal logits keep the
// lower expert id first. Runs on FP32 logits, so no centering pass is needed for bf16 precision; the softmax is
// integer fixed-point arithmetic (the data-movement cores have no FPU).
// compile-time args: [cb_scr, num_experts, k] + TensorAccessorArgs(logits, pes, out)
// runtime args: [logits_addr, pes_addr, out_addr]
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

#include "router_topk_math.hpp"

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t pes_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t E = get_compile_time_arg_val(1);
    constexpr uint32_t k = get_compile_time_arg_val(2);
    constexpr auto l_args = TensorAccessorArgs<3>();
    constexpr auto p_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr uint32_t f32_tile = 4096, f32_face = 1024, row16 = 64;  // FP32 tile, face, 16-value face row
    constexpr uint32_t tiles = E / 32;
    const auto s_l = TensorAccessor(l_args, logits_addr, f32_tile);
    const auto s_p = TensorAccessor(p_args, pes_addr, E * 2);
    const auto s_o = TensorAccessor(o_args, out_addr, 4 * k);

    // logits row 0: per tile, face 0 row 0 (columns 0-15) then face 1 row 0 (16-31); then the scale row
    const uint32_t scr = get_write_ptr(cb_scr);
    for (uint32_t t = 0; t < tiles; ++t) {
        const uint64_t src = s_l.get_noc_addr(t);
        noc_async_read(src, scr + t * 2 * row16, row16);
        noc_async_read(src + f32_face, scr + t * 2 * row16 + row16, row16);
    }
    const uint32_t pes_l1 = scr + E * 4;
    noc_async_read(s_p.get_noc_addr(0), pes_l1, E * 2);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* logit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    volatile tt_l1_ptr uint16_t* pes = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pes_l1);

    volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + E * 4 + E * 2);
    router_tail::topk_softmax<E, k>(logit, pes, row);
    noc_async_write(scr + E * 4 + E * 2, s_o.get_noc_addr(0), 4 * k);
    noc_async_write_barrier();
}
