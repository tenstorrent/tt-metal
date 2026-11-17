// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 router tail on the reducer (BRISC) (fused_decode/router.py: RouterB1): row 0 of the E / 32 FP32 logits tiles
// (local) -> top-k, softmax over the picks, times per_expert_scale -> the packed routing row (moe_decode router_topk_b1
// format, moe_decode/kernels/router_topk_math.hpp).
//
// runtime args: [pes_addr, out_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../../moe_decode/kernels/router_topk_math.hpp"

void kernel_main() {
    const uint32_t pes_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_logits = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(1);
    constexpr uint32_t E = get_compile_time_arg_val(2);
    constexpr uint32_t k = get_compile_time_arg_val(3);
    constexpr uint32_t extra = get_compile_time_arg_val(4);  // always-routed slots after the picks (weight 1, ids E..)
    constexpr auto p_args = TensorAccessorArgs<5>();
    constexpr auto o_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    const auto s_p = TensorAccessor(p_args, pes_addr, E * 2);
    constexpr uint32_t n = k + extra;
    const auto s_o = TensorAccessor(o_args, out_addr, 4 * n);

    // E FP32 logits, E bf16 scales (read from DRAM: 64 B aligned), the 2 k bf16 output row
    const uint32_t scr = (get_write_ptr(cb_scr) + 63) & ~63u;
    const uint32_t pes_l1 = scr + E * 4;
    noc_async_read(s_p.get_noc_addr(0), pes_l1, E * 2);
    cb_wait_front(cb_logits, E / 32);
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_logits));
    noc_async_read_barrier();
#ifdef DEBUG_LOGITS
    constexpr auto d_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    {
        volatile tt_l1_ptr uint32_t* logit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
        for (uint32_t n = 0; n < E / 32; ++n) {
            for (uint32_t c = 0; c < 16; ++c) {
                logit[32 * n + c] = src[1024 * n + c];
                logit[32 * n + 16 + c] = src[1024 * n + 256 + c];
            }
        }
    }
    noc_async_write(scr, TensorAccessor(d_args, get_arg_val<uint32_t>(2), E * 4).get_noc_addr(0), E * 4);
    noc_async_write_barrier();
#endif
    volatile tt_l1_ptr uint16_t* row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + E * 4 + E * 2);
    volatile tt_l1_ptr uint16_t* picks = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + E * 4 + E * 2 + 4 * n);
    router_tail::topk_softmax_tiles<E, k>(src, reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pes_l1), picks);
    // [w_0 .. w_{k-1}, 1 x extra | id_0 .. id_{k-1}, E .. E + extra - 1]
    for (uint32_t i = 0; i < k; ++i) {
        row[i] = picks[i];
        row[n + i] = picks[k + i];
    }
    for (uint32_t i = 0; i < extra; ++i) {
        row[k + i] = 0x3F80;  // 1.0
        row[n + k + i] = router_tail::uint_to_bf16(E + i);
    }
    noc_async_write(scr + E * 4 + E * 2, s_o.get_noc_addr(0), 4 * n);
    noc_async_write_barrier();
    cb_pop_front(cb_logits, E / 32);
}
