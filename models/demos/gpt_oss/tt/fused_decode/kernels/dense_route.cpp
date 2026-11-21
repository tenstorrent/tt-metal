// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batched decode routing (topk.py TopKRouter.dense_decode): the router logits [32 tokens, E] (one BF16 tile) -> the
// dense routing weights [32, E] (each token's softmax over its routed experts, zeros elsewhere) in one program, one
// core per token (route_row.hpp, as the grouped MoE's route program) instead of top-k, two reductions, a compare, exp,
// multiply, sum and divide ops.
//
// common runtime args: [logits, w]; compile-time args: [cb_scr, GX, K_SEL, K, accessors (logits, w)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "route_row.hpp"

void kernel_main() {
    const uint32_t logits_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t GX = get_compile_time_arg_val(1);
    constexpr uint32_t K_SEL = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr auto l_args = TensorAccessorArgs<4>();
    constexpr auto w_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile = 2048, E = 32;

    const uint32_t r = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();  // the token
    const auto s_l = TensorAccessor(l_args, logits_addr, tile);
    const auto s_w = TensorAccessor(w_args, w_addr, tile);
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);  // [the row's logits (64 bytes) | its weights (64 bytes)]
    for (uint32_t f = 0; f < 2; ++f) {           // the row's two 16-value face rows
        noc_async_read(s_l.get_noc_addr(0) + ((r / 16) * 2 + f) * 512 + (r % 16) * 32, scr + f * 32, 32);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint32_t b[E];
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    for (uint32_t w = 0; w < 16; ++w) {
        const uint32_t two = src[w];
        b[2 * w] = two & 0xFFFF;
        b[2 * w + 1] = two >> 16;
    }
    uint32_t sel[K], n, taken;
    uint16_t wb[K];
    route_row::route<E, K, K_SEL>(b, sel, n, taken, wb);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 64);
    for (uint32_t w = 0; w < 16; ++w) {
        out[w] = 0;
    }
    volatile tt_l1_ptr uint16_t* o16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + 64);
    for (uint32_t k = 0; k < n; ++k) {
        o16[sel[k]] = wb[k];  // (expert e: face e / 16, column e % 16: the 64 bytes are the two face rows in order)
    }
    for (uint32_t f = 0; f < 2; ++f) {
        noc_async_write(scr + 64 + f * 32, s_w.get_noc_addr(0) + ((r / 16) * 2 + f) * 512 + (r % 16) * 32, 32);
    }
    noc_async_write_barrier();
}
