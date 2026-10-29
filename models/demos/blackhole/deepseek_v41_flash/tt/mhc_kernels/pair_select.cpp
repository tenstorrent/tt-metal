// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// DSV4.1 drafter sampler tail, one core per user row u: the 8 per-column (max, local index) pairs of the vocab-sharded
// logits are gathered as an fp32 TILE row [max_0, idx_0, ..., max_7, idx_7] (cols 0..15 of row u). Best column = first
// maximum (lowest column wins ties, like argmax), token = col * SHARD + idx. Outputs: tok uint32 row-major [U,1] (4 B
// pages) and the markov-embedding row of that token: embed[tok] (row-major bf16 [V,256], 512 B pages) -> emb_out page
// u. Replaces to_layout / slice / slice / argmax / gather / typecast / multiply / add / typecast / reshape + the
// embedding lookup of the next step (11 programs).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

static inline float as_f(uint32_t b) {
    union {
        uint32_t u;
        float f;
    } c;
    c.u = b;
    return c.f;
}

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t SHARD = get_compile_time_arg_val(1);
    constexpr uint32_t COLS = get_compile_time_arg_val(2);
    constexpr uint32_t EMB_PAGE = get_compile_time_arg_val(3);
    constexpr uint32_t TILE_PAIRS =
        get_compile_time_arg_val(4);  // 1: the gathered row is [max_0 tile, idx_0 tile, max_1 tile, ...] (col 0 of each
                                      // 32-wide tile), 0: packed (max_c, idx_c) in cols 0..15 of tile 0
    constexpr auto p_args = TensorAccessorArgs<5>();
    constexpr auto t_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto e_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t u = get_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t t_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t e_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(3);

    Noc noc;
    const auto p_acc = TensorAccessor(p_args, p_addr, TILE);
    const auto t_acc = TensorAccessor(t_args, t_addr, 4);
    const auto e_acc = TensorAccessor(e_args, e_addr, EMB_PAGE);
    const auto o_acc = TensorAccessor(o_args, o_addr, EMB_PAGE);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);

    const uint32_t roff =
        ((u >> 4) << 11) + ((u & 15) << 6);  // row u of tile 0: face 0 (rows 0..15) / face 2 (rows 16..31), cols 0..15
    uint32_t best = 0, il = 0;
    if constexpr (TILE_PAIRS) {
        for (uint32_t t = 0; t < 2 * COLS; ++t) {
            noc.async_read(p_acc, cb, 64, {.page_id = t, .offset_bytes = roff}, {.offset_bytes = t * 64});
        }
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* pr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
        float bv = as_f(pr[0]);
        for (uint32_t c = 1; c < COLS; ++c) {
            const float v =
                as_f(pr[32 * c]);  // tile 2c, element 0 (64 B per tile read = 16 words; tile 2c at word 32c)
            if (v > bv) {
                bv = v;
                best = c;
            }
        }
        il = (uint32_t)as_f(pr[32 * best + 16]);  // tile 2c + 1
    } else {
        noc.async_read(p_acc, cb, 64, {.page_id = 0, .offset_bytes = roff}, {.offset_bytes = 0});
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* pr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
        float bv = as_f(pr[0]);
        for (uint32_t c = 1; c < COLS; ++c) {
            const float v = as_f(pr[2 * c]);
            if (v > bv) {  // strictly greater: the lowest column wins ties
                bv = v;
                best = c;
            }
        }
        il = (uint32_t)as_f(pr[2 * best + 1]);
    }
    const uint32_t tok = best * SHARD + il;
    volatile tt_l1_ptr uint32_t* pr0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* ot =
        pr0 + (TILE_PAIRS ? 16 * 2 * COLS : 16);  // after the staged pair data (64 B aligned)
    ot[0] = tok;
    constexpr uint32_t OT = TILE_PAIRS ? 64 * 2 * COLS : 64;  // byte offset of the token word / the embedding row (+64)
    noc.async_read(e_acc, cb, EMB_PAGE, {.page_id = tok, .offset_bytes = 0}, {.offset_bytes = OT + 64});
    noc.async_write(cb, t_acc, 4, {.offset_bytes = OT}, {.page_id = u, .offset_bytes = 0});
    noc.async_read_barrier();
    noc.async_write(cb, o_acc, EMB_PAGE, {.offset_bytes = OT + 64}, {.page_id = u, .offset_bytes = 0});
    noc.async_write_barrier();
}
