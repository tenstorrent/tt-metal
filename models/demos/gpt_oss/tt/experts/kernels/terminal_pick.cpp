// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Greedy pick of the fused decode sampler (tt/decode_terminal.py), one core: the argmax over the gathered candidates
// of user 0 (row 0 of values BF16 [1, 1, 32, n] tile layout and ids UINT32 [1, 1, 32, n] row-major), ties to the
// lowest id, written as the user-0 token into the decode token tensor (UINT32 [1, 1, 1, 32] row-major, one page);
// the other 31 token slots are written as 0.
//
// runtime args: [v_addr, i_addr, tok_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
inline uint32_t key_of(uint16_t b) { return (b & 0x8000) ? (~b & 0xFFFFu) : (b | 0x8000u); }
}  // namespace

void kernel_main() {
    const uint32_t v_addr = get_arg_val<uint32_t>(0);
    const uint32_t i_addr = get_arg_val<uint32_t>(1);
    const uint32_t tok_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(1);  // candidate tiles in a row
    constexpr auto v_args = TensorAccessorArgs<2>();
    constexpr auto i_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr uint32_t n = n_tiles * 32;

    const auto s_v = TensorAccessor(v_args, v_addr, 2048);
    const auto s_i = TensorAccessor(i_args, i_addr, n * 4);
    const auto s_t = TensorAccessor(t_args, tok_addr, 128);

    cb_reserve_back(cb_scr, 1);
    const uint32_t base = get_write_ptr(cb_scr);  // [row 0 values: n_tiles x (16 | 16) BF16 | ids n x 4 | tokens 128]
    for (uint32_t t = 0; t < n_tiles; ++t) {
        const uint64_t src = s_v.get_noc_addr(t);
        noc_async_read(src, base + t * 64, 32);
        noc_async_read(src + 512, base + t * 64 + 32, 32);
    }
    const uint32_t ids_l1 = base + n_tiles * 64;
    noc_async_read(s_i.get_noc_addr(0), ids_l1, n * 4);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_l1);

    uint32_t best_key = key_of(v[0]);
    uint32_t best_id = ids[0];
    for (uint32_t j = 1; j < n; ++j) {
        const uint32_t kk = key_of(v[j]);
        const uint32_t id = ids[j];
        if (kk > best_key || (kk == best_key && id < best_id)) {
            best_key = kk;
            best_id = id;
        }
    }
    const uint32_t tok_l1 = ids_l1 + n * 4;
    volatile tt_l1_ptr uint32_t* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tok_l1);
    tok[0] = best_id;
    for (uint32_t j = 1; j < 32; ++j) {
        tok[j] = 0;
    }
    noc_async_write(tok_l1, s_t.get_noc_addr(0), 128);
    noc_async_write_barrier();
}
