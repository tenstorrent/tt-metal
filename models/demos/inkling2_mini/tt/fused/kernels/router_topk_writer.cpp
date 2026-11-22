// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Router top-k for one token, writer: scalars for the normalize tail (as the moe_grouped_topk writer), then
// the top-k over row 0 of the biased-score tiles, the gather of the unbiased scores, and the two outputs.
// Selection and order match moe_grouped_topk's stable rank-tag sort: keys compare at 17 mantissa bits (the
// low 6 bits cleared, -0 folded into +0) as sign-magnitude floats, largest first, equal keys by lower index.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/dataflow/moe_gate_common_dataflow.hpp"

namespace {
constexpr uint32_t TAG_CLEAR = ~0x3fu;

FORCE_INLINE uint32_t order_key(uint32_t a) {
    uint32_t m = a & TAG_CLEAR;
    if ((m << 1) == 0) {
        m = 0;
    }
    return (m & 0x80000000u) ? ~m : (m | 0x80000000u);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t k = get_named_compile_time_arg_val("k");
    constexpr uint32_t packed_eps = get_named_compile_time_arg_val("packed_eps");
    constexpr uint32_t packed_scale = get_named_compile_time_arg_val("packed_scale");
    constexpr uint32_t cb_out_weights = 2, cb_out_indices = 3, cb_sigmoid = 4, cb_biased = 5;
    constexpr uint32_t cb_ones = 17, cb_eps = 18, cb_scale = 19, cb_gathered = 22;
    static_assert(k <= 16, "indices are written to face 0 row 0 only");

    const uint32_t weights_addr = get_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_arg_val<uint32_t>(1);
    constexpr auto w_args = TensorAccessorArgs<0>();
    constexpr auto i_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto w_acc = TensorAccessor(w_args, weights_addr, 2048);
    const auto i_acc = TensorAccessor(i_args, indices_addr, 2048);

    generate_reduce_scalar(cb_ones, 0x3f800000u, k);
    write_single_scalar(cb_eps, packed_eps);
    write_single_scalar(cb_scale, packed_scale);

    // top-k by insertion into a sorted list; candidates arrive in index order, so a strict compare keeps
    // equal keys in index order
    uint32_t key[k];
    uint16_t idx[k];
    uint32_t n = 0;
    cb_wait_front(cb_biased, width_tiles);
    volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_biased));
    for (uint32_t e = 0; e < width_tiles * 32; ++e) {
        const uint32_t c = e & 31;
        const uint32_t kv = order_key(b[(e >> 5) * 1024 + (c < 16 ? c : 256 + c - 16)]);
        if (n == k && kv <= key[k - 1]) {
            continue;
        }
        uint32_t p = n < k ? n++ : k - 1;
        while (p > 0 && kv > key[p - 1]) {
            key[p] = key[p - 1];
            idx[p] = idx[p - 1];
            --p;
        }
        key[p] = kv;
        idx[p] = e;
    }
    cb_pop_front(cb_biased, width_tiles);

    cb_reserve_back(cb_out_indices, 1);
    const uint32_t it = get_write_ptr(cb_out_indices);
    volatile tt_l1_ptr uint32_t* iw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(it);
    for (uint32_t i = 0; i < 512; ++i) {
        iw[i] = 0;
    }
    volatile tt_l1_ptr uint16_t* ih = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(it);
    for (uint32_t i = 0; i < k; ++i) {
        ih[i] = idx[i];
    }
    cb_push_back(cb_out_indices, 1);

    cb_reserve_back(cb_gathered, 1);  // zero the gathered tile: only row 0, columns < k are written
    volatile tt_l1_ptr uint32_t* g = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_gathered));
    for (uint32_t i = 0; i < 1024; ++i) {
        g[i] = 0;
    }
    gather<cb_out_indices, cb_sigmoid, cb_gathered, width_tiles, k, 1>(1);

    noc_async_write(it, i_acc.get_noc_addr(0), 2048);
    cb_wait_front(cb_out_weights, 1);
    noc_async_write(get_read_ptr(cb_out_weights), w_acc.get_noc_addr(0), 2048);
    noc_async_write_barrier();
    cb_pop_front(cb_out_weights, 1);
    cb_pop_front(cb_out_indices, 1);
}
