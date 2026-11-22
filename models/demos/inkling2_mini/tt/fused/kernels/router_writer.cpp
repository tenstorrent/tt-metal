// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused decode router, writer. Every core: row 0 of its GEMM output tile -> its place in the collector's
// partial-sum pages (page = K chunk, dense: expert e at element e), then a bump of the collector's semaphore.
// The collector then: scalars for the normalize tail (as the moe_grouped_topk writer), the top-k over the
// dense biased-score page, the gather of the unbiased scores, and the two outputs.
// Selection and order match moe_grouped_topk's stable rank-tag sort: keys compare at 17 mantissa bits (the
// low 6 bits cleared, -0 folded into +0) as sign-magnitude floats, largest first, equal keys by lower index.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/dataflow/moe_gate_common_dataflow.hpp"  // scalars

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

    constexpr uint32_t sem_id = get_named_compile_time_arg_val("sem");
    constexpr uint32_t cb_mm = 8, cb_parts = 9;
    const uint32_t weights_addr = get_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_arg_val<uint32_t>(1);
    const uint32_t chunk = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    const uint32_t collector = get_arg_val<uint32_t>(4);
    const uint32_t cx = get_arg_val<uint32_t>(5);
    const uint32_t cy = get_arg_val<uint32_t>(6);

    // fp32 row 0 of the tile: columns 0..15 in face 0, 16..31 in face 1 (+1024 B)
    cb_wait_front(cb_mm, 1);
    const uint32_t src = get_read_ptr(cb_mm);
    const uint32_t dst = get_write_ptr(cb_parts) + chunk * 4096 + n * 128;  // same address on every core
    noc_async_write(src, get_noc_addr(cx, cy, dst), 64);
    noc_async_write(src + 1024, get_noc_addr(cx, cy, dst + 64), 64);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(sem_id)), 1);
    noc_async_atomic_barrier();
    cb_pop_front(cb_mm, 1);
    if (!collector) {
        return;
    }

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
    uint32_t cnt = 0;
    cb_wait_front(cb_biased, 1);
    volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_biased));
    for (uint32_t e = 0; e < width_tiles * 32; ++e) {  // dense: expert e at element e
        const uint32_t kv = order_key(b[e]);
        if (cnt == k && kv <= key[k - 1]) {
            continue;
        }
        uint32_t p = cnt < k ? cnt++ : k - 1;
        while (p > 0 && kv > key[p - 1]) {
            key[p] = key[p - 1];
            idx[p] = idx[p - 1];
            --p;
        }
        key[p] = kv;
        idx[p] = e;
    }
    cb_pop_front(cb_biased, 1);

    cb_reserve_back(cb_out_indices, 1);
    const uint32_t it = get_write_ptr(cb_out_indices);
    volatile tt_l1_ptr uint32_t* iw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(it);
    for (uint32_t i = 0; i < 8; ++i) {  // row 0 (consumers read the first k ids)
        iw[i] = 0;
        iw[128 + i] = 0;
    }
    volatile tt_l1_ptr uint16_t* ih = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(it);
    for (uint32_t i = 0; i < k; ++i) {
        ih[i] = idx[i];
    }
    cb_push_back(cb_out_indices, 1);

    // gathered tile: row 0 = the picks' unbiased scores, then zeros (the op leaves stale data past column k,
    // which the row reduce weights by 0; rows > 0 never reach row 0's result)
    cb_reserve_back(cb_gathered, 1);
    volatile tt_l1_ptr uint32_t* g = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_gathered));
    for (uint32_t i = 0; i < 16; ++i) {
        g[i] = 0;
        g[256 + i] = 0;
    }
    cb_wait_front(cb_sigmoid, 1);  // unbiased scores of the picks -> row 0, columns 0..k-1 (as the op's gather)
    volatile tt_l1_ptr uint32_t* sg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_sigmoid));
    for (uint32_t i = 0; i < k; ++i) {
        g[i] = sg[idx[i]];
    }
    cb_pop_front(cb_sigmoid, 1);
    cb_push_back(cb_gathered, 1);

    noc_async_write(it, i_acc.get_noc_addr(0), 2048);
    cb_wait_front(cb_out_weights, 1);
    noc_async_write(get_read_ptr(cb_out_weights), w_acc.get_noc_addr(0), 2048);
    noc_async_write_barrier();
    cb_pop_front(cb_out_weights, 1);
    cb_pop_front(cb_out_indices, 1);
}
