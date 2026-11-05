// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Combine reduction, reader: unit u = (token t, 1024-element column chunk j). Pushes the N partial chunks
// comb[d*T + t][j] (zeros for devices the token did not hit) as N "tiles" of 1024 contiguous bf16.
// Elementwise add is layout-agnostic, so row-major chunks can go through the FPU as pseudo-tiles.
#include "api/dataflow/dataflow_api.h"
#include "ep_common.hpp"

using namespace k31ep;

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t MY = get_compile_time_arg_val(1);
    constexpr uint32_t T = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
    constexpr uint32_t EPD = get_compile_time_arg_val(4);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(5);
    constexpr uint32_t in_cb = get_compile_time_arg_val(6);
    constexpr uint32_t aux_cb = get_compile_time_arg_val(7);  // [meta T*64 | zero chunk 2048]
    constexpr auto meta_args = TensorAccessorArgs<8>();
    constexpr auto comb_args = TensorAccessorArgs<meta_args.next_compile_time_args_offset()>();
    constexpr uint32_t CHUNK = 2048;
    constexpr uint32_t NCH = ROW_BYTES / CHUNK;
    const uint32_t meta_addr = get_arg_val<uint32_t>(0);
    const uint32_t comb_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t u1 = get_arg_val<uint32_t>(3);
    if (u0 >= u1) {
        return;
    }
    const auto meta = TensorAccessor(meta_args, meta_addr, META_BYTES);
    const auto comb = TensorAccessor(comb_args, comb_addr, ROW_BYTES);

    cb_reserve_back(aux_cb, 1);
    const uint32_t meta_l1 = get_write_ptr(aux_cb);
    const uint32_t zero_l1 = meta_l1 + T * META_BYTES;
    for (uint32_t t = 0; t < T; ++t) {
        noc_async_read(meta.get_noc_addr(t), meta_l1 + t * META_BYTES, META_BYTES);
    }
    auto* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_l1);
    for (uint32_t i = 0; i < CHUNK / 4; ++i) {
        z[i] = 0;
    }
    noc_async_read_barrier();
    const uint64_t zero_noc = get_noc_addr(zero_l1);

    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t t = u / NCH;
        const uint32_t j = u % NCH;
        const uint32_t mask =
            hit_mask<K, EPD>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1 + t * META_BYTES));
        cb_reserve_back(in_cb, N);
        const uint32_t w = get_write_ptr(in_cb);
        for (uint32_t d = 0; d < N; ++d) {
            const uint64_t src = ((mask >> d) & 1u) ? comb.get_noc_addr(d * T + t, j * CHUNK) : zero_noc;
            noc_async_read(src, w + d * CHUNK, CHUNK);
        }
        noc_async_read_barrier();
        cb_push_back(in_cb, N);
    }
}
