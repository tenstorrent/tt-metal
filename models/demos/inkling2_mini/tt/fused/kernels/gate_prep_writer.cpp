// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Message-gate prep, writer (one core per input vector): bf16 rms(v) (dense) -> row r of the split-K x tensor
// [1, SPLIT, R, KS] TILE bf16, at x columns col0 .. col0 + C (h: col0 = 0, m: col0 = C), ready for the batched
// down_proj matmul (E2 split_k_matmul without its reshape / permute).
#include "rowio.hpp"

void kernel_main() {
    constexpr uint32_t ntc = get_named_compile_time_arg_val("ntc");
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");
    constexpr uint32_t cb_np = 16;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t r = get_arg_val<uint32_t>(1);
    const uint32_t col0_tiles = get_arg_val<uint32_t>(2);

    const InterleavedAddrGen<true> dst = {.bank_base_address = x_addr, .page_size = 2048};
    const uint32_t off = rowio::row_offset(r, 2);
    cb_wait_front(cb_np, T);
    const uint32_t src = get_read_ptr(cb_np);
    for (uint32_t j = 0; j < ntc; ++j) {
        const uint32_t page = col0_tiles + j;  // split b, tile jj of x column tile t = b * ks_tiles + jj = page
        noc_async_write(src + j * 64, dst.get_noc_addr(page, off), 32);
        noc_async_write(src + j * 64 + 32, dst.get_noc_addr(page, off + 512), 32);
    }
    noc_async_write_barrier();
}
