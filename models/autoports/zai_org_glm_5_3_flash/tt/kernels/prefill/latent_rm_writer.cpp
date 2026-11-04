// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 GLM53_PF_LAT_BOUND writer: the untilized tile rows (32 rows x ROW_BYTES each) -> the ROW_MAJOR key store
// [1, 1, T_local, 512] (page = row), local row lb * BLOCK_ROWS + tr * 32 + k; rows past the visible count are not
// written (sparse_sdpa_stats reads only selected ids, all < the visible local rows). The same device word as the
// reader. Compile-time: ROW_BYTES, TROWS, BPU, BLOCK_ROWS, TPR. Runtime: out, bound, core i, NB, NC (the reader's
// strided blocks).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(0);
    constexpr uint32_t TROWS = get_compile_time_arg_val(1);
    constexpr uint32_t BPU = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(3);
    constexpr uint32_t TPR = get_compile_time_arg_val(4);
    constexpr uint32_t CB_OUT = 1, CB_WS = 4;
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t bound_addr = get_arg_val<uint32_t>(1);
    const uint32_t ci = get_arg_val<uint32_t>(2);
    const uint32_t nb = get_arg_val<uint32_t>(3);
    const uint32_t nc = get_arg_val<uint32_t>(4);

    const uint32_t ws = get_write_ptr(CB_WS);
    const InterleavedAddrGen<true> bound_g{.bank_base_address = bound_addr, .page_size = 4};
    noc_async_read(bound_g.get_noc_addr(0), ws, 64);
    noc_async_read_barrier();
    uint32_t lim = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws)[0] * BPU;
    lim = lim < nb ? lim : nb;
    if (lim <= ci) {
        return;
    }
    const InterleavedAddrGen<true> dst{.bank_base_address = out_addr, .page_size = ROW_BYTES};
    for (uint32_t lb = ci; lb < lim; lb += nc) {
        for (uint32_t tr = 0; tr < TROWS; ++tr) {
            cb_wait_front(CB_OUT, TPR);
            const uint32_t l1 = get_read_ptr(CB_OUT);
            const uint32_t row0 = lb * BLOCK_ROWS + tr * 32;
            for (uint32_t k = 0; k < 32; ++k) {
                noc_async_write(l1 + k * ROW_BYTES, dst.get_noc_addr(row0 + k), ROW_BYTES);
            }
            noc_async_write_barrier();
            cb_pop_front(CB_OUT, TPR);
        }
    }
}
