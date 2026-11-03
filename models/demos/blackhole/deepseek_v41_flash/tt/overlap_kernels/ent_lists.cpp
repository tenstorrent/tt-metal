// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Key-slab update indices of the column-sharded indexer key append, one program instead of ~45 tiny ops:
//   entry(u, j)  = floor(pos[u*n + j] / ratio)  if (pos + 1) % ratio == 0 else -1   (the entry written by block row j
//   of user u) vloc[j][u]   = entry / cols (floor)  if entry's column (entry - floor(entry/cols)*cols) == this device's
//   column else -1 vhead[j][u]  = entry if entry < n_head else -1
// Same arithmetic as spec_paged._ent_matrix / _cs_ent_lists (float32 floor / eq / where) in integers. Outputs: 2n int32
// [U] row-major tensors (vloc[0..n), vhead[0..n)).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

static inline int32_t fdiv(int32_t a, int32_t b) {
    int32_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) {
        q -= 1;
    }
    return q;
}

void kernel_main() {
    constexpr uint32_t CB = get_compile_time_arg_val(0);
    constexpr uint32_t U = get_compile_time_arg_val(1);
    constexpr uint32_t N = get_compile_time_arg_val(2);
    constexpr int32_t RATIO = get_compile_time_arg_val(3);
    constexpr int32_t COLS = get_compile_time_arg_val(4);
    constexpr int32_t NHEAD = get_compile_time_arg_val(5);
    constexpr uint32_t R = U * N;
    constexpr auto pos_args = TensorAccessorArgs<6>();
    constexpr auto col_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<col_args.next_compile_time_args_offset()>();

    const uint32_t pos_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t col_addr = get_common_arg_val<uint32_t>(1);

    cb_reserve_back(CB, 1);
    const uint32_t base = get_write_ptr(CB);
    const uint32_t pos_l1 = base;
    const uint32_t col_l1 = base + 256;
    const uint32_t out_l1 = base + 512;  // 2N staging slots of 64 bytes (U <= 16)

    const auto pos = TensorAccessor(pos_args, pos_addr, R * 4);
    const auto col = TensorAccessor(col_args, col_addr, 64);
    noc_async_read(pos.get_noc_addr(0), pos_l1, R * 4);
    noc_async_read(col.get_noc_addr(0), col_l1, 64);
    noc_async_read_barrier();

    volatile tt_l1_ptr int32_t* p = reinterpret_cast<volatile tt_l1_ptr int32_t*>(pos_l1);
    const int32_t c = reinterpret_cast<volatile tt_l1_ptr int32_t*>(col_l1)[0];
    for (uint32_t j = 0; j < N; ++j) {
        volatile tt_l1_ptr int32_t* vl = reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1 + j * 64);
        volatile tt_l1_ptr int32_t* vh = reinterpret_cast<volatile tt_l1_ptr int32_t*>(out_l1 + (N + j) * 64);
        for (uint32_t u = 0; u < U; ++u) {
            const int32_t pp = p[u * N + j];
            const bool comp = ((pp + 1) % RATIO + RATIO) % RATIO == 0;
            const int32_t e = comp ? fdiv(pp, RATIO) : -1;
            const int32_t loc = fdiv(e, COLS);
            vl[u] = ((e - loc * COLS) == c) ? loc : -1;
            vh[u] = (e < NHEAD) ? e : -1;
        }
    }
    const uint32_t addr0 = get_common_arg_val<uint32_t>(2);
    for (uint32_t k = 0; k < 2 * N; ++k) {
        const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(2 + k), U * 4);
        noc_async_write(out_l1 + k * 64, out.get_noc_addr(0), U * 4);
    }
    (void)addr0;
    noc_async_write_barrier();
}
