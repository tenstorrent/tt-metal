// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// R-core reader. Reads the RoPE transformation matrix once, then the cos and sin rows for each
// position block straight from the full [1, 1, S_rot, 128] rot mats (no per-layer slicing), and
// receives the normalised q rows from the paired N core (see norm_handoff_writer.cpp).
// Rows run head-fastest, so one cos/sin row serves up to H consecutive q rows: it is pushed on the
// first row and on every row with h == 0, and rope_compute.cpp pops it after h == H - 1 or the last row.
//
// Compile-time args: 0+ accessors cos, sin, trans_mat
// Runtime args: 0 cos_addr, 1 sin_addr, 2 trans_addr, 3 num_rows, 4 h0, 5 j0, 6 tl0, 7 sl_t, 8 s0_t,
//               9 s1_t, 10 ht, 11 n_heads, 12 n_noc_x, 13 n_noc_y

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "qprep_common.hpp"

using namespace qprep;

template <typename Acc>
FORCE_INLINE void read_row(const Noc& noc, const Acc& acc, uint32_t cb_id, uint32_t first_page) {
    constexpr uint32_t bytes = get_tile_size(CB_COS);
    CircularBuffer cb(cb_id);
    cb.reserve_back(WT);
    uint32_t dst = cb.get_write_ptr();
    for (uint32_t i = 0; i < WT; ++i) {
        noc.async_read(acc, CoreLocalMem<uint32_t>(dst), bytes, {.page_id = first_page + i}, {});
        dst += bytes;
    }
    noc.async_read_barrier();
    cb.push_back(WT);
}

void kernel_main() {
    constexpr auto cos_args = TensorAccessorArgs<0>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    static_assert(get_tile_size(CB_COS) == get_tile_size(CB_SIN), "cos and sin tiles must match");

    const uint32_t cos_addr = get_arg_val<uint32_t>(0);
    const uint32_t sin_addr = get_arg_val<uint32_t>(1);
    const uint32_t trans_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);
    RowWalker w{
        get_arg_val<uint32_t>(4),
        get_arg_val<uint32_t>(5),
        get_arg_val<uint32_t>(6),
        get_arg_val<uint32_t>(7),
        get_arg_val<uint32_t>(8),
        get_arg_val<uint32_t>(9),
        get_arg_val<uint32_t>(10),
        get_arg_val<uint32_t>(11)};
    const uint32_t n_noc_x = get_arg_val<uint32_t>(12);
    const uint32_t n_noc_y = get_arg_val<uint32_t>(13);

    const auto cos = TensorAccessor(cos_args, cos_addr);
    const auto sin = TensorAccessor(sin_args, sin_addr);
    const auto trans = TensorAccessor(trans_args, trans_addr);
    Noc noc;

    {
        CircularBuffer cb(CB_TRANS);
        cb.reserve_back(1);
        noc.async_read(trans, CoreLocalMem<uint32_t>(cb.get_write_ptr()), get_tile_size(CB_TRANS), {.page_id = 0}, {});
        noc.async_read_barrier();
        cb.push_back(1);
    }

    const uint64_t credit_noc = get_noc_addr(n_noc_x, n_noc_y, get_semaphore(SEM_CREDIT));
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_READY));
    CircularBuffer cb_normed(CB_NORMED);

    for (uint32_t r = 0; r < num_rows; ++r) {
        // Slot r % depth is free once compute has popped row r - depth; grant it to the N core.
        cb_normed.reserve_back(WT);
        noc_semaphore_inc(credit_noc, 1);

        if (r == 0 || w.h == 0) {
            const uint32_t rot_page = w.rot_page();
            read_row(noc, sin, CB_SIN, rot_page);
            read_row(noc, cos, CB_COS, rot_page);
        }

        noc_semaphore_wait_min(ready, r + 1);
        cb_normed.push_back(WT);
        w.next();
    }
    noc_async_atomic_barrier();
}
