// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-32 LM-head argmax, writer (NCRISC): the compute's FP32 max / column tiles -> each row's best (value, vocabulary
// index) over this core's columns (ties: the smaller index), sent to the row tile's leader (slot `chunk` of its
// cb_recv, then its semaphore); the leader (chunk 0) reduces the NCH chunks and writes tile rt of the outputs:
// vals / idx [ROWS, 1] FP32 (column 0; the index as a float, exact below 2^24).
//
// runtime args: [rt, chunk, leader noc (x << 8 | y)]; common runtime args: [vals, idx]
// compile-time args: [cb_part, cb_recv, NCH, SEM, accessors (vals, idx)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
// FP32 bits -> an unsigned key in the same order as the floats
inline uint32_t key(uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
}  // namespace

void kernel_main() {
    const uint32_t rt = get_arg_val<uint32_t>(0);
    const uint32_t chunk = get_arg_val<uint32_t>(1);
    const uint32_t leader = get_arg_val<uint32_t>(2);
    const uint32_t vals_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_common_arg_val<uint32_t>(1);
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(1);
    constexpr uint32_t NCH = get_compile_time_arg_val(2);
    constexpr uint32_t SEM = get_compile_time_arg_val(3);
    constexpr auto v_args = TensorAccessorArgs<4>();
    constexpr auto i_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr uint32_t ftile = 4096;   // FP32 tile
    constexpr uint32_t SLOT = 32 * 8;  // per chunk: 32 rows x (value bits, index)
    const auto s_v = TensorAccessor(v_args, vals_addr, ftile);
    const auto s_i = TensorAccessor(i_args, idx_addr, ftile);

    cb_reserve_back(cb_recv, 1);
    const uint32_t recv = get_write_ptr(cb_recv);  // [NCH slots | local staging slot | 2 output tiles]
    const uint32_t mine = recv + NCH * SLOT;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mine);

    cb_wait_front(cb_part, 2);
    invalidate_l1_cache();
    const volatile tt_l1_ptr uint32_t* M = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_part));
    const volatile tt_l1_ptr float* I = reinterpret_cast<volatile tt_l1_ptr float*>(get_read_ptr(cb_part) + ftile);
    for (uint32_t r = 0; r < 32; ++r) {
        uint32_t bk = 0, bb = 0, bi = 0xFFFFFFFFu;
        for (uint32_t j = 0; j < 32; ++j) {
            const uint32_t o = ((r / 16) * 2 + j / 16) * 256 + (r % 16) * 16 + j % 16;
            const uint32_t b = M[o], k = key(b);
            const uint32_t idx = static_cast<uint32_t>(I[o]) * 32 + j;
            if (k > bk || (k == bk && idx < bi)) {
                bk = k;
                bb = b;
                bi = idx;
            }
        }
        out[2 * r] = bb;
        out[2 * r + 1] = bi;
    }
    cb_pop_front(cb_part, 2);
    const uint32_t lx = leader >> 8, ly = leader & 0xFF;
    noc_async_write(mine, get_noc_addr(lx, ly, recv + chunk * SLOT), SLOT);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(lx, ly, get_semaphore(SEM)), 1);
    noc_async_atomic_barrier();
    if (chunk != 0) {
        return;
    }
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM));
    noc_semaphore_wait(sem, NCH);
    noc_semaphore_set(sem, 0);
    invalidate_l1_cache();
    const uint32_t vt = mine + SLOT, it = vt + ftile;  // the output tiles (FP32), column 0
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t z = 0; z < 2 * ftile; z += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, vt + z, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    const volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv);
    volatile tt_l1_ptr uint32_t* V = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vt);
    volatile tt_l1_ptr float* X = reinterpret_cast<volatile tt_l1_ptr float*>(it);
    for (uint32_t r = 0; r < 32; ++r) {
        uint32_t bk = 0, bb = 0, bi = 0xFFFFFFFFu;
        for (uint32_t c = 0; c < NCH; ++c) {
            const uint32_t b = in[c * 64 + 2 * r], idx = in[c * 64 + 2 * r + 1], k = key(b);
            if (k > bk || (k == bk && idx < bi)) {
                bk = k;
                bb = b;
                bi = idx;
            }
        }
        const uint32_t o = (r / 16) * 2 * 256 + (r % 16) * 16;  // element (r, 0)
        V[o] = bb;
        X[o] = static_cast<float>(bi);
    }
    noc_async_write(vt, s_v.get_noc_addr(rt), ftile);
    noc_async_write(it, s_i.get_noc_addr(rt), ftile);
    noc_async_write_barrier();
}
