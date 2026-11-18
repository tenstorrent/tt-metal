// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B17 fused q-prep: writer. Row 0 of each of the 2 output tiles (face 0 cols 0-15, face 1 cols 16-31) -> 64
// contiguous bytes of the row-major out [1,H,1,576] bf16 (page = head, 1152 B): absorb core (h, j): byte offset (2j +
// t) * 64 of row h; rope core h: 1024 + t * 64.  CT: 0 cb_out, 1 cb_scr, 2 S (rows), then 2 S, TensorAccessorArgs(out).
// Common: out addr.  RT: role, h, j.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr auto o_args = TensorAccessorArgs<3>();
    const uint32_t role = get_arg_val<uint32_t>(0), h = get_arg_val<uint32_t>(1), j = get_arg_val<uint32_t>(2);
    const auto oa = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 1152);
    CircularBuffer o(cb_out), sc(cb_scr);
    o.wait_front(2);
    const uint32_t p = o.get_read_ptr();
    const uint32_t scr = sc.get_write_ptr();
    for (uint32_t t = 0; t < 2; ++t) {
        const uint32_t off = (role == 0) ? (2 * j + t) * 64 : 1024 + t * 64;
        for (uint32_t r = 0; r < S; ++r) {  // row r of the tile -> 64 contiguous bytes of out row (h * S + r)
            const uint32_t sp = scr + (t * S + r) * 64;
            volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sp);
            volatile tt_l1_ptr uint32_t* f0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p + t * 2048 + r * 32);
            volatile tt_l1_ptr uint32_t* f1 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p + t * 2048 + 512 + r * 32);
            for (uint32_t i = 0; i < 8; ++i) {
                d[i] = f0[i];
                d[8 + i] = f1[i];
            }
            noc_async_write(sp, oa.get_noc_addr(h * S + r, off), 64);
        }
    }
    noc_async_write_barrier();
#ifdef QF_NC
    // B47b (HY4_B47B_QFOLD): tell the QF_NC push comm cores (phys x,y at runtime args 3..) that this core's q rows are
    // in DRAM
    for (uint32_t i = 0; i < QF_NC; ++i) {
        const uint64_t sem =
            get_noc_addr(get_arg_val<uint32_t>(3 + 2 * i), get_arg_val<uint32_t>(4 + 2 * i), get_semaphore(0));
        noc_semaphore_inc(sem, 1);
    }
    noc_async_atomic_barrier();
#endif
    o.pop_front(2);
}
