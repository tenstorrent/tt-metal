// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B47b XPUSH worker (one core per DRAM bank b, NCRISC): re-lays the chip's T token rows (RM [T,1,1,6144] bf16, 12288 B
// per page) for the tp24 consumers: bank block b = columns 768 b .. 768 b + 767 = tile columns j = 0..23 of the std
// tiles, block layout [j][face f][row r][16 cols] bf16 (piece (j, f) = T rows x 32 B contiguous). The block (BLK = 1536
// T B) is NoC-written to the packet buffer (CB_BUF, same L1 address on every core) of the NC comm cores at byte b *
// BLK, then the comm cores' semaphore 0 is incremented. CT: 0 CB_SCR, 1 CB_BUF, 2 T, 3 NC (comm cores), then
// TensorAccessorArgs(x). Common: x addr. RT: b, then NC x (x, y).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(0);
    constexpr uint32_t CB_BUF = get_compile_time_arg_val(1);
    constexpr uint32_t T = get_compile_time_arg_val(2);
    constexpr uint32_t NCOMM = get_compile_time_arg_val(3);
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr uint32_t BLK = 1536 * T;
    const uint32_t b = get_arg_val<uint32_t>(0);
    const auto xa = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), 12288);
    const uint32_t scr = (get_write_ptr(CB_SCR) + 63u) & ~63u;  // staging [j][r][64 B]
    const uint32_t out = scr + 24 * T * 64;                     // block [j][f][r][32 B]
    {
        DeviceZoneScopedN("XW_READ");
        for (uint32_t r = 0; r < T; ++r) {
            for (uint32_t j = 0; j < 24; ++j) {
                noc_async_read(xa.get_noc_addr(r, 1536 * b + 64 * j), scr + (j * T + r) * 64, 64);
            }
        }
        noc_async_read_barrier();
    }
    {
        DeviceZoneScopedN(
            "XW_SHUF");  // face de-interleave by local NoC copies (32 B pieces; DRAM reads must stay 64 B aligned)
        for (uint32_t j = 0; j < 24; ++j) {
            for (uint32_t r = 0; r < T; ++r) {
                noc_async_write(scr + (j * T + r) * 64, get_noc_addr(out + ((j * 2) * T + r) * 32), 32);
                noc_async_write(scr + (j * T + r) * 64 + 32, get_noc_addr(out + ((j * 2 + 1) * T + r) * 32), 32);
            }
        }
        noc_async_write_barrier();
    }
    const uint32_t buf = get_write_ptr(CB_BUF);
    const uint32_t sem = get_semaphore(0);
    for (uint32_t c = 0; c < NCOMM; ++c) {
        const uint32_t cx = get_arg_val<uint32_t>(1 + 2 * c), cy = get_arg_val<uint32_t>(2 + 2 * c);
        noc_async_write(out, get_noc_addr(cx, cy, buf + b * BLK), BLK);
    }
    noc_async_write_barrier();
    for (uint32_t c = 0; c < NCOMM; ++c) {
        const uint32_t cx = get_arg_val<uint32_t>(1 + 2 * c), cy = get_arg_val<uint32_t>(2 + 2 * c);
        noc_semaphore_inc(get_noc_addr(cx, cy, sem), 1);
    }
    noc_async_atomic_barrier();
}
