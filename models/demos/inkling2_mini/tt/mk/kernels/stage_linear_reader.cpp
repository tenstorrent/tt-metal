// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Staged copy of tt/fused/kernels/bank_linear_reader.cpp (M2): the first `staged` bytes of this core's weight range
// were copied earlier into this core's L1 staging shard (tt/mk/stage.py, while another program computed); they are
// read from there, the rest from DRAM as before. Same bytes into the same cb_w slots, so the output is unchanged.
// Bank-streamed decode linear, reader (NCRISC, NOC0), one core next to DRAM bank `bank` (tt/fused/bank_linear.py).
// Streams this core's weight tiles, stored contiguous in its own bank in the order compute eats them: per pass nblk
// blocks of kb * ncq tiles, each into one fixed slot of kb * max_ncq tiles of cb_w (as the reference's BT), up to 3
// blocks in flight (transaction ids 1..4). The activation comes in on the writer's NOC.
// Runtime args: weight address, bank, first tile slot, staging L1 address, staged bytes, number of passes, ncq per
// pass.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_w = 1;
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");
constexpr uint32_t nblk = get_named_compile_time_arg_val("nblk");
constexpr uint32_t nb = get_named_compile_time_arg_val("n_banks");
constexpr uint32_t max_ncq = get_named_compile_time_arg_val("max_ncq");
constexpr uint32_t SLOT = kb * max_ncq;
constexpr uint32_t NSLOT = 4, IN_FLIGHT = 3;

void kernel_main() {
    uint32_t a = 0;
    const uint32_t w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t bank = get_arg_val<uint32_t>(a++);
    uint32_t slot = get_arg_val<uint32_t>(a++);
    const uint32_t stage = get_arg_val<uint32_t>(a++);
    const uint32_t staged = get_arg_val<uint32_t>(a++);
    const uint32_t npass = get_arg_val<uint32_t>(a++);
    uint32_t done = 0;  // bytes of this core's range already handed out

    constexpr auto w_args = TensorAccessorArgs<0>();
    const auto ws = TensorAccessor(w_args, w_addr);

    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    const uint32_t tb = get_tile_size(cb_w);
    const uint32_t base = get_write_ptr(cb_w);
    uint32_t issued = 0, retired = 0;  // blocks
    for (uint32_t q = 0; q < npass; ++q) {
        const uint32_t ncq = get_arg_val<uint32_t>(a + q);
        const uint32_t bytes = kb * ncq * tb;
        for (uint32_t p = 0; p < nblk; ++p) {
            cb_reserve_back(cb_w, SLOT * (issued - retired + 1));  // this block's slot plus the ones in flight
            noc_async_read_set_trid(issued % NSLOT + 1);
            const uint32_t dst = base + (issued % NSLOT) * SLOT * tb;
            if (done + bytes <= staged) {
                noc_async_read(get_noc_addr(stage + done), dst, bytes);  // local L1 staging
            } else if (done >= staged) {
                noc_async_read(ws.get_noc_addr(slot * nb + bank), dst, bytes);
            } else {  // block straddles the end of the staged prefix
                const uint32_t s0 = staged - done;
                noc_async_read(get_noc_addr(stage + done), dst, s0);
                noc_async_read(ws.get_noc_addr(slot * nb + bank) + s0, dst + s0, bytes - s0);
            }
            done += bytes;
            slot += kb * ncq;
            ++issued;
            if (issued - retired == IN_FLIGHT) {
                noc_async_read_barrier_with_trid(retired % NSLOT + 1);
                cb_push_back(cb_w, SLOT);
                ++retired;
            }
        }
    }
    while (retired < issued) {
        noc_async_read_barrier_with_trid(retired % NSLOT + 1);
        cb_push_back(cb_w, SLOT);
        ++retired;
    }
    noc_async_read_set_trid(0);
}
