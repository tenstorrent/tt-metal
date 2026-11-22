// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// M2 weight staging (tt/mk/stage.py): copies bytes [off, off + n) of this core's contiguous weight range in DRAM bank
// `bank` (tile slot `slot`, page = slot * n_banks + bank) into this core's L1 staging shard at the same offset.
// Runs as an extra data-movement kernel inside another program (on cores that program does not use), so the copy
// overlaps that program's compute. Reads in up to 16 KB pieces, 4 in flight.
// Runtime args: weight address, bank, first tile slot, staging L1 address, byte offset, bytes.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t nb = get_named_compile_time_arg_val("n_banks");
constexpr uint32_t PIECE = 16384, IN_FLIGHT = 4;

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(1);
    const uint32_t slot = get_arg_val<uint32_t>(2);
    const uint32_t stage = get_arg_val<uint32_t>(3);
    const uint32_t off = get_arg_val<uint32_t>(4);
    const uint32_t n = get_arg_val<uint32_t>(5);
    constexpr auto w_args = TensorAccessorArgs<0>();
    const auto ws = TensorAccessor(w_args, w_addr);
    const uint64_t src = ws.get_noc_addr(slot * nb + bank);  // the range is contiguous in the bank
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    uint32_t issued = 0;
    for (uint32_t o = 0; o < n; o += PIECE) {
        const uint32_t len = n - o < PIECE ? n - o : PIECE;
        if (issued >= IN_FLIGHT) {
            noc_async_read_barrier_with_trid((issued - IN_FLIGHT) % IN_FLIGHT + 1);
        }
        noc_async_read_set_trid(issued % IN_FLIGHT + 1);
        noc_async_read(src + off + o, stage + off + o, len);
        ++issued;
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
}
