// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert: one core writes the chip's wall clock (64 bits) into page `slot` of a DRAM tensor (64-B pages), staged in
// a 64-B aligned L1 scratch (DRAM transfers need the L1 address congruent to the DRAM one mod 64). Placed between
// programs on a command queue (which run one after another), the clocks' differences are the programs' times between
// them, dispatch included. RT: 0 the tensor's address, 1 slot, 2 the scratch. CT: TensorAccessorArgs(tensor).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    const auto ts = TensorAccessor(args, get_arg_val<uint32_t>(0), 64);
    const uint32_t src = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    w[0] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    w[1] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    noc_async_write(src, ts.get_noc_addr(get_arg_val<uint32_t>(1)), 64);
    noc_async_write_barrier();
}
