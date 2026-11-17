// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One core writes the chip's wall clock into a stamp word: placed between programs on a command queue (which run one
// after another), the stamps' differences are the device time of the programs between them.
// RT: 0 stamps addr, 1 word.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    s[get_arg_val<uint32_t>(1)] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
}
