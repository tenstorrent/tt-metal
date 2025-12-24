// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

extern uint32_t rta_count;
extern uint32_t crta_count;

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    uint32_t l1_write_addr = get_write_ptr(cb_in0);
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(l1_write_addr);

    // Write all args to L1 to check
    ptr[0] = get_arg_val<uint32_t>(0);
    ptr[1] = get_arg_val<uint32_t>(1);
    ptr[2] = get_arg_val<uint32_t>(2);
    ptr[3] = get_arg_val<uint32_t>(3);
    ptr[4] = get_arg_val<uint32_t>(4);
    ptr[5] = get_arg_val<uint32_t>(5);
    uint32_t crta = get_common_arg_val<uint32_t>(0);
    DPRINT << HEX() << "rta_count : " << rta_count << '\n';
    DPRINT << HEX() << "get_arg_val<uint32_t>(0) : " << get_arg_val<uint32_t>(0) << '\n';
    DPRINT << HEX() << "get_arg_val<uint32_t>(1) : " << get_arg_val<uint32_t>(1) << '\n';
    DPRINT << HEX() << "get_arg_val<uint32_t>(2) : " << get_arg_val<uint32_t>(2) << '\n';
    DPRINT << HEX() << "get_arg_val<uint32_t>(3) : " << get_arg_val<uint32_t>(3) << '\n';
    DPRINT << HEX() << "get_arg_val<uint32_t>(4) : " << get_arg_val<uint32_t>(4) << '\n';
    DPRINT << HEX() << "get_arg_val<uint32_t>(5) : " << get_arg_val<uint32_t>(5) << '\n';

    DPRINT << '\n';
    DPRINT << HEX() << "crta_count                     : " << crta_count << '\n';
    DPRINT << HEX() << "get_common_arg_val<uint32_t>(0): " << get_common_arg_val<uint32_t>(0) << '\n';
    DPRINT << HEX() << "get_common_arg_val<uint32_t>(1): " << get_common_arg_val<uint32_t>(1) << '\n';
    DPRINT << HEX() << "get_common_arg_val<uint32_t>(2): " << get_common_arg_val<uint32_t>(2) << '\n';
    DPRINT << HEX() << "get_common_arg_val<uint32_t>(3): " << get_common_arg_val<uint32_t>(3) << '\n';
    DPRINT << HEX() << "get_common_arg_val<uint32_t>(4): " << get_common_arg_val<uint32_t>(4) << '\n';
    DPRINT << HEX() << "get_common_arg_val<uint32_t>(5): " << get_common_arg_val<uint32_t>(5) << '\n';
    DPRINT << HEX() << "get_common_arg_val<uint32_t>(6): " << get_common_arg_val<uint32_t>(6) << '\n';
    DPRINT << HEX() << "get_common_arg_val<uint32_t>(7): " << get_common_arg_val<uint32_t>(7) << '\n';

    // (uintptr_t)&rta_l1_base[arg_idx]
}
