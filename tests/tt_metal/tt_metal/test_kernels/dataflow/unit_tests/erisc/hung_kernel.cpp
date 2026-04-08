// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    RISC_POST_STATUS(0xdeadCAFE);
    risc_context_switch();
    RISC_POST_STATUS(0xdeadFEAD);
    // while(get_arg_val<uint32_t>(0) != 1) {
    //    RISC_POST_STATUS(0xabcd0213);
    //     check_and_context_switch();
    // }
}
