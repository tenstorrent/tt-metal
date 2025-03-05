// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_addr_map.h"
#include "ckernel_globals.h"
#include "ckernel_main.h"
#include "ckernel_pcbuf.h"
// Necessary for ckernel variables
#include "ckernel_helper.h"
#include "params.h"

int main()
{
#ifdef LLK_TRISC_UNPACK
    volatile std::uint32_t* mailbox = reinterpret_cast<volatile std::uint32_t*>(0x19FFC);
#elif defined(LLK_TRISC_MATH)
    volatile std::uint32_t* mailbox = reinterpret_cast<volatile std::uint32_t*>(0x19FF8);
#elif defined(LLK_TRISC_PACK)
    volatile std::uint32_t* mailbox = reinterpret_cast<volatile std::uint32_t*>(0x19FF4);
#endif
    *mailbox = 0x2; // write value different than 1 to mailbox to indicate kernel is running

    std::fill(ckernel::regfile, ckernel::regfile + 64, 0);

    ckernel::reset_cfg_state_id();
    ckernel::reset_dest_offset_id();
    ckernel::tensix_sync();

    run_kernel();

    *mailbox = ckernel::KERNEL_COMPLETE; // 0x1

    // Use a volatile variable to prevent the compiler from optimizing away the loop
    volatile bool run = true;

    // Infinite loop
    while (run)
    {
    }
}
