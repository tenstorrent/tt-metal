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
#include "profiler.h"

#ifdef LLK_PROFILER

namespace llk_profiler
{
barrier_ptr_t barrier_ptr = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
buffer_ptr_t buffer       = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
uint32_t write_idx        = 0;
uint32_t open_zone_cnt    = 0;

} // namespace llk_profiler

#endif

int main()
{
    volatile std::uint64_t* TIMESTAMP_ADDRESS = reinterpret_cast<volatile std::uint64_t*>(0x19000);
#if defined(LLK_TRISC_UNPACK)
    const std::uint32_t core_idx          = 0;
    volatile std::uint32_t* const mailbox = reinterpret_cast<volatile std::uint32_t*>(0x19FFC);
#elif defined(LLK_TRISC_MATH)
    const std::uint32_t core_idx          = 1;
    volatile std::uint32_t* const mailbox = reinterpret_cast<volatile std::uint32_t*>(0x19FF8);
#elif defined(LLK_TRISC_PACK)
    const std::uint32_t core_idx          = 2;
    volatile std::uint32_t* const mailbox = reinterpret_cast<volatile std::uint32_t*>(0x19FF4);
#endif
    std::uint64_t wall_clock = ckernel::read_wall_clock();

    *(TIMESTAMP_ADDRESS + core_idx * 2) = wall_clock;

    std::fill(ckernel::regfile, ckernel::regfile + 64, 0);
    ckernel::reset_cfg_state_id();
    ckernel::reset_dest_offset_id();

#if defined(LLK_PROFILER)
    llk_profiler::reset();
    llk_profiler::sync_threads();
#endif

    {
        ZONE_SCOPED("KERNEL")
        run_kernel();
        ckernel::tensix_sync();
    }

    *mailbox = ckernel::KERNEL_COMPLETE; // 0x1

    // Use a volatile variable to prevent the compiler from optimizing away the loop
    volatile bool run = true;

    // Infinite loop
    while (run)
    {
    }
}
