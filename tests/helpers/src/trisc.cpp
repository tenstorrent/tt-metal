// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#ifndef ARCH_QUASAR
#include "ckernel_globals.h" // Only for WH/BH
// Necessary for ckernel variables
#include "ckernel_helper.h" // Only for WH/BH
#endif
#include "profiler.h"

#if defined(LLK_TRISC_UNPACK) && defined(LLK_BOOT_MODE_TRISC)
#include "boot.h"
#endif

#ifdef LLK_PROFILER

namespace llk_profiler
{
barrier_ptr_t barrier_ptr = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
buffer_ptr_t buffer       = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
uint32_t write_idx        = 0;
uint32_t open_zone_cnt    = 0;

} // namespace llk_profiler

#endif

__attribute__((weak)) void run_kernel()
{
    return;
}

int main()
{
#if defined(LLK_TRISC_UNPACK) && defined(LLK_BOOT_MODE_TRISC)
    device_setup();

    // Release the rest of the triscs
    clear_trisc_soft_reset();
#endif

#if defined(LLK_TRISC_UNPACK)
    volatile std::uint32_t* const mailbox = reinterpret_cast<volatile std::uint32_t*>(0x19FFC);
#elif defined(LLK_TRISC_MATH)
    volatile std::uint32_t* const mailbox = reinterpret_cast<volatile std::uint32_t*>(0x19FF8);
#elif defined(LLK_TRISC_PACK)
    volatile std::uint32_t* const mailbox = reinterpret_cast<volatile std::uint32_t*>(0x19FF4);
#endif

    std::fill(ckernel::regfile, ckernel::regfile + 64, 0);
#ifndef ARCH_QUASAR
    ckernel::reset_cfg_state_id();
    ckernel::reset_dest_offset_id();
#endif

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
}
