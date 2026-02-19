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
barrier_ptr_t barrier_ptr   = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
buffer_ptr_t buffer         = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
std::uint32_t write_idx     = 0;
std::uint32_t open_zone_cnt = 0;

} // namespace llk_profiler

#endif

extern const volatile struct RuntimeParams __runtime_args_start[];
extern void run_kernel(const volatile struct RuntimeParams* params);

int main()
{
#if defined(LLK_TRISC_UNPACK) && defined(LLK_BOOT_MODE_TRISC)
    device_setup();
    clear_trisc_soft_reset(); // Release the rest of the triscs
#endif

#ifdef COVERAGE
    constexpr std::uint32_t mailboxes_start = 0x63FC0;
#else
    constexpr std::uint32_t mailboxes_start = 0x1FFC0;
#endif

#if defined(LLK_TRISC_UNPACK)
    constexpr std::uint32_t mailbox_offset = sizeof(std::uint32_t);
#elif defined(LLK_TRISC_MATH)
    constexpr std::uint32_t mailbox_offset = 2 * sizeof(std::uint32_t);
#elif defined(LLK_TRISC_PACK)
    constexpr std::uint32_t mailbox_offset = 3 * sizeof(std::uint32_t);
#endif

    volatile std::uint32_t* const mailbox = reinterpret_cast<volatile std::uint32_t*>(mailboxes_start + mailbox_offset);
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
        run_kernel(__runtime_args_start);
        ckernel::tensix_sync();
    }

    *mailbox = ckernel::KERNEL_COMPLETE;
}
