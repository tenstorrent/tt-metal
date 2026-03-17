// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifdef LLK_BOOT_MODE_BRISC
#include "boot.h"
#endif

extern "C" __attribute__((section(".init"), naked, noreturn)) std::uint32_t _start()
{
    do_crt0();
#ifdef LLK_BOOT_MODE_BRISC
    device_setup();

    // Reset profiler barrier before releasing T[0-2] from reset
    volatile std::uint32_t *ptr = (std::uint32_t *)0x16AFF4;
    ckernel::store_blocking(ptr, 0);
    ckernel::store_blocking(ptr + 1, 0);
    ckernel::store_blocking(ptr + 2, 0);

    // Release reset of triscs here in order to achieve brisc <-> trisc synchronization
    clear_trisc_soft_reset();
#endif

    for (;;)
    {
    } // Loop forever
}
