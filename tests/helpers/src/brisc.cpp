// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifdef LLK_BOOT_MODE_BRISC
#include "boot.h"
#endif

int main()
{
#ifdef LLK_BOOT_MODE_BRISC
    device_setup();

    // Release reset of triscs here in order to achieve brisc <-> trisc synchronization
    clear_trisc_soft_reset();
#endif
}
