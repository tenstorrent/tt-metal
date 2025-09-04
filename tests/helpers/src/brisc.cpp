// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "boot.h"
#include "ckernel.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_structs.h"
#include "dev_mem_map.h"
#include "risc_attribs.h"
#include "tensix.h"

int main()
{
#ifdef LLK_BOOT_MODE_BRISC
    device_setup();

    // Release reset of triscs here in order to achieve brisc <-> trisc synchronization
    clear_trisc_soft_reset();
#endif
}
