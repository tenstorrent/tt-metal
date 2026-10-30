// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llk_pack_common.h"
void const_loop()
{
    for (std::uint32_t i = 0; i < 100; i++)
    {
        TT_SETDMAREG(0, i, 0, 5);
    }
}
