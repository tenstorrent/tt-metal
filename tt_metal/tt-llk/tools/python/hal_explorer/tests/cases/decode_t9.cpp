// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llk_pack_common.h"
void loop_counter(const std::uint32_t n)
{
    for (std::uint32_t i = 0; i < n; i++)
    {
        TT_SETDMAREG(0, i, 0, 5);
    }
}
void loop_stride(const std::uint32_t n, const std::uint32_t base)
{
    std::uint32_t addr = base;
    for (std::uint32_t i = 0; i < n; i++)
    {
        TT_SETDMAREG(0, addr & 0xffff, 0, 7);
        addr += 16;
    }
}
