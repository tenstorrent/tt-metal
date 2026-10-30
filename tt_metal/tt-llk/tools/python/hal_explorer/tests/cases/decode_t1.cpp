// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llk_pack_common.h"

void t_setdma(const std::uint32_t a, const std::uint32_t idx)
{
    TT_SETDMAREG(0, a & 0xffffu, 0, 5);
    TT_SETDMAREG(0, a >> 16, 0, idx);
    TT_SETDMAREG(0, 0x1234, 0, idx * 2 + 1);
}
void t_scalar_set(const std::uint32_t v, const std::uint32_t g)
{
    scalar::set(hal::gpr<p_gpr_unpack::TMP0>(), v);
    scalar::set(hal::gpr(g), v);
}
void t_scalar_add(const std::uint32_t g, const std::uint32_t h)
{
    scalar::add(hal::gpr(g), hal::gpr<p_gpr_unpack::TMP0>(), hal::gpr(h));
    scalar::add(hal::gpr<p_gpr_unpack::TMP0>(), hal::gpr(g), scalar::immediate(h));
}
