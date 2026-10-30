// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llk_pack_common.h"
void zero_offset(const std::uint32_t v)
{
    ckernel::regfile[0]     = v;
    ckernel::pc_buf_base[0] = v;
    ckernel::mailbox_write(2, v);
}
std::uint32_t read_zero()
{
    return ckernel::regfile[0] + ckernel::mailbox_base[1][1];
}
void const_zero()
{
    *reinterpret_cast<volatile std::uint32_t *>(0xFFEF0000) = 3;
    *reinterpret_cast<volatile std::uint32_t *>(0xFFEC1000) = 3;
}
