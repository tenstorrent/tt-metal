// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llk_pack_common.h"
void narrow(const std::uint32_t v)
{
    ckernel::debug_mailbox_base[3] = v;
    ckernel::debug_mailbox_base[1] = v;
    reinterpret_cast<volatile std::uint8_t *>(0xFFEC0000)[5] = 1;
    reinterpret_cast<volatile std::uint16_t *>(0xFFEC0000)[3] = 1;
    reinterpret_cast<volatile std::uint64_t *>(0xFFEC0000)[1] = 1;
}
