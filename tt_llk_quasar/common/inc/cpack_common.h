// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"

namespace ckernel::pack
{
constexpr static std::uint32_t TRISC_ID = 2;
static std::uint32_t clear_dest_bank_id = 0;

inline void _update_clear_dest_bank_id_()
{
    clear_dest_bank_id = 1 - clear_dest_bank_id;
}
} // namespace ckernel::pack
