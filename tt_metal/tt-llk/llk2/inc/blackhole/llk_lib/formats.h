// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, Blackhole: a DataFormat as the format id the unpacker, packer and ALU format fields take. Shared by
// unpack.h and pack.h.
//
// Not HAL yet: masked_data_format (ckernel_defs.h).

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "tensix_types.h" // DataFormat

namespace llk2::llk_lib
{

/// The format id a THCON or ALU format field takes for f: its DataFormat value with the high flag bits masked off
/// (ckernel::masked_data_format).
constexpr std::uint32_t reg_fmt(DataFormat f)
{
    return ckernel::masked_data_format(static_cast<std::uint32_t>(f));
}

} // namespace llk2::llk_lib
