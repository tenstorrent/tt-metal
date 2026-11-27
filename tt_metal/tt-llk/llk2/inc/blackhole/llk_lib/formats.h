// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, Blackhole: a DataFormat as the format id the unpacker, packer and ALU format fields take. Shared by
// unpack.h and pack.h.
//
// The Src zero-flag rule (src_zero_flag_disabled) is llk-lib's own: the HAL has the flag, not the rule.

#pragma once

#include <cstdint>

#include "llk_lib/platform/format.h"
#include "tensix_types.h" // DataFormat

namespace llk2::llk_lib
{

/// The format id a THCON or ALU format field takes for f (platform::format::id).
constexpr std::uint32_t format_id(DataFormat f)
{
    return platform::format::id(f);
}

/// Whether an FPU op whose Src registers hold formats a and b needs the Src zero flag disabled (every datum kept on
/// its way through Src): only with a UInt16 operand. ckernel::requires_disabled_src_zero_flag (ckernel_defs.h) as a
/// constant expression: the 1.0 function is not constexpr, and a value known only after inlining is written in the
/// run-time form (an instruction-buffer store instead of an immediate instruction).
constexpr bool src_zero_flag_disabled(DataFormat a, DataFormat b)
{
    return a == DataFormat::UInt16 || b == DataFormat::UInt16;
}

} // namespace llk2::llk_lib
