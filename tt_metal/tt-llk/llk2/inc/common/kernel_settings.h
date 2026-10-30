// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel-wide settings: the Dest mode and the math fidelity. Compile time for now; production has to change the
// Dest mode mid-kernel (see "Dest mode" in the design doc, https://claude.ai/artifact/B6gUhxaMbCzqyx3d3cJhFv).
//
// A kernel sets them with defines before including any LLK 2.0 header, or takes the defaults:
//   LLK2_DEST_FP32       1: Dest holds 32-bit values; 0: 16-bit
//   LLK2_MATH_FIDELITY   0 LoFi, 2 HiFi2, 3 HiFi3, 4 HiFi4 (the value of ckernel::MathFidelity)

#pragma once

#include <cstdint>

#include "tensix_types.h" // DataFormat

#ifndef LLK2_DEST_FP32
#define LLK2_DEST_FP32 0
#endif

#ifndef LLK2_MATH_FIDELITY
#define LLK2_MATH_FIDELITY 0
#endif

namespace llk2
{

/// The settings every Compute API call of a kernel shares.
struct KernelSettings
{
    bool dest_fp32;              ///< Dest holds 32-bit datums (LLK2_DEST_FP32)
    std::uint32_t math_fidelity; ///< ckernel::MathFidelity value: 0 LoFi (one phase), else the number of fidelity phases (2..4)
};

/// The kernel's settings, from LLK2_DEST_FP32 and LLK2_MATH_FIDELITY.
inline constexpr KernelSettings kernel_settings {LLK2_DEST_FP32 != 0, LLK2_MATH_FIDELITY};

/// The format of a Dest tensor in s's Dest mode: FP32 with a 32-bit Dest, else BF16.
constexpr DataFormat dest_format_for(const KernelSettings& s)
{
    return s.dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;
}

/// The kernel's Dest format: what its dst_specs are in.
inline constexpr DataFormat kernel_dest_format = dest_format_for(kernel_settings);

} // namespace llk2
