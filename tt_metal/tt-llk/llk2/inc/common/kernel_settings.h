// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel-wide settings: the Dest mode and the math fidelity. Compile time for now; production has to change the
// Dest mode mid-kernel (see the hackathon doc, "Dest mode").
//
// A kernel sets them with defines before including any LLK 2.0 header, or takes the defaults:
//   LLK2_DEST_FP32       1: Dest holds 32-bit values; 0: 16-bit
//   LLK2_MATH_FIDELITY   0 LoFi, 2 HiFi2, 3 HiFi3, 4 HiFi4 (the value of ckernel::MathFidelity)

#pragma once

#include <cstdint>

#ifndef LLK2_DEST_FP32
#define LLK2_DEST_FP32 0
#endif

#ifndef LLK2_MATH_FIDELITY
#define LLK2_MATH_FIDELITY 0
#endif

namespace llk2
{

struct KernelSettings
{
    bool dest_fp32;
    std::uint32_t math_fidelity; // phases - 1 in the low bits, as ckernel::MathFidelity
};

inline constexpr KernelSettings kernel_settings {LLK2_DEST_FP32 != 0, LLK2_MATH_FIDELITY};

} // namespace llk2
