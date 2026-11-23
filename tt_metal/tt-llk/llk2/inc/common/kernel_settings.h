// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Kernel-wide settings: the Dest mode, the math fidelity and the Dest ring's sections. Compile-time constants: a
// kernel has one of each from its first call to its last.
//
// A kernel sets them with defines before including any LLK 2.0 header, or takes the defaults (0, 0, 2). A value
// outside the ones listed fails to compile here, with the define's name:
//   LLK2_DEST_FP32       1: Dest holds 32-bit values; 0: 16-bit
//   LLK2_MATH_FIDELITY   0 LoFi, 2 HiFi2, 3 HiFi3, 4 HiFi4 (the value of ckernel::MathFidelity)
//   LLK2_DEST_SECTIONS   sections of the kernel's one Dest ring (compute_api/dest.h): 2 (SyncHalf) or 1

#pragma once

#include <cstdint>

#include "tensix_types.h" // DataFormat

#ifndef LLK2_DEST_FP32
#define LLK2_DEST_FP32 0
#endif

#ifndef LLK2_MATH_FIDELITY
#define LLK2_MATH_FIDELITY 0
#endif

#ifndef LLK2_DEST_SECTIONS
#define LLK2_DEST_SECTIONS 2
#endif

namespace llk2
{

/// The settings every Compute API call of a kernel shares.
struct KernelSettings
{
    bool dest_fp32;              ///< Dest holds 32-bit datums (LLK2_DEST_FP32)
    std::uint32_t math_fidelity; ///< ckernel::MathFidelity value: 0 LoFi (one phase), else the number of fidelity phases (2..4)
};

/// KernelSettings::math_fidelity of LoFi (ckernel::MathFidelity::LoFi): one fidelity phase. What an op that does not
/// multiply mantissas in phases (add, sub, a max reduce) runs with, whatever the kernel's fidelity.
inline constexpr std::uint32_t math_fidelity_lofi = 0;

/// The kernel's settings, from LLK2_DEST_FP32 and LLK2_MATH_FIDELITY.
inline constexpr KernelSettings kernel_settings {LLK2_DEST_FP32 != 0, LLK2_MATH_FIDELITY};
static_assert(
    kernel_settings.math_fidelity == math_fidelity_lofi || (kernel_settings.math_fidelity >= 2 && kernel_settings.math_fidelity <= 4),
    "LLK2_MATH_FIDELITY is 0 (LoFi), 2, 3 or 4 (HiFi2, HiFi3, HiFi4)");

/// The format of a Dest tensor in s's Dest mode: FP32 with a 32-bit Dest, else BF16.
constexpr DataFormat dest_format_for(const KernelSettings& s)
{
    return s.dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;
}

/// Whether `format` is a Dest format of the Dest mode: 32-bit datums (FP32, INT32) or 16-bit ones (FP16, BF16, UINT16).
/// The one definition: a Dest ring's reserve_back (compute_api/dest.h) and pack's accepts both use it. An op may take
/// fewer (the SFPU functions take dest_format_for only).
constexpr bool fits_dest_mode(DataFormat format, bool dest_fp32)
{
    if (dest_fp32)
    {
        return format == DataFormat::Float32 || format == DataFormat::Int32;
    }
    return format == DataFormat::Float16 || format == DataFormat::Float16_b || format == DataFormat::UInt16;
}

/// The kernel's Dest format: what its dst_specs are in.
inline constexpr DataFormat kernel_dest_format = dest_format_for(kernel_settings);

/// Sections of the kernel's one Dest ring (LLK2_DEST_SECTIONS): every DestRing of the kernel is a view of that ring.
inline constexpr std::uint32_t kernel_dest_sections = LLK2_DEST_SECTIONS;
static_assert(kernel_dest_sections == 1 || kernel_dest_sections == 2, "LLK2_DEST_SECTIONS is 1 or 2");

} // namespace llk2
