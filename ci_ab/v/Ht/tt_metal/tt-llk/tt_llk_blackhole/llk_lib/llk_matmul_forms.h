// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"

// Default matmul unpack and math forms for the keys a kernel fixes at compile time: k steps per DEST section (one call
// per k step), math fidelity, DEST sync, 32-bit DEST and output format. Each entry is the form that, on the tt-llk
// matmul sweep, runs fastest with no configuration slower than main beyond noise: '.' per-row unpack and tile MOP,
// 'r' per-row unpack and row MOP, 'h' hoisted unpack and tile MOP, 'R' hoisted unpack and row MOP. One string per
// (k class, fidelity, sync, DEST width), one character per output format (Float16_b, Float16, Float32, Bfp8_b, Bfp4_b).
// An unknown k (0) or another output format takes '.', the forms main runs.
namespace ckernel
{
inline constexpr const char* matmul_forms_table[48] = {
    ".r..h", "hh..R", "rRrrR", "..r.r", // k 1, LoFi
    ".R..R", "rR..R", "rRRrR", "RRrrr", // k 1, HiFi2
    "rr.rR", "rrrRR", "RRRRR", "rRrRr", // k 1, HiFi3
    "RR.RR", "RRRRR", "rrrrr", "rrrrr", // k 1, HiFi4
    ".h.rR", "rrrrR", "RRRrR", "..r.R", // k 2-4, LoFi
    "RRRRR", "RRRRR", "RRRRR", "RRRRR", // k 2-4, HiFi2
    "rRRrR", "rRrrR", "rRRRR", "rRRrR", // k 2-4, HiFi3
    "RRRRR", "RRRRR", "rrrrr", "rrrrr", // k 2-4, HiFi4
    "RRRRR", "RRRRR", "RRRRR", "RRrRR", // k 5+, LoFi
    "RRRRR", "RRRRR", "RRRRR", "RRRRR", // k 5+, HiFi2
    "RRRRR", "RRRRR", "RRRRR", "RRRRR", // k 5+, HiFi3
    "RRRRR", "RRRRR", "rrrrr", "rrrrr", // k 5+, HiFi4
};

template <MathFidelity math_fidelity, DstSync dst_sync, bool fp32_dest, std::uint32_t kt_per_section, DataFormat out_format>
constexpr char matmul_default_form()
{
    constexpr int fmt = out_format == DataFormat::Float16_b ? 0
                        : out_format == DataFormat::Float16 ? 1
                        : out_format == DataFormat::Float32 ? 2
                        : out_format == DataFormat::Bfp8_b  ? 3
                        : out_format == DataFormat::Bfp4_b  ? 4
                                                            : -1;
    if constexpr (kt_per_section == 0 || fmt < 0)
    {
        return '.';
    }
    else
    {
        constexpr std::uint32_t k_class = kt_per_section == 1 ? 0 : (kt_per_section <= 4 ? 1 : 2);
        constexpr std::uint32_t fid     = math_fidelity == MathFidelity::LoFi    ? 0
                                          : math_fidelity == MathFidelity::HiFi2 ? 1
                                          : math_fidelity == MathFidelity::HiFi3 ? 2
                                                                                 : 3;
        return matmul_forms_table[((k_class * 4 + fid) * 2 + (dst_sync == DstSync::SyncFull ? 1 : 0)) * 2 + (fp32_dest ? 1 : 0)][fmt];
    }
}
} // namespace ckernel
