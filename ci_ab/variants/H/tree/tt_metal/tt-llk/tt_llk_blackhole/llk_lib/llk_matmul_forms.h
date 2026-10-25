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
    "....h", "r.r.R", "rrrrR", "..r.r", // k 1, LoFi
    ".R..R", "rr..R", "rrRrR", "rrrrr", // k 1, HiFi2
    "rr.rR", "rrrrR", "rrRrR", "rrrrr", // k 1, HiFi3
    "rR.rR", "rrrrR", "rrrrr", "rrrrr", // k 1, HiFi4
    "rh..R", "rrrrR", "RRrrR", "r.rrR", // k 2-4, LoFi
    "RRRRR", "rRrrR", "RRRRR", "rRrRR", // k 2-4, HiFi2
    "rRRrR", "rRrrR", "rRRRR", "rRRrR", // k 2-4, HiFi3
    "RRRRR", "RRRRR", "rrrrr", "rrrrr", // k 2-4, HiFi4
    "RRRRR", "RRRRR", "RRRRR", "RRrRR", // k 5+, LoFi
    "RRRRR", "RRRRR", "RRRRR", "RRRRR", // k 5+, HiFi2
    "RRRRR", "RRRRR", "RRRRR", "RRRRR", // k 5+, HiFi3
    "RRRRR", "RRRRR", "RrrRr", "rrrrr", // k 5+, HiFi4
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
// Cells where the keys of the ttnn matmul kernels (the format each DEST section is packed to, the row MOP rule) pick a form that reads
// slower than main's tile MOP and per-row unpack on the tt-llk sweep (perf_matmul_forms: every form of every perf_matmul Half cell,
// ABBA against main, no fused post flip). Those cells keep main's form. A cell is the math fidelity, whether the contraction of a section
// is one tile, a 32-bit DEST, one input format (of in0 or in1), the output format and the format the section is packed to.
// several_blocks_ok: the cell keeps the keyed form when the kernel runs several inner blocks. Bfp8_b input to Float32 output from a 16-bit DEST
// with Float16_b partials (packer_l1_acc) at HiFi3 and HiFi4 with one tile per section: the last section reads slower than main (up to +1.2 cy
// per tile) but the partial sections gain more, so two or more inner blocks are faster (ttnn, HiFi3, 4 inner blocks: 1.2 % faster than main).
struct matmul_main_form_cell
{
    MathFidelity fidelity;
    bool k1;
    bool fp32_dest;
    DataFormat in;
    DataFormat out;
    DataFormat pack;
    bool several_blocks_ok;
    std::uint32_t min_row = 0;
    std::uint32_t max_row = 0xFFFF;
    int sync = -1; // 0 half, 1 full, -1 any
    std::uint32_t min_k = 0;
    std::uint32_t max_k = 0xFFFF;
    std::uint32_t min_dim = 0; // the smaller side of the block, in tiles
};

inline constexpr matmul_main_form_cell matmul_main_form_cells[] = {
    {MathFidelity::HiFi2, true, false, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::HiFi2, true, false, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float16_b, false},
    {MathFidelity::HiFi2, true, false, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float16_b, false},
    {MathFidelity::HiFi2, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::HiFi2, true, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, false},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::HiFi2, true, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float16_b, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Bfp8_b, DataFormat::Bfp8_b, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Bfp8_b, DataFormat::Float16_b, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float32, false},
    {MathFidelity::HiFi2, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, false},
    {MathFidelity::HiFi3, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, true},
    {MathFidelity::HiFi3, true, false, DataFormat::Float16_b, DataFormat::Float32, DataFormat::Float16_b, false},
    {MathFidelity::HiFi3, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Float32, false},
    {MathFidelity::HiFi3, true, true, DataFormat::Float32, DataFormat::Float16_b, DataFormat::Float32, false},
    {MathFidelity::HiFi3, true, true, DataFormat::Float32, DataFormat::Float32, DataFormat::Float32, false},
    {MathFidelity::HiFi4, true, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, true},
    {MathFidelity::LoFi, false, false, DataFormat::Bfp8_b, DataFormat::Float32, DataFormat::Float16_b, false},
    {MathFidelity::LoFi, false, false, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false, 7},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false},
    {MathFidelity::LoFi, false, false, DataFormat::Float16_b, DataFormat::Bfp8_b, DataFormat::Float16_b, false},
    // LoFi at full sync: the row MOP of a Float32 input to Bfp8_b with a 32-bit DEST reads 2.6 cycles slower than the tile form on four row blocks
    {MathFidelity::LoFi, false, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false, 4, 4, 1, 2, 4},
};

template <MathFidelity math_fidelity, std::uint32_t kt_per_section, bool fp32_dest, DataFormat in0, DataFormat in1, DataFormat out_format, DataFormat pack_format, bool several_blocks, std::uint32_t row_tiles = 0, int sync = 0, std::uint32_t min_side = 99>
constexpr bool matmul_main_form_cell_selected()
{
    for (const matmul_main_form_cell& c : matmul_main_form_cells)
    {
        if (c.fidelity == math_fidelity && c.k1 == (kt_per_section == 1) && c.fp32_dest == fp32_dest && c.out == out_format && c.pack == pack_format &&
            (c.in == in0 || c.in == in1) && !(c.several_blocks_ok && several_blocks) && row_tiles >= c.min_row && row_tiles <= c.max_row && (c.sync < 0 || c.sync == sync) &&
            kt_per_section >= c.min_k && kt_per_section <= c.max_k && min_side >= c.min_dim)
        {
            return true;
        }
    }
    return false;
}

// Cells whose rows read slower than main's with the keyed form but faster with the hoisted rows form 'R' ('r' per row unpack
// loses there); the kernels key them with Bfp4_b, which the table reads as 'R' at k 1, LoFi, half sync and a 32-bit DEST.
inline constexpr matmul_main_form_cell matmul_forced_r_cells[] = {
    // Float32 input to Bfp8_b at LoFi, one k step, half sync and a 32-bit DEST: the 2x2 block reads 1.5 cycles behind main on the per row unpack;
    // the four tile row and column blocks lose with the hoisted form, so only blocks with both sides of two tiles or more
    {MathFidelity::LoFi, true, true, DataFormat::Float32, DataFormat::Bfp8_b, DataFormat::Bfp8_b, false, 0, 0xFFFF, 0, 0, 0xFFFF, 2},
};

template <MathFidelity math_fidelity, std::uint32_t kt_per_section, bool fp32_dest, DataFormat in0, DataFormat in1, DataFormat out_format, DataFormat pack_format, bool several_blocks, std::uint32_t row_tiles, int sync, std::uint32_t min_side>
constexpr bool matmul_forced_r_cell_selected()
{
    for (const matmul_main_form_cell& c : matmul_forced_r_cells)
    {
        if (c.fidelity == math_fidelity && c.k1 == (kt_per_section == 1) && c.fp32_dest == fp32_dest && c.out == out_format && c.pack == pack_format &&
            (c.in == in0 || c.in == in1) && !(c.several_blocks_ok && several_blocks) && row_tiles >= c.min_row && row_tiles <= c.max_row &&
            (c.sync < 0 || c.sync == sync) && kt_per_section >= c.min_k && kt_per_section <= c.max_k && min_side >= c.min_dim)
        {
            return true;
        }
    }
    return false;
}

static_assert(matmul_default_form<MathFidelity::LoFi, DstSync::SyncHalf, true, 1, DataFormat::Bfp4_b>() == 'R', "the forced hoisted rows cells key as Bfp4_b");

/**
 * @brief The math fidelity whose phases cover every nonzero mantissa slice of the two matmul operands.
 *
 * A phase multiplies one slice of each operand: SrcA hi {1, m[9:6]} or lo m[5:1], SrcB hi {1, m[9:4]} or lo {m[3:0], 000}
 * (tt_fpu_tile.sv). An in1 (SrcA) of Bfp4_b has 2 fraction bits, so its lo slice is zero and phase 1 adds exactly 0;
 * an in0 (SrcB) of Bfp8_b or Bfp4_b has at most 6 fraction bits, so its lo slice is zero and phase 2 adds exactly 0;
 * phase 3 needs both lo slices. Only a dropped suffix maps to an existing fidelity, so HiFi3 and HiFi4 with a narrow
 * in1 and a wide in0 keep their phases.
 *
 * Only equal operand formats are keyed; Invalid (unknown, the default of the compute API) keeps the requested fidelity.
 * check_out is set by the compute API, which knows the output format and DEST width; the harness keys that cell in python.
 *
 * @tparam math_fidelity: Requested math fidelity.
 * @tparam in0_format: L1 data format of in0 (unpacked to SrcB).
 * @tparam in1_format: L1 data format of in1 (unpacked to SrcA).
 */
template <
    MathFidelity math_fidelity,
    DataFormat in0_format,
    DataFormat in1_format,
    DataFormat out_format = DataFormat::Invalid,
    bool fp32_dest        = false,
    int throttle_level    = 0,
    bool check_out        = false>
constexpr MathFidelity matmul_phase_fidelity()
{
    constexpr bool lo_a_zero = in1_format == DataFormat::Bfp4_b;
    constexpr bool lo_b_zero = in0_format == DataFormat::Bfp8_b || in0_format == DataFormat::Bfp4_b;
    // Bfp8_b x Bfp8_b at HiFi3 and HiFi4 with a 32-bit DEST or a Float32 output (or an unknown output when check_out) is pack bound and
    // measured slower with the phases dropped, so it keeps the requested fidelity
    constexpr bool keep = check_out && in0_format == DataFormat::Bfp8_b && in1_format == DataFormat::Bfp8_b && throttle_level == 0 &&
                          (math_fidelity == MathFidelity::HiFi3 || math_fidelity == MathFidelity::HiFi4) &&
                          (fp32_dest || out_format == DataFormat::Float32 || out_format == DataFormat::Invalid);
    if constexpr (in0_format != in1_format || in0_format == DataFormat::Invalid || keep)
    {
        return math_fidelity; // mixed or unknown operand formats (bits not proven on hardware), or a cell that keeps main's fidelity
    }
    else if constexpr (lo_a_zero && lo_b_zero)
    {
        return MathFidelity::LoFi;
    }
    else if constexpr (lo_b_zero)
    {
        return math_fidelity == MathFidelity::LoFi ? MathFidelity::LoFi : MathFidelity::HiFi2;
    }
    else if constexpr (lo_a_zero)
    {
        return math_fidelity == MathFidelity::HiFi2 ? MathFidelity::LoFi : math_fidelity;
    }
    else
    {
        return math_fidelity;
    }
}
} // namespace ckernel
