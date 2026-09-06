// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "tensix_types.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// A Dst-to-Dst copy must not convert.
//
// GetSfpLoadStoreInstrMod picks a *type conversion* mode for the float formats (FP16A / FP16B /
// FP32), and SFPSTORE flushes denormals to zero as part of that conversion, so the float spelling
// silently drops every subnormal it is asked to copy: all 254 bf16 subnormals, measured identically
// on Blackhole and Wormhole. The ISA documentation names the escape - use an opaque integer mode
// instead of the float one - and the where kernel already does this by hand for Float16_b.
//
// So map each conversion mode onto the opaque integer mode of the same width. The bfp* and integer
// formats already carry opaque bits and pass through untouched. This costs nothing: the mode is a
// 4-bit immediate in the same SFPLOAD / SFPSTORE pair.
template <DataFormat DATA_FORMAT, bool is_fp32_dest_acc_en>
constexpr InstrModLoadStore GetSfpCopyInstrMod() {
    constexpr InstrModLoadStore conv = GetSfpLoadStoreInstrMod<DATA_FORMAT, is_fp32_dest_acc_en>();
    return (conv == InstrModLoadStore::FP32)                                        ? InstrModLoadStore::INT32
           : (conv == InstrModLoadStore::FP16A || conv == InstrModLoadStore::FP16B) ? InstrModLoadStore::LO16
                                                                                    : conv;
}

// The mapping, pinned. A copy of a 16-bit Dst word must move opaque 16 bits, and of a 32-bit word
// opaque 32 bits; nothing here may name a float mode.
static_assert(GetSfpCopyInstrMod<DataFormat::Float16_b, false>() == InstrModLoadStore::LO16);
static_assert(GetSfpCopyInstrMod<DataFormat::Float16_b, true>() == InstrModLoadStore::INT32);
static_assert(GetSfpCopyInstrMod<DataFormat::Float16, false>() == InstrModLoadStore::LO16);
static_assert(GetSfpCopyInstrMod<DataFormat::Float16, true>() == InstrModLoadStore::INT32);
static_assert(GetSfpCopyInstrMod<DataFormat::Float32, false>() == InstrModLoadStore::INT32);
static_assert(GetSfpCopyInstrMod<DataFormat::Bfp8_b, false>() == InstrModLoadStore::DEFAULT);

// Generalized copy_dest_value that works with any DataFormat
template <DataFormat DATA_FORMAT, bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
void copy_dest_value(
    const std::uint32_t dst_index_in, const std::uint32_t dst_index_out, const std::uint32_t /* unused */) {
    constexpr InstrModLoadStore instr_mod_index = GetSfpCopyInstrMod<DATA_FORMAT, is_fp32_dest_acc_en>();
    // size of each tile in Dest is 64 rows
    constexpr std::uint32_t dst_tile_size = 64;
    for (int d = 0; d < ITERATIONS; d++) {
        // For some reason using __builtin_rvtt_sfp{load,store} here
        // results in test failures.  The compiler unrolls this loop
        // and with the builtin emits assembly directly, rather than
        // synthesize the insn.  Presumably the same problem occurs
        // with using sfpi -- if it was extended to expose the
        // ADDR_MOD PR #41879
        TT_SFPLOAD(p_sfpu::LREG0, instr_mod_index, ADDR_MOD_3, dst_index_in * dst_tile_size);
        TT_SFPSTORE(p_sfpu::LREG0, instr_mod_index, ADDR_MOD_3, dst_index_out * dst_tile_size);
        dst_reg++;
    }
}

// Deprecated: Use the DataFormat template parameter version instead
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
[[deprecated("Use copy_dest_value<DataFormat, APPROXIMATION_MODE, ITERATIONS> instead")]]
void copy_dest_value(
    const std::uint32_t dst_index_in, const std::uint32_t dst_index_out, const std::uint32_t /* unused */) {
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] =
            sfpi::vFloat(sfpi::dst_reg[dst_index_in * dst_tile_size_sfpi]);
        dst_reg++;
    }
}

void copy_dest_value_init() {
    // No initialization required
}

}  // namespace sfpu
}  // namespace ckernel
