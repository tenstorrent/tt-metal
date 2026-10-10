// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "tensix_types.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// A Dst-to-Dst copy must not convert. GetSfpLoadStoreInstrMod gives the float formats a conversion
// mode, and SFPSTORE flushes denormals in that conversion, so a float-mode copy drops every subnormal.
// Map each conversion mode, DEFAULT included (bfp* and Lf8 get it), onto the opaque integer mode of the
// Dst word width, as the where kernel does by hand for Float16_b. Integer formats pass through, and the
// mode is an immediate in the same SFPLOAD / SFPSTORE pair, so this costs nothing.
template <DataFormat DATA_FORMAT, bool is_fp32_dest_acc_en>
constexpr InstrModLoadStore GetSfpCopyInstrMod() {
    constexpr InstrModLoadStore conv = GetSfpLoadStoreInstrMod<DATA_FORMAT, is_fp32_dest_acc_en>();
    return (conv == InstrModLoadStore::FP32)                                        ? InstrModLoadStore::INT32
           : (conv == InstrModLoadStore::FP16A || conv == InstrModLoadStore::FP16B) ? InstrModLoadStore::LO16
           : (conv == InstrModLoadStore::DEFAULT)
               ? (is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16)
               : conv;
}

// The mapping, pinned. A copy of a 16-bit Dst word must move opaque 16 bits, and of a 32-bit word
// opaque 32 bits; nothing here may name a float mode.
static_assert(GetSfpCopyInstrMod<DataFormat::Float16_b, false /*is_fp32_dest_acc_en*/>() == InstrModLoadStore::LO16);
static_assert(GetSfpCopyInstrMod<DataFormat::Float16_b, true /*is_fp32_dest_acc_en*/>() == InstrModLoadStore::INT32);
static_assert(GetSfpCopyInstrMod<DataFormat::Float16, false /*is_fp32_dest_acc_en*/>() == InstrModLoadStore::LO16);
static_assert(GetSfpCopyInstrMod<DataFormat::Float16, true /*is_fp32_dest_acc_en*/>() == InstrModLoadStore::INT32);
static_assert(GetSfpCopyInstrMod<DataFormat::Float32, false /*is_fp32_dest_acc_en*/>() == InstrModLoadStore::INT32);
static_assert(GetSfpCopyInstrMod<DataFormat::Float32, true /*is_fp32_dest_acc_en*/>() == InstrModLoadStore::INT32);
static_assert(GetSfpCopyInstrMod<DataFormat::Bfp8_b, false /*is_fp32_dest_acc_en*/>() == InstrModLoadStore::LO16);
static_assert(GetSfpCopyInstrMod<DataFormat::Bfp8_b, true /*is_fp32_dest_acc_en*/>() == InstrModLoadStore::INT32);
static_assert(GetSfpCopyInstrMod<DataFormat::UInt16, false /*is_fp32_dest_acc_en*/>() == InstrModLoadStore::LO16);
static_assert(GetSfpCopyInstrMod<DataFormat::UInt16, true /*is_fp32_dest_acc_en*/>() == InstrModLoadStore::INT32);
static_assert(GetSfpCopyInstrMod<DataFormat::UInt32, false /*is_fp32_dest_acc_en*/>() == InstrModLoadStore::INT32);
static_assert(GetSfpCopyInstrMod<DataFormat::UInt32, true /*is_fp32_dest_acc_en*/>() == InstrModLoadStore::INT32);

// Generalized copy_dest_value that works with any DataFormat
template <DataFormat DATA_FORMAT, bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
void copy_dest_value(const uint dst_index_in, const uint dst_index_out, const uint /* unused */) {
    constexpr InstrModLoadStore instr_mod_index = GetSfpCopyInstrMod<DATA_FORMAT, is_fp32_dest_acc_en>();
    // size of each tile in Dest is 64 rows
    constexpr uint dst_tile_size = 64;
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

// Deprecated: Use the DataFormat template parameter version instead. This one still converts
// through sfpi::vFloat, so it flushes subnormals on copy.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
[[deprecated("Use copy_dest_value<DataFormat, APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en> instead")]]
void copy_dest_value(const uint dst_index_in, const uint dst_index_out, const uint /* unused */) {
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;
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
