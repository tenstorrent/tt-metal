// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "tensix_types.h"
#include "llk_math_eltwise_sfpu_op.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Generalized copy_dest_value that works with any DataFormat
template <DataFormat DATA_FORMAT, bool APPROXIMATION_MODE, int ITERATIONS = 8>
void copy_dest_value(const uint dst_index_in, const uint dst_index_out, const uint /* unused */) {
    constexpr InstrModLoadStore instr_mod_index = GetSfpLoadStoreInstrMod<DATA_FORMAT>();
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

// Deprecated: Use the DataFormat template parameter version instead
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
[[deprecated("Use copy_dest_value<DataFormat, APPROXIMATION_MODE, ITERATIONS> instead")]]
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

// ---------------------------------------------------------------------------------------------------
// CopyDestValues<DATA_FORMAT, DST_SYNC, DST_ACCUM, APPROXIMATION_MODE, ITERATIONS>
//   calculate(in, out, 0 /*unused*/, vector_mode) -> copy_dest_value<DATA_FORMAT, ..>;  init() -> copy_dest_value_init
//   DATA_FORMAT = DataFormat::Invalid selects the deprecated format-agnostic copy_dest_value<APPROXIMATION_MODE, ..>
//   overload (sfpi::vFloat path) used by the deprecated non-templated copy_dest_values().
//   Backs copy_dest_values<DataFormat>, copy_dest_values, copy_dest_values_init (api/compute/copy_dest_values.h).
// ---------------------------------------------------------------------------------------------------
template <DataFormat DATA_FORMAT, DstSync DST_SYNC, bool DST_ACCUM, bool APPROXIMATION_MODE = false, int ITERATIONS = 8>
struct CopyDestValues : SfpuBinaryOp<
                            CopyDestValues<DATA_FORMAT, DST_SYNC, DST_ACCUM, APPROXIMATION_MODE, ITERATIONS>,
                            DST_SYNC,
                            DST_ACCUM> {
    static void kernel(uint32_t dst_index_in, uint32_t dst_index_out, uint32_t unused) {
        if constexpr (DATA_FORMAT == DataFormat::Invalid) {
            copy_dest_value<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out, unused);
        } else {
            copy_dest_value<DATA_FORMAT, APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out, unused);
        }
    }

    static void init_kernel() { copy_dest_value_init(); }
};
}  // namespace sfpu
}  // namespace ckernel
