// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Experimental shim for the clean matmul API.
//
// Provides matmul_init / matmul_block / matmul_uninit — the target programming
// model — by wrapping existing Compute API functions (mm_init_short, matmul_tiles,
// etc.) that we can't rename yet without touching ~70 kernel files.
//
// Lives in experimental/ so the demo kernel can use the clean names today.
// Once the full Compute API cleanup lands, these wrappers get promoted to the
// main API and the old mm_* names are deleted.
//
// Target include path:
//   tt_metal/hw/inc/api/compute/experimental/matmul_api.h
// or:
//   ttnn/cpp/ttnn/kernel_lib/matmul_helpers.hpp

#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"

namespace compute_kernel_lib {

namespace matmul_config {

enum class ReconfigurePackMode : uint8_t {
    NoPackReconfigure,  // Default — only handle the operand swap if needed
    PackReconfigure     // Also reconfigure pack output to out_cb format
};

}  // namespace matmul_config

// ============================================================================
// Internal: compile-time format comparison using JIT-generated descriptor arrays
// ============================================================================
namespace detail {

// Returns true when in0_cb and in1_cb have different L1 (source) data formats,
// meaning the matmul operand swap (srcA=in1, srcB=in0) requires actual register
// writes. When formats match, the swap is a no-op.
template <uint32_t in0_cb, uint32_t in1_cb>
constexpr bool needs_format_swap() {
#if defined(UCK_CHLKC_PACK)
    // PACK TRISC never executes reconfig_data_format (gated by UNPACK()/MATH() macros).
    return false;
#else
    return unpack_src_format[in0_cb] != unpack_src_format[in1_cb];
#endif
}

}  // namespace detail

// clang-format off
/**
 * Initialize the compute engine for matmul mode.
 *
 * Handles the matmul operand swap (in0 -> srcB, in1 -> srcA) automatically:
 * only emits reconfig register writes when the two input CBs have different
 * data formats (compile-time check). When formats match — the common case —
 * the swap is eliminated entirely by the compiler.
 *
 * PREREQUISITE: compute_kernel_hw_startup(in0_cb, in1_cb, out_cb) called once
 * at kernel start. Standard CB order, no operand swap needed.
 *
 * | Template Param     | Description                                              |
 * |--------------------|----------------------------------------------------------|
 * | in0_cb             | First matmul input CB  (maps to srcB in HW)              |
 * | in1_cb             | Second matmul input CB (maps to srcA in HW)              |
 * | out_cb             | Output CB (only used when PackReconfigure is set)        |
 * | transpose          | Transpose flag for B operand (default: false)            |
 * | pack_reconfig_mode | Whether to also reconfigure packer (default: No)         |
 */
// clang-format on
template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    bool transpose = false,
    matmul_config::ReconfigurePackMode pack_reconfig_mode = matmul_config::ReconfigurePackMode::NoPackReconfigure>
ALWI void matmul_init() {
    // Fix the matmul operand swap: srcA needs in1's format, srcB needs in0's.
    // Compile-time eliminated when both CBs share the same data format.
    if constexpr (detail::needs_format_swap<in0_cb, in1_cb>()) {
        reconfig_data_format(in1_cb, in0_cb);
    }

    if constexpr (pack_reconfig_mode == matmul_config::ReconfigurePackMode::PackReconfigure) {
        pack_reconfig_data_format(out_cb);
    }

    // Wraps existing mm_init_short (unpack AB matmul mode + math MOP setup)
    mm_init_short(in0_cb, in1_cb, transpose);
}

// clang-format off
/**
 * Execute a tile-sized matmul: C += A * B.
 *
 * Wraps the existing matmul_tiles() under the consistent op_block naming.
 * DST must be in acquired state via tile_regs_acquire().
 *
 * | Argument       | Description                                                 |
 * |----------------|-------------------------------------------------------------|
 * | in0_cb_id      | First input CB                                              |
 * | in1_cb_id      | Second input CB                                             |
 * | in0_tile_index | Tile index in first input CB                                |
 * | in1_tile_index | Tile index in second input CB                               |
 * | dst_tile_index | Destination tile index in DST register                      |
 */
// clang-format on
ALWI void matmul_block(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t dst_tile_index) {
    // Wraps existing matmul_tiles()
    matmul_tiles(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, dst_tile_index);
}

// clang-format off
/**
 * Uninitialize matmul mode.
 *
 * Completes the init -> block -> uninit triplet. Currently a no-op, but must
 * always be called after the last matmul_block() for programming model
 * consistency. If matmul ever needs to restore HW state on exit, all call
 * sites will already be correct.
 */
// clang-format on
ALWI void matmul_uninit() {
    // No-op today. Placeholder for future state restoration.
}

}  // namespace compute_kernel_lib
