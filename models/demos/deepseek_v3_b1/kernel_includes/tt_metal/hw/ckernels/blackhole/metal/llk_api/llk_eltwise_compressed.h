// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_compressed.h"

// NOTE: Requires eltwise_binary.h to be included before this header.

namespace compressed {

// ---------------------------------------------------------------------------
// Init + Add tiles with in1 compressed
//
// These replace add_tiles_init / add_tiles for the case where in1 (srcB)
// is a compressed tensor. They bypass constexpr array reads for in1 and
// use explicit addresses / runtime format params instead.
// ---------------------------------------------------------------------------

/**
 * @brief Initialize for eltwise add where in1 is compressed.
 *
 * Unlike add_tiles_init which reads tile shape, data format, and page size
 * from constexpr arrays for both CBs, this only reads from cb_in0 (normal tensor).
 *
 * @param cb_in0 CB index for operand A (normal bf16 tiled tensor)
 */
FORCE_INLINE void add_tiles_init_in1_compressed(uint32_t cb_in0) {
    UNPACK(({
        const ckernel::TensorShape ts = get_operand_tensor_shape(get_operand_id(cb_in0));
        _llk_unpack_AB_init_<BroadcastType::NONE>(ts, 0 /*transpose*/);
    }));
    MATH(({
        const ckernel::TensorShape ts = get_operand_tensor_shape(get_operand_id(cb_in0));
        _llk_math_eltwise_binary_init_<ELWADD, NONE, MATH_FIDELITY>(ts, 0 /*acc_to_dest*/);
    }));
}

/**
 * @brief Add two tiles at explicit L1 addresses.
 *
 * Replaces add_tiles() when in1 is compressed. Both addresses are provided
 * explicitly — no CB page_size lookups.
 *
 * The caller must:
 *   1. Call reconfig_unpack_srca() before this for the correct in1 format
 *   2. Compute both L1 addresses (in >> cb_addr_shift units)
 *
 * @param addr_a L1 address of tile A (in >> cb_addr_shift units)
 * @param addr_b L1 address of compressed tile B (in >> cb_addr_shift units)
 * @param dst_index Destination register index
 */
FORCE_INLINE void add_tiles_in1_compressed(uint32_t cb_in0, uint32_t addr_a, uint32_t addr_b, uint32_t dst_index) {
    UNPACK((_llk_unpack_AB_<BroadcastType::NONE>(addr_a, addr_b)));
    MATH(({
        const ckernel::TensorShape ts = get_operand_tensor_shape(get_operand_id(cb_in0));
        _llk_math_eltwise_binary_<
            ELWADD,
            NONE,
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            MATH_FIDELITY,
            EltwiseBinaryReuseDestType::NONE>(ts, dst_index, true);
    }));
}

}  // namespace compressed
