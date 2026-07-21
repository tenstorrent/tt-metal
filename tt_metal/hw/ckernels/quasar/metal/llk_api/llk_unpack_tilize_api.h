// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_unpack_common_api.h"
#include "llk_unpack_tilize.h"
#include "llk_unpack_reduce_col_tilizeA_strided.h"
#include "api/dataflow/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK TILIZE
 *************************************************************************/

/**
 * @brief Initializes the unpacker for tilize operations on Quasar.
 *
 * Configures UNP_A stride registers and programs the MOP for tilizing
 * block_ct_dim tiles from row-major L1 data into face format in SrcA.
 *
 * @param operand       The input dataflow buffer identifier.
 * @param full_ct_dim   Number of tiles in a full row of the input tensor.
 * @param block_ct_dim  Number of tiles per MOP invocation (defaults to 1).
 */
inline void llk_unpack_tilize_init(
    const std::uint32_t operand, const std::uint32_t full_ct_dim, const std::uint32_t block_ct_dim = 1) {
    const std::uint32_t operand_id = get_operand_id(operand);

    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    _llk_unpack_tilize_init_<p_unpacr::UNP_A, DST_ACCUM_MODE>(operand_id, full_ct_dim, block_ct_dim, tensor_shape);
}

/**
 * @brief Tilizes a block of tiles from L1 row-major layout into SrcA.
 *
 * Computes the L1 face index from the DFB read position and the input
 * tile index, then runs the MOP configured by llk_unpack_tilize_init.
 *
 * @param operand          The input dataflow buffer identifier.
 * @param block_c_tiles    Number of tiles in one block row (must match BLOCK_CT_DIM from init).
 * @param input_tile_index Starting tile index (encodes row offset via block_c_tiles stride).
 */
inline void llk_unpack_tilize_block(
    const std::uint32_t operand, const std::uint32_t block_c_tiles, const std::uint32_t input_tile_index = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);

    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    const std::uint32_t faces_per_entry = tensor_shape.num_faces_r_dim * tensor_shape.face_r_dim;

    const LocalDFBInterface& local_dfb = g_dfb_interface[operand_id];
    const std::uint32_t rd_entry_idx = local_dfb.tc_slots[local_dfb.tc_idx].rd_entry_idx;

    // TODO (SK) #42757: Remove ct_dim loop when block_ct_dim unpacking optimization implemented.
    // BLOCK_CT_DIM is currently hardcoded to 1 in tilize_init (see compute/tilize.h), so the MOP
    // emits one SrcA dvalid per invocation. Loop to match the per-tile math consumption same
    // structural pattern as BH/WH llk_unpack_tilize_block
    const std::uint32_t l1_base_idx = (rd_entry_idx + input_tile_index) * faces_per_entry;
    for (std::uint32_t t = 0; t < block_c_tiles; t++) {
        _llk_unpack_tilize_<p_unpacr::UNP_A>(l1_base_idx + t);
    }
}

/**
 * @brief No-op on Quasar — tilize teardown is not required.
 */
inline void llk_unpack_tilize_uninit([[maybe_unused]] const std::uint32_t operand) {}

/*************************************************************************
 * LLK UNPACK TILIZE SRC A, UNPACK SRC B
 *************************************************************************/

/**
 * @brief Initialize the unpacker for the combined tilize-A / unpack-B reduce operation.
 *
 * This function is only compatible with the math reduce kernel. It configures both UNP_A (tilize
 * path) and UNP_B (scalar path) so that each subsequent llk_unpack_tilizeA_B call produces one
 * tilized srcA tile alongside the reloaded srcB scalar tile required by the reduce math op.
 *
 * On Quasar, operand A's buffer descriptor is reprogrammed to y_dim=1, z_dim=1
 * required by the UNPACR_STRIDE tilize sequence, overriding the configuration
 * set by llk_unpack_hw_configure.
 *
 * @tparam neginf_srcA      No effect on Quasar; accepted for API compatibility with WH/BH.
 * @tparam reload_srcB      Must be true on Quasar (asserted true, srcB is reloaded every iteration for reduce);
 * accepted for API compatibility with WH/BH.
 * @tparam zero_srcA        No effect on Quasar (asserted false); accepted for API compatibility with WH/BH.
 * @tparam zero_srcA_reduce No effect on Quasar; accepted for API compatibility with WH/BH.
 * @param  operandA         Input A dataflow buffer identifier.
 * @param  operandB         Input B (scaler) dataflow buffer identifier.
 * @param  ct_dim           Number of column tiles in the tilize block.
 */
template <
    bool neginf_srcA [[maybe_unused]] = false,
    std::uint32_t reload_srcB [[maybe_unused]] = false,
    bool zero_srcA [[maybe_unused]] = false,
    bool zero_srcA_reduce [[maybe_unused]] = false>
inline void llk_unpack_tilizeA_B_init(
    const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t ct_dim) {
    static_assert(!zero_srcA, "zero_srcA = true does not trigger any functionality on Quasar.");
    static_assert(
        reload_srcB,
        "reload_srcB has to be true for tilizeA_B_block on Quasar, due to the compatibility with the math reduce "
        "kernel.");

    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    const ckernel::TensorShape tensor_shape_A = get_operand_tensor_shape(operandA_id);

    // UNPACR_STRIDE used in unpack_tilize_operands_reduce requires the following buffer descriptor configuration:
    // Overwrite the buffer descriptor configuration from llk_unpack_hw_configure for operandA.
    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B = get_local_dfb_interface(operandA_id).tc_slots[0].base_addr;
    bd_val.f.format = static_cast<std::uint8_t>(unpack_src_format[operandA_id]);
    bd_val.f.x_dim = ckernel::trisc::FACE_C_DIM;
    bd_val.f.y_dim = 1;
    bd_val.f.z_dim = 1;
    ckernel::trisc::_configure_buf_desc_table_(operandA_id, bd_val);

    _llk_unpack_reduce_col_tilizeA_strided_init_(operandA_id, operandB_id, ct_dim, tensor_shape_A);
}

/**
 * @brief Tilize one tile into srcA and unpack scaler tile into srcB for the math reduce column kernel.
 *
 * This function is only compatible with the math reduce column kernel. It tilizes a single tile
 * from operand A's row-major L1 data into SrcA while simultaneously unpacking the scalar tile
 * from operand B into SrcB. The resulting srcA/srcB pair is consumed by a single reduce column math
 * iteration.
 *
 * @tparam neginf_srcA      No effect on Quasar; accepted for API compatibility with WH/BH.
 * @tparam reload_srcB      Must be true on Quasar (asserted true, srcB is reloaded every iteration for reduce);
 * accepted for API compatibility with WH/BH.
 * @tparam zero_srcA        No effect on Quasar (asserted false); accepted for API compatibility with WH/BH.
 * @tparam zero_srcA_reduce No effect on Quasar; accepted for API compatibility with WH/BH.
 * @param  operandA     Input A dataflow buffer identifier.
 * @param  operandB     Input B (scaler) dataflow buffer identifier.
 * @param  tile_index_a Column tile index within operand A.
 * @param  tile_index_b Tile index within operand B.
 * @param  block_ct_dim Number of column tiles in the tilize block.
 */
template <
    bool neginf_srcA [[maybe_unused]] = false,
    std::uint32_t reload_srcB [[maybe_unused]] = false,
    bool zero_srcA [[maybe_unused]] = false,
    bool zero_srcA_reduce [[maybe_unused]] = false>
inline void llk_unpack_tilizeA_B(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    [[maybe_unused]] const std::uint32_t block_ct_dim) {
    static_assert(!zero_srcA, "zero_srcA = true does not trigger any functionality on Quasar.");
    static_assert(
        reload_srcB,
        "reload_srcB has to be true for tilizeA_B on Quasar, due to the compatibility with the math reduce kernel.");

    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    const ckernel::TensorShape tensor_shape_A = get_operand_tensor_shape(operandA_id);

    const LocalDFBInterface& local_dfb_interface_a = get_local_dfb_interface(operandA_id);
    const LocalDFBInterface& local_dfb_interface_b = get_local_dfb_interface(operandB_id);

    const std::uint32_t rd_entry_idx_a = local_dfb_interface_a.tc_slots[local_dfb_interface_a.tc_idx].rd_entry_idx;
    // Compute how many l1_index units fit in one DFB entry.
    // _llk_unpack_reduce_col_tilizeA_strided_ internally scales
    // l1_index by num_faces_c_dim, so one l1_index unit = num_faces_c_dim face-rows in L1.
    const std::uint32_t entry_size_16B = local_dfb_interface_a.entry_size;                           // DFB entry size in 16B
    const std::uint32_t face_row_16B =                                                               // Buffer Descriptor granularity in 16B
        SCALE_DATUM_SIZE(unpack_src_format[operandA_id], ckernel::trisc::FACE_C_DIM) >> 4;
    const std::uint32_t l1_index_per_entry =                                                         // l1_index steps per entry
        entry_size_16B / (face_row_16B * tensor_shape_A.num_faces_c_dim);
    const std::uint32_t l1_index_a = rd_entry_idx_a * l1_index_per_entry + tile_index_a;

    const std::uint32_t l1_index_b =
        local_dfb_interface_b.tc_slots[local_dfb_interface_b.tc_idx].rd_entry_idx + tile_index_b;

    WAYPOINT("UPTW");

    _llk_unpack_reduce_col_tilizeA_strided_(tensor_shape_A, l1_index_a, l1_index_b);

    WAYPOINT("UPTD");
}

/**
 * @brief Tilize a block of srcA column tiles and unpack srcB for the math reduce column kernel.
 *
 * This function is only compatible with the math reduce column kernel. It iterates over block_c_tiles_a
 * column tiles, calling llk_unpack_tilizeA_B for each one. Each iteration produces a tilized srcA
 * tile paired with the reloaded srcB scalar tile consumed by a reduce column math step.
 *
 * @tparam neginf_srcA      No effect on Quasar; accepted for API compatibility with WH/BH.
 * @tparam reload_srcB      Must be true on Quasar (asserted true, srcB is reloaded every iteration for reduce);
 * accepted for API compatibility with WH/BH.
 * @tparam zero_srcA        No effect on Quasar (asserted false); accepted for API compatibility with WH/BH.
 * @tparam zero_srcA_reduce No effect on Quasar; accepted for API compatibility with WH/BH.
 * @param  operandA        Input A dataflow buffer identifier.
 * @param  operandB        Input B (scaler) dataflow buffer identifier.
 * @param  block_c_tiles_a Number of column tiles in operand A's block.
 * @param  tile_idx_b      Tile index within operand B.
 */
template <
    bool neginf_srcA [[maybe_unused]] = false,
    std::uint32_t reload_srcB [[maybe_unused]] = false,
    bool zero_srcA [[maybe_unused]] = false,
    bool zero_srcA_reduce [[maybe_unused]] = false>
inline void llk_unpack_tilizeA_B_block(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t block_c_tiles_a,
    const std::uint32_t tile_idx_b) {
    static_assert(!zero_srcA, "zero_srcA = true does not trigger any functionality on Quasar.");
    static_assert(
        reload_srcB,
        "reload_srcB has to be true for tilizeA_B_block on Quasar, due to the compatibility with math reduce kernel.");

    for (std::uint32_t tile_idx_a = 0; tile_idx_a < block_c_tiles_a; tile_idx_a++) {
        llk_unpack_tilizeA_B<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
            operandA, operandB, tile_idx_a, tile_idx_b, block_c_tiles_a);
    }
}

/**
 * Tear down the combined tilize-A / unpack-B configuration so a subsequent operation can reprogram
 * the unpacker. -> No-op for Quasar.
 *
 * @param operand Input circular buffer / operand index.
 */
inline void llk_unpack_tilizeA_B_uninit([[maybe_unused]] const std::uint32_t operand) {}
