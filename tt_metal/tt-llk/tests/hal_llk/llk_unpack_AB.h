// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_unpack_AB.h. Same instruction sequence and access kinds as production,
// expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "hal/mop.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_unpack.h"

using namespace ckernel;
using namespace ckernel::unpacker;

namespace hal_port
{

// One face from L1 into the next SrcA/SrcB bank, advancing channel-0 Z by z_increment, handing the bank to math:
// TT_OP_UNPACR(Src, z_increment, 0, 0, 0, 1, 1, RAREFYB_DISABLE, 0, 0, 0, 0, 1).
constexpr hal::unpack::DataTransfer unpack_face(const hal::unpack::Engine engine, const std::uint8_t z_increment)
{
    return {
        .engine     = engine,
        .increments = {.channel0 = {.z = z_increment}},
        .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
    };
}

inline constexpr std::uint32_t unpack_srca = hal::unpack::get_operation<unpack_face(hal::unpack::Engine::Unpacker0, 1)>();
inline constexpr std::uint32_t unpack_srcb = hal::unpack::get_operation<unpack_face(hal::unpack::Engine::Unpacker1, 1)>();
// Z += 2 moves down a column of faces (transpose allows at most two face columns).
inline constexpr std::uint32_t unpack_srca_transpose = hal::unpack::get_operation<unpack_face(hal::unpack::Engine::Unpacker0, 2)>();

// TT_OP_SETADCZW(UNP_A, 0, 0, 0, 1, 0b0001): SrcA face (channel-0 Z) counter to 1, i.e. the top of the next column of faces.
inline constexpr std::uint32_t setadc_srca_z1 = adc::AddrCounter {.channel0 = {.z = 1}}.for_clients<adc::Client::Unpacker0>().operation();
// TT_OP_SETADCZW(UNP_B, 0, 0, 0, 0, 0b0001): SrcB face counter back to 0.
inline constexpr std::uint32_t setadc_srcb_z0 = adc::AddrCounter {.channel0 = {.z = 0}}.for_clients<adc::Client::Unpacker1>().operation();

} // namespace hal_port

/**
 * @brief Rows per face that unpacker B reads.
 *
 * ROW and SCALAR read B's own face rows: math uses only SrcB row 0, and a B shaped like A keeps both unpacks the
 * same length. NONE and COL read A's face rows, so B needs at least that many.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param tensor_shape_a: Tensor shape of source A
 * @param tensor_shape_b: Tensor shape of source B
 */
template <BroadcastType BType>
inline std::uint32_t _llk_unpack_AB_srcb_face_r_dim_(const ckernel::TensorShape tensor_shape_a, const ckernel::TensorShape tensor_shape_b)
{
    if constexpr (BType == BroadcastType::ROW || BType == BroadcastType::SCALAR)
    {
        return tensor_shape_b.face_r_dim;
    }
    else
    {
        LLK_ASSERT(tensor_shape_b.face_r_dim >= tensor_shape_a.face_r_dim, "SrcB faces must have at least as many rows as SrcA faces for NONE/COL");
        return tensor_shape_a.face_r_dim;
    }
}

/**
 * @brief Configure the MOP (Micro-Operation Program) for unpacking two source operands A and B
 *
 * B faces read per broadcast type:
 * - NONE: one per A face.
 * - COL: the first face of each row of B faces, once per row of A faces.
 * - ROW: B's first row of faces, for every row of A faces.
 * - SCALAR: B's first face, once per tile.
 *
 * @note With no broadcast and no inter-face transpose, a face shorter than MAX_FPU_ROWS rows zeroes the unpacker's
 * own SrcA and SrcB banks before each face, because the FPU reads at least MAX_FPU_ROWS rows per face.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param transpose_of_faces: Whether to transpose faces (reorder faces 0,2,1,3)
 * @param tensor_shape_a: Tensor shape of source A
 * @param tensor_shape_b: Tensor shape of source B
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_mop_config_(const bool transpose_of_faces, const ckernel::TensorShape tensor_shape_a, const ckernel::TensorShape tensor_shape_b)
{
    using namespace hal::mop;

    const std::uint32_t num_faces_r_dim = tensor_shape_a.num_faces_r_dim;
    const std::uint32_t num_faces_c_dim = tensor_shape_a.num_faces_c_dim;
    // TODO: Remove this assert after testing >4 num_faces because there is no reason to limit this for non-broadcast versions
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_mop_config_", tensor_shape_a);
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_mop_config_", tensor_shape_b);

    if (transpose_of_faces)
    {
        LLK_ASSERT(num_faces_r_dim == num_faces_c_dim, "num_faces_r_dim must be equal to num_faces_c_dim when transpose_of_faces is true");
        LLK_ASSERT(
            num_faces_c_dim == 1 || num_faces_c_dim == 2,
            "num_faces_c_dim has to be either 1 or 2 with transpose due to stride limitations in UNPACR instruction");
    }

    const std::uint32_t outerloop = transpose_of_faces ? num_faces_c_dim : num_faces_r_dim;
    const std::uint32_t innerloop = transpose_of_faces ? num_faces_r_dim : num_faces_c_dim;
    const std::uint32_t srca_op   = transpose_of_faces ? hal_port::unpack_srca_transpose : hal_port::unpack_srca;
    // With transpose, move SrcA to the top of the next column of faces.
    const std::uint32_t srca_next_column = transpose_of_faces ? hal_port::setadc_srca_z1 : TT_OP_NOP;

    if constexpr (BType == BroadcastType::COL)
    {
        LLK_ASSERT(num_faces_c_dim >= num_faces_r_dim, "BroadcastType::COL does not support a 32x16 SrcA tile");
        LLK_ASSERT(tensor_shape_b.num_faces_r_dim >= num_faces_r_dim, "BroadcastType::COL needs a SrcB face for every row of SrcA faces");
        // After each row of A faces, move B to its next row of faces.
        // TT_OP_SETADCZW(UNP_B, 0, 0, 0, tensor_shape_b.num_faces_c_dim, 0b0001)
        const std::uint32_t srcb_next_row = hal::counters::adc::AddrCounter {.channel0 = {.z = tensor_shape_b.num_faces_c_dim}}
                                                .for_clients<hal::counters::adc::Client::Unpacker1>()
                                                .operation();
        program(loop(outerloop, hal_port::unpack_srcb, loop(innerloop, srca_op), srcb_next_row, srca_next_column));
    }
    else if constexpr (BType == BroadcastType::ROW)
    {
        LLK_ASSERT(num_faces_c_dim >= num_faces_r_dim, "BroadcastType::ROW does not support a 32x16 SrcA tile");
        LLK_ASSERT(tensor_shape_b.num_faces_c_dim >= num_faces_c_dim, "BroadcastType::ROW needs a SrcB face for every column of SrcA faces");
        // Re-read B's first row of faces for every row of A faces.
        program(loop(outerloop, loop(innerloop, hal_port::unpack_srcb, srca_op), hal_port::setadc_srcb_z0, srca_next_column));
    }
    else if constexpr (BType == BroadcastType::SCALAR)
    {
        LLK_ASSERT(!transpose_of_faces, "SrcA transpose is not supported with scalar broadcast");
        program(loop(1 /*count*/, hal_port::unpack_srcb, loop(tensor_shape_a.total_num_faces(), hal_port::unpack_srca)));
    }
    else // BType == BroadcastType::NONE
    {
        LLK_ASSERT(tensor_shape_b.total_num_faces() >= tensor_shape_a.total_num_faces(), "SrcB needs a face for every SrcA face");
        if (!transpose_of_faces && tensor_shape_a.face_r_dim < MAX_FPU_ROWS)
        {
            // The FPU reads at least MAX_FPU_ROWS rows per face. Zero the unpacker's own bank of both sources
            // before each face, so ELWADD and ELWSUB also see zeros in the rows past the face.
            // HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the clear-source flavor (CLR_SRC_0 / CLR_SRC,
            // WAIT_LIKE_UNPACR) has no HAL descriptor, so the production encoding is kept.
            static constexpr std::uint32_t clear_srca = TT_OP_UNPACR_NOP(
                SrcA,
                0 /*Stream_Id*/,
                0 /*Msg_Clr_Cnt*/,
                0 /*Set_Dvalid*/,
                0 /*Clr_to1_fmt_Ctrl*/,
                p_unpacr_nop::WAIT_LIKE_UNPACR,
                0 /*Bank_Clr_Ctrl*/,
                p_unpacr_nop::CLR_SRC_0,
                p_unpacr_nop::CLR_SRC);
            static constexpr std::uint32_t clear_srcb = TT_OP_UNPACR_NOP(
                SrcB,
                0 /*Stream_Id*/,
                0 /*Msg_Clr_Cnt*/,
                0 /*Set_Dvalid*/,
                0 /*Clr_to1_fmt_Ctrl*/,
                p_unpacr_nop::WAIT_LIKE_UNPACR,
                0 /*Bank_Clr_Ctrl*/,
                p_unpacr_nop::CLR_SRC_0,
                p_unpacr_nop::CLR_SRC);
            constexpr hal::replay::BufferRange replay_range {0 /*start*/, 4 /*count*/};
            hal::replay::record<replay_range>(
                []
                {
                    TTI_INSN(clear_srca);
                    TTI_INSN(hal_port::unpack_srca);
                    TTI_INSN(clear_srcb);
                    TTI_INSN(hal_port::unpack_srcb);
                });
            program(loop(num_faces_r_dim, loop(num_faces_c_dim, hal::replay::get_operation<replay_range>())));
            return;
        }
        program(loop(outerloop, loop(innerloop, srca_op, hal_port::unpack_srcb), srca_next_column));
    }
}

/**
 * @brief Initialize unpacker to unpack two source operands A and B into SrcA and SrcB registers
 *
 * Configures two-operand unpacking with optional broadcast and transpose. A reads full faces; B reads the rows
 * set by @ref _llk_unpack_AB_srcb_face_r_dim_.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param tensor_shape_a: Tensor shape of source A
 * @param tensor_shape_b: Tensor shape of source B, e.g. [1,32] for ROW/SCALAR broadcast
 * @param transpose: Transpose mode for SrcA face order and/or within-face transpose, values = <None/IntraFace/InterFace/Both>
 * @note With no broadcast and no inter-face transpose, a face shorter than MAX_FPU_ROWS rows zeroes the unpacker's
 * own SrcA and SrcB banks before each face (see @ref _llk_unpack_AB_mop_config_).
 * @note B's datum count (x_end) can differ from A's. Each op's init sets the counts it uses;
 * @ref _llk_unpack_AB_uninit_ only turns Auto TTSync off.
 * @ref _llk_unpack_AB_ is the matching execute call.
 * @ref _llk_math_eltwise_binary_init_ is the matching init on the math thread (consumes SrcA/SrcB).
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape_a, const ckernel::TensorShape tensor_shape_b, const ckernel::Transpose transpose)
{
    namespace cfg = hal::cfg;

    // TODO: Remove this assert after testing >4 num_faces because there is no reason to limit this for non-broadcast versions
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_init_", tensor_shape_a);
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_init_", tensor_shape_b);
    const bool within_face_16x16_transpose = transpose == ckernel::Transpose::IntraFace || transpose == ckernel::Transpose::Both;
    const bool transpose_of_faces          = transpose == ckernel::Transpose::InterFace || transpose == ckernel::Transpose::Both;
    // _llk_unpack_AB_ uses config context 0 only, ordered by Auto TTSync (AutoTTSync.md): RISC stores to unpacker
    // config are ordered against the UNPACR/MOP pushes around them, so context 0 is rewritten in place without the
    // two-context semaphore handshake. Scoped to AB unpacking; _llk_unpack_AB_uninit_ turns it off again, and every
    // kernel starts with it off. Free when already on (re-init inside a loop).
    hal_port::reset_config_context();
    hal_port::unpack_sync_mode<hal_port::UnpackSyncMode::AutoTTSync>();

    // transpose within the face
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(within_face_16x16_transpose);

    const std::uint32_t srcb_face_r_dim = _llk_unpack_AB_srcb_face_r_dim_<BType>(tensor_shape_a, tensor_shape_b);
    if (srcb_face_r_dim == tensor_shape_a.face_r_dim)
    {
        hal_port::config_unpacker_x_end<p_setadc::UNP_AB>(tensor_shape_a.face_r_dim);
    }
    else
    {
        hal_port::config_unpacker_x_end<p_setadc::UNP_A>(tensor_shape_a.face_r_dim);
        hal_port::config_unpacker_x_end<p_setadc::UNP_B>(srcb_face_r_dim);
    }

    _llk_unpack_AB_mop_config_<BType>(transpose_of_faces, tensor_shape_a, tensor_shape_b);
}

/**
 * @brief Initialize unpacker for operands A and B that share one tensor shape.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param tensor_shape: Tensor shape of both sources (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
 * @param transpose: Transpose mode for SrcA face order and/or within-face transpose, values = <None/IntraFace/InterFace/Both>
 * @note Calls the two-shape @ref _llk_unpack_AB_init_; its notes apply.
 * @ref _llk_unpack_AB_ is the matching execute call.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const ckernel::Transpose transpose)
{
    _llk_unpack_AB_init_<BType>(tensor_shape, tensor_shape, transpose);
}

/**
 * @brief Initialize unpacker to unpack operands A and B with no transpose.
 *
 * Convenience overload that forwards to the transpose-aware init with @ref ckernel::Transpose::None.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
 * @note Calls the two-shape @ref _llk_unpack_AB_init_; its notes apply.
 * @ref _llk_unpack_AB_ is the matching execute call.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    _llk_unpack_AB_init_<BType>(tensor_shape, ckernel::Transpose::None);
}

/**
 * @brief Initialize unpacker to unpack operands A and B with a boolean transpose flag.
 *
 * Convenience overload taking an integer flag: nonzero selects @ref ckernel::Transpose::Both,
 * zero selects @ref ckernel::Transpose::None.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param tensor_shape: Tensor shape describing tile dimensions (face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
 * @param transpose: Nonzero to enable both inter-face and within-face transpose, zero for none.
 * @note Calls the two-shape @ref _llk_unpack_AB_init_; its notes apply.
 * @ref _llk_unpack_AB_ is the matching execute call.
 */
template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_init_(const ckernel::TensorShape tensor_shape, const std::uint32_t transpose)
{
    _llk_unpack_AB_init_<BType>(tensor_shape, transpose > 0 ? ckernel::Transpose::Both : ckernel::Transpose::None);
}

/**
 * @brief Teardown after AB (two-operand) unpacking: turn the unpack thread's Auto TTSync off again.
 *
 * The SrcA/SrcB unpacker x-start/x-end (datum-count) state is transient and reprogrammed by each
 * operation's init (see tt-llk#1036). Auto TTSync, enabled by @ref _llk_unpack_AB_init_, is sticky per thread.
 *
 * @note Call @ref _llk_unpack_AB_init_ before this function.
 */
inline void _llk_unpack_AB_uninit_()
{
    hal_port::unpack_sync_mode<hal_port::UnpackSyncMode::Handshake>();
}

/**
 * @brief Unpack two tiles from L1 memory into SrcA and SrcB registers
 *
 * Performs the actual unpacking operation by programming base addresses and running
 * the configured MOP. Uses config context 0 only; Auto TTSync orders the base stores against the MOPs.
 *
 * @tparam BType: Broadcast type for source B, values = <NONE/COL/ROW/SCALAR>
 * @param address_a: L1 memory address of source A tile
 * @param address_b: L1 memory address of source B tile
 * @param bcast_row_idx: Row index within source B tile for ROW broadcast
 * @param srcb_format: Source B data format used to calculate ROW broadcast address offset
 * @note Call @ref _llk_unpack_AB_init_ with matching template args before this function, and
 *       @ref _llk_unpack_AB_uninit_ after it to restore modified state.
 * @ref _llk_math_eltwise_binary_ on the math thread consumes the SrcA/SrcB tiles unpacked here.
 */

template <BroadcastType BType = BroadcastType::NONE>
inline void _llk_unpack_AB_(
    const std::uint32_t address_a,
    std::uint32_t address_b,
    [[maybe_unused]] const std::uint32_t bcast_row_idx = 0,
    [[maybe_unused]] const std::uint32_t srcb_format   = 0)
{
    LLK_ASSERT(hal_port::unpack_sync_mode_state == hal_port::UnpackSyncMode::AutoTTSync, "_llk_unpack_AB_ needs Auto TTSync: call _llk_unpack_AB_init_ first");
    volatile std::uint32_t tt_reg_ptr *cfg = hal::cfg::detail::state_cfg_bank(); // get pointer to registers for current state ID

    if constexpr (BType == BroadcastType::ROW)
    {
        if (bcast_row_idx > 0)
        {
            // Row broadcast reads a full 32-element row, which spans two faces:
            //   Row 0: Face0 row 0 (cols 0-15) + Face1 row 0 (cols 16-31)
            //   Row 31: Face2 row 15 (cols 0-15) + Face3 row 15 (cols 16-31)
            //
            // Within each face, rows are stored contiguously.
            const std::uint32_t bytes_per_row_in_face = SCALE_DATUM_SIZE(srcb_format, FACE_WIDTH);
            const std::uint32_t bytes_per_face        = SCALE_DATUM_SIZE(srcb_format, FACE_WIDTH * FACE_HEIGHT);

            std::uint32_t row_offset_bytes;
            if (bcast_row_idx < FACE_HEIGHT)
            {
                // Rows 0-15 are in Face 0/1. Offset to the row within Face 0.
                row_offset_bytes = bcast_row_idx * bytes_per_row_in_face;
            }
            else
            {
                // Rows 16-31 are in Face 2/3. Skip first two faces, then offset to the row within Face 2.
                row_offset_bytes = 2 * bytes_per_face + (bcast_row_idx - FACE_HEIGHT) * bytes_per_row_in_face;
            }

            address_b += row_offset_bytes >> 4;
        }
    }

    // Single config context with Auto TTSync (enabled by _llk_unpack_AB_init_): these base stores stall until the
    // previous tile's MOP expansion has passed the Wait Gate, where its UNPACRs have taken their base addresses
    // (MemoryOrdering.md), and the first UNPACR of this MOP waits until the stores have landed (AutoTTSync.md).
    // This replaces the busy-context semaphore, the STALLWAIT on RISC CFG writes and the context flip.
    _llk_unpack_configure_addresses_(address_a, address_b, cfg, 0 /*context*/);

    // reset counters: Z and W of both unpackers, channels 0 and 1 (one SETADCZW, mask 0b1111), right before the MOP
    hal::counters::adc::set<hal::counters::adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}
                                .for_clients<hal::counters::adc::Client::Unpacker0, hal::counters::adc::Client::Unpacker1>()>();

    // Run MOP
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}
