// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_reduce_custom.h. Same instruction sequence and
// access kinds as production, expressed through the HAL. The helpers in hal_port::reduce_block_max are shared with
// llk_unpack_AB_reduce_custom_runtime.h, whose functions differ only in taking block_ct_dim / respect_trigger at
// runtime.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/scalar.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_unpack_AB_reduce.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::unpacker;

namespace hal_port
{
namespace reduce_block_max
{

// ckernel::semaphore::FPU_SFPU / UNPACK_MATH_DONE
inline constexpr hal::sync::Semaphore FPU_SFPU_SEM         = static_cast<hal::sync::Semaphore>(ckernel::semaphore::FPU_SFPU);
inline constexpr hal::sync::Semaphore UNPACK_MATH_DONE_SEM = static_cast<hal::sync::Semaphore>(ckernel::semaphore::UNPACK_MATH_DONE);

// TT_OP_UNPACR(SrcA, 0b00000001 /*ch0_z += 1*/, 0, 0, 0, 1, 1 /*Set Dvalid*/, RAREFYB_DISABLE, 0, 0, 0, 0, 1):
// one whole operand tile (X end is the tile size), handed to math, Z moved to the next tile.
inline constexpr std::uint32_t unpack_srca_op = reduce_unpack(hal::unpack::Engine::Unpacker0, 1, 0, hal::unpack::SourceHandoff::FlipAndSetDataValid);
// TT_OP_UNPACR(SrcB, 0b00000000, 0, 0, 0, 1, 1 /*Set Dvalid*/, RAREFYB_DISABLE, 0, 0, 0, 0, 1): the scaler face.
inline constexpr std::uint32_t unpack_srcb_op = reduce_unpack(hal::unpack::Engine::Unpacker1, 0, 0, hal::unpack::SourceHandoff::FlipAndSetDataValid);

// ckernel_template(outerloop, 1, unpack_srca_op).program(): the NOP start/end ops are written as production does.
inline void program_mop(const std::uint32_t outerloop)
{
    using namespace hal::mop;
    program(loop(outerloop, TT_OP_NOP, loop(1, unpack_srca_op), TT_OP_NOP, TT_OP_NOP));
}

// THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32 and THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1 (y_dim | z_dim << 16)
inline constexpr const hal::cfg::Field &TileXDim = hal::cfg::Thcon[hal::cfg::Reg5].Tile_x_dim_cntx0;
inline constexpr const hal::cfg::Field &TileYZ   = hal::cfg::Thcon[hal::cfg::Reg0].TileDescriptor.YDim;

// Shared body of the two inits: haloize, X ends, save tile x/y dims to the SR GPRs, then program the reduce dims.
inline void configure(const ckernel::TensorShape tensor_shape)
{
    namespace cfg    = hal::cfg;
    namespace scalar = hal::scalar;
    using hal::gpr;

    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    // REDUCE_ROW requires transpose itself; additionally, within_face_16x16_transpose flag could require transpose;
    // if we have the flag set with REDUCE_ROW, we don't need to do anything
    // cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(1)
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0, 1>();

    // Unpack a single face of a scaler: TTI_SETADCXX(UNP_B, FACE_R_DIM * FACE_C_DIM - 1, 0x0)
    set_unpacker_x_end<p_setadc::UNP_B, FACE_R_DIM>();
    // Unpack all faces of an operand: TT_SETADCXX(UNP_A, tensor_shape.total_tensor_size() - 1, 0x0)
    adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = tensor_shape.total_tensor_size() - 1u}}.for_clients<adc::Client::Unpacker0>());

    constexpr auto tmp0 = gpr<p_gpr_unpack::TMP0>();

    // save the following state that is going to be modified:
    // tile x, y, and z dims for both unpackers
    // CH1 Z stride for both unpackers
    cfg::read<cfg::Access::TensixCfgUnit, TileXDim, cfg::Sec::S0>(gpr<p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0>());
    cfg::read<cfg::Access::TensixCfgUnit, TileYZ, cfg::Sec::S0>(gpr<p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1>());

    scalar::set_low<1 * FACE_C_DIM * FACE_R_DIM>(tmp0);
    scalar::set_high<1 * FACE_C_DIM * FACE_R_DIM>(tmp0);
    cfg::write<cfg::Access::TensixCfgUnit, TileXDim, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(tmp0);
    // y_dim = number of faces (4 for a 32x32 tile, 2 for a 16x32 tiny tile). Hardcoding 4 makes the
    // unpacker stride a full 32x32 tile between block tiles, over-reading the next tile (out-of-bounds
    // / wrong tile) for a genuine 16x32 operand.
    if (tensor_shape.num_faces_r_dim == 1)
    {
        scalar::set_low<2 /* y_dim: 2 faces (16x32 tiny tile) */>(tmp0);
    }
    else
    {
        scalar::set_low<4 /* y_dim: 4 faces (32x32 tile) */>(tmp0);
    }
    scalar::set_high<1 /* z_dim */>(tmp0);
    cfg::write<cfg::Access::TensixCfgUnit, TileYZ, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(tmp0);
}

// Per-call prologue: reset counters, acquire a context, program the addresses, unpack the scaler.
inline __attribute__((always_inline)) void begin(const std::uint32_t address_a, const std::uint32_t address_b)
{
    // reset counters: TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();

    // Program srcA and srcB base addresses
    volatile std::uint32_t tt_reg_ptr *cfg = hal::cfg::detail::state_cfg_bank(); // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Validate and configure addresses
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);

    // Trisc::SEMPOST for context acquire (MMIO)
    hal::sync::semaphore::post<hal::sync::Access::MMIO, UNPACK_SYNC_SEM>();

    // Stall unpacker until pending CFG writes from Trisc have completed
    hal::sync::wait::stall<hal::sync::StallTarget::Unpack, hal::sync::StallCondition::RiscvAccessProcessed>();

    TTI_INSN(unpack_srcb_op);
}

// Per-call epilogue: release the context and switch the unpacker config context.
inline __attribute__((always_inline)) void end()
{
    // T6::SEMGET for context release
    hal::sync::semaphore::get<UNPACK_SYNC_SEM>();

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}

// t6_semaphore_wait_on_zero<STALL_UNPACK>(S)
template <hal::sync::Semaphore S>
inline __attribute__((always_inline)) void wait_while_zero()
{
    hal::sync::wait::semaphore<hal::sync::StallTarget::Unpack, hal::sync::SemaphoreCondition::WhileZero, S>();
}

// Uninit: restore the tile x/y dims saved by configure.
inline void restore()
{
    namespace cfg = hal::cfg;
    using hal::gpr;
    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::Unpacker0Idle | hal::sync::StallCondition::Unpacker1Idle>();
    cfg::write<cfg::Access::TensixCfgUnit, TileXDim, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        gpr<p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0>());
    cfg::write<cfg::Access::TensixCfgUnit, TileYZ, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        gpr<p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1>());
}

} // namespace reduce_block_max
} // namespace hal_port

/**
 * Configures MOP (Macro Operation) for block-based reduce_max_row unpacking operations.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native reduce unpacking LLK MOP configuration.
 * Use the standard _llk_unpack_AB_mop_config_ for general-purpose block reduction operations.
 */
template <std::uint32_t block_ct_dim, bool respect_trigger = false>
inline void _llk_unpack_AB_reduce_block_max_row_mop_config_()
{
    // Constraint on the outerloop and innerloop dim
    static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");
    // Single UNPACR because TTI_SETADCXX for UNP_A is 1023, increment Z counter to point to the next tile, set dvalid each time
    constexpr std::uint32_t outerloop = respect_trigger ? (block_ct_dim / 2) : block_ct_dim;
    hal_port::reduce_block_max::program_mop(outerloop); // innerloop = 1: unpack tile by tile of the input operand into SrcA
}

/**
 * Initializes unpacker configuration for block-based reduce_max_row operations.
 * Sets up tile dimensions and saves unpacker state that will be modified during operation.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile is num_faces faces (4 for a 32x32 tile, 2 for a 16x32 tiny tile)
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * @param tensor_shape Shape of the operand tile (4 faces for 32x32, 2 faces for a 16x32 tiny tile).
 *
 * This function should NOT be used as a substitute for native reduce unpacking LLK initialization.
 * Use the standard _llk_unpack_AB_reduce_init_ for general-purpose reduction operations.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false, bool respect_trigger = false>
inline void _llk_unpack_AB_reduce_block_max_row_init_(const ckernel::TensorShape tensor_shape)
{
    hal_port::reduce_block_max::configure(tensor_shape);

    _llk_unpack_AB_reduce_block_max_row_mop_config_<block_ct_dim, respect_trigger>(); // Unpack operand and scaler
}

/**
 * Performs unpacking for block-based reduce_max_row operation across multiple tiles.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native _llk_unpack_AB_ LLK.
 * Use the standard _llk_unpack_AB_ in a loop for general-purpose block reduction operations.
 */
template <bool respect_trigger = false>
inline void _llk_unpack_AB_reduce_block_max_row_(const std::uint32_t address_a, const std::uint32_t address_b)
{
    namespace rbm = hal_port::reduce_block_max;
    using Mop     = hal::mop::Runner<hal::mop::MopTemplate::Template1>;

    rbm::begin(address_a, address_b);

    if constexpr (respect_trigger)
    {
        // MOP is programmed for half of block_ct_dim; run first half immediately
        Mop::run();
        // Wait for blocked_matmul_and_pack to signal that second-half data is ready
        rbm::wait_while_zero<rbm::FPU_SFPU_SEM>();
        // Run MOP again for the second half (Z counter continues from where first half stopped)
        Mop::run();
    }
    else
    {
        Mop::run();
    }

    rbm::end();
}

/**
 * Uninitializes block-based reduce_max_row unpacker operation.
 * Restores the unpacker state that was saved during initialization.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for native reduce unpacking cleanup.
 * Standard _llk_unpack_AB_reduce_init_ operations typically don't require explicit cleanup.
 */
template <bool respect_trigger = false>
inline void _llk_unpack_AB_reduce_block_max_row_uninit_()
{
    hal_port::reduce_block_max::restore();

    if constexpr (respect_trigger)
    {
        // t6_semaphore_get(semaphore::FPU_SFPU)
        hal::sync::semaphore::get<hal_port::reduce_block_max::FPU_SFPU_SEM>();
    }
}
