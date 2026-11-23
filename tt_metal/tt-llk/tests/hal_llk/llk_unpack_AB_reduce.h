// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_unpack_AB_reduce.h. Same instruction sequence and access kinds as
// production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_unpack.h"

using namespace ckernel;
using namespace ckernel::unpacker;

namespace hal_port
{

// One reduce UNPACR: TT_OP_UNPACR(engine, address_mode, 0, 0, 0, 1, set_dvalid, RAREFYB_DISABLE, 0, 0, 0, 0, 1),
// i.e. explicit context 0, last (flush) set, Z increments per channel as given.
constexpr std::uint32_t reduce_unpack(
    const hal::unpack::Engine engine, const std::uint8_t ch0_z, const std::uint8_t ch1_z, const hal::unpack::SourceHandoff handoff)
{
    return hal::unpack::get_operation(hal::unpack::DataTransfer {
        .engine     = engine,
        .increments = {.channel0 = {.z = ch0_z}, .channel1 = {.z = ch1_z}},
        .handoff    = handoff,
    });
}

// Clear one source bank of the unpacker that owns it, waiting like an UNPACR:
// TT_OP_UNPACR_NOP(src, 0, 0, 0, 0, WAIT_LIKE_UNPACR, 0, clear_val, CLR_SRC).
// HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the clear-source flavor has no HAL descriptor, so the
// production encoding is kept.
constexpr std::uint32_t reduce_clear_src(const std::uint32_t src, const std::uint32_t clear_val)
{
    return TT_OP_UNPACR_NOP(
        src,
        0 /*Stream_Id*/,
        0 /*Msg_Clr_Cnt*/,
        0 /*Set_Dvalid*/,
        0 /*Clr_to1_fmt_Ctrl*/,
        p_unpacr_nop::WAIT_LIKE_UNPACR,
        0 /*Bank_Clr_Ctrl*/,
        clear_val,
        p_unpacr_nop::CLR_SRC);
}

// p_setadc::UNP_A / UNP_B as an address-counter client.
template <std::uint32_t UNP_SEL>
constexpr hal::counters::adc::AddrCounter reduce_for_unpacker(const hal::counters::adc::AddrCounter counter)
{
    static_assert(UNP_SEL == p_setadc::UNP_A || UNP_SEL == p_setadc::UNP_B, "UNP_SEL must be UNP_A or UNP_B");
    if constexpr (UNP_SEL == p_setadc::UNP_A)
    {
        return counter.for_clients<hal::counters::adc::Client::Unpacker0>();
    }
    else
    {
        return counter.for_clients<hal::counters::adc::Client::Unpacker1>();
    }
}

} // namespace hal_port

/**
 * @brief Whether the reduce unpacks the scaler into SrcA and the data into SrcB.
 *
 * True for SUM/AVG REDUCE_ROW, whose MVMUL takes the scaler from SrcA.
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 */
template <PoolType pool_type, ReduceDim reduce_dim>
constexpr bool _llk_unpack_AB_reduce_swaps_operands_()
{
    return (reduce_dim == ReduceDim::REDUCE_ROW) && (pool_type != PoolType::MAX);
}

/**
 * @brief Whether the swapped path reads one row per scaler face, into a zeroed SrcA.
 *
 * True for a scaler shorter than a face, e.g. [1,32]; the zeros keep the result's other columns zero. A full-height
 * scaler face is read whole, so its rows past the first must be zero, as in a standard reduce scaler tile.
 *
 * @param tensor_shape_scaler: Shape of the scaler operand.
 */
inline bool _llk_unpack_AB_reduce_scaler_row_only_(const ckernel::TensorShape tensor_shape_scaler)
{
    return tensor_shape_scaler.face_r_dim < FACE_R_DIM;
}

/**
 * @brief Whether the swapped path reads a 32x32 scaler in one UNPACR, like the data.
 *
 * @param tensor_shape: Shape of the data operand.
 * @param tensor_shape_scaler: Shape of the scaler operand.
 */
inline bool _llk_unpack_AB_reduce_scaler_whole_tile_(const ckernel::TensorShape tensor_shape, const ckernel::TensorShape tensor_shape_scaler)
{
    return tensor_shape.total_num_faces() == MAX_NUM_FACES && tensor_shape_scaler.total_num_faces() == MAX_NUM_FACES &&
           !_llk_unpack_AB_reduce_scaler_row_only_(tensor_shape_scaler);
}

/**
 * @brief Configures the unpacker MOP for reduction operations.
 *
 * The data is unpacked face by face, or in one UNPACR for a 4-face SUM/AVG REDUCE_ROW. The scaler gives the first
 * row of one of its faces per data face, cycling through its faces when it has fewer than the data. The swapped
 * path can read scaler faces whole (see @ref _llk_unpack_AB_reduce_scaler_row_only_).
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param tensor_shape: Shape of the data operand.
 * @param tensor_shape_scaler: Shape of the scaler operand.
 *
 * @note For faces shorter than 16 rows, the data bank is cleared before each face.
 * @note For REDUCE_SCALAR operations, SrcA is cleared before unpacking because SrcA is clobbered in the Math kernel.
 * @note With a short scaler on the swapped path, the unpacker's own SrcA bank is zeroed before each scaler unpack.
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_mop_config_(const ckernel::TensorShape tensor_shape, const ckernel::TensorShape tensor_shape_scaler)
{
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_reduce_mop_config_", tensor_shape);
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_reduce_mop_config_", tensor_shape_scaler);

    using hal::unpack::Engine;
    using hal::unpack::SourceHandoff;
    using namespace hal::mop;

    constexpr bool is_max        = pool_type == PoolType::MAX;
    constexpr bool is_scalar     = reduce_dim == ReduceDim::REDUCE_SCALAR;
    constexpr bool swap_operands = _llk_unpack_AB_reduce_swaps_operands_<pool_type, reduce_dim>();

    constexpr std::uint32_t data_unpacker  = swap_operands ? Srcs::SrcB : Srcs::SrcA;
    constexpr std::uint32_t scaler_adc     = swap_operands ? p_setadc::UNP_A : p_setadc::UNP_B;
    constexpr std::uint32_t data_clear_val = is_max ? p_unpacr_nop::CLR_SRC_NEGINF : p_unpacr_nop::CLR_SRC_0;

    // TT_OP_UNPACR(Src, 0b0, ..., SetDatValid=1): whole tile / rest of the scaler, handed to math.
    constexpr std::uint32_t unpack_srca_whole = hal_port::reduce_unpack(Engine::Unpacker0, 0, 0, SourceHandoff::FlipAndSetDataValid);
    constexpr std::uint32_t unpack_srcb_whole = hal_port::reduce_unpack(Engine::Unpacker1, 0, 0, SourceHandoff::FlipAndSetDataValid);
    // TT_OP_UNPACR(SrcA, 0b00010001 /*ch0_z+=1, ch1_z+=1*/, ..., SetDatValid=0): first scaler face, bank kept.
    constexpr std::uint32_t unpack_srca_first_face = hal_port::reduce_unpack(Engine::Unpacker0, 1, 1, SourceHandoff::Keep);
    // TT_OP_UNPACR(Src, 0b01 /*ch0_z+=1*/, ..., SetDatValid=1): one face, handed to math.
    constexpr std::uint32_t unpack_srca_face = hal_port::reduce_unpack(Engine::Unpacker0, 1, 0, SourceHandoff::FlipAndSetDataValid);
    constexpr std::uint32_t unpack_srcb_face = hal_port::reduce_unpack(Engine::Unpacker1, 1, 0, SourceHandoff::FlipAndSetDataValid);
    constexpr std::uint32_t clear_srca       = hal_port::reduce_clear_src(SrcA, p_unpacr_nop::CLR_SRC_0);
    constexpr std::uint32_t clear_data       = hal_port::reduce_clear_src(data_unpacker, data_clear_val);

    const std::uint32_t num_faces        = tensor_shape.total_num_faces();
    const std::uint32_t num_scaler_faces = tensor_shape_scaler.total_num_faces();

    if (swap_operands && num_faces == MAX_NUM_FACES)
    {
        // One handoff per tile: the whole data tile, and scaler faces 0 and 1 into the SrcA faces MVMUL reads.
        LLK_ASSERT(tensor_shape_scaler.num_faces_c_dim == tensor_shape.num_faces_c_dim, "The scaler needs a face for every column of data faces");
        const bool clear_scaler = _llk_unpack_AB_reduce_scaler_row_only_(tensor_shape_scaler);
        if (_llk_unpack_AB_reduce_scaler_whole_tile_(tensor_shape, tensor_shape_scaler))
        {
            constexpr hal::replay::BufferRange whole_tile_replay {0 /*start*/, 2 /*count*/};
            hal::replay::record<whole_tile_replay>(
                []
                {
                    TTI_INSN(unpack_srca_whole);
                    TTI_INSN(unpack_srcb_whole);
                });
            program(loop(1, loop(1, hal::replay::get_operation<whole_tile_replay>())));
            return;
        }
        const hal::replay::BufferRange replay_range {0 /*start*/, (clear_scaler ? 1u : 0u) + 3};
        hal::replay::record(
            replay_range,
            [clear_scaler]
            {
                if (clear_scaler)
                {
                    TTI_INSN(clear_srca);
                }
                TTI_INSN(unpack_srca_first_face);
                TTI_INSN(unpack_srca_whole);
                TTI_INSN(unpack_srcb_whole);
            });

        program(loop(1, loop(1, hal::replay::get_operation(replay_range))));
        return;
    }

    // One handoff per data face; the scaler wraps when it has fewer faces.
    const bool wrap_scaler = num_scaler_faces < num_faces;
    LLK_ASSERT(
        !wrap_scaler || tensor_shape_scaler.num_faces_c_dim == tensor_shape.num_faces_c_dim,
        "A scaler with fewer faces than the data needs a face for every column of data faces");
    const std::uint32_t outerloop = wrap_scaler ? num_faces / num_scaler_faces : 1;
    const std::uint32_t innerloop = wrap_scaler ? num_scaler_faces : num_faces;
    const bool clear_data_bank    = is_scalar || tensor_shape.face_r_dim < FACE_R_DIM;
    const bool clear_scaler       = swap_operands && _llk_unpack_AB_reduce_scaler_row_only_(tensor_shape_scaler);

    const hal::replay::BufferRange replay_range {0 /*start*/, (clear_data_bank ? 1u : 0u) + (clear_scaler ? 1u : 0u) + 2};
    hal::replay::record(
        replay_range,
        [clear_data_bank, clear_scaler]
        {
            if (clear_data_bank)
            {
                TTI_INSN(clear_data);
            }
            if (clear_scaler)
            {
                TTI_INSN(clear_srca);
            }
            TTI_INSN(unpack_srca_face);
            TTI_INSN(unpack_srcb_face);
        });

    // After each pass over the scaler's faces, move it back to face 0: TT_OP_SETADCZW(scaler_adc, 0, 0, 0, 0, 0b0001).
    constexpr std::uint32_t scaler_face0 = hal_port::reduce_for_unpacker<scaler_adc>(hal::counters::adc::AddrCounter {.channel0 = {.z = 0}}).operation();
    program(loop(outerloop, loop(innerloop, hal::replay::get_operation(replay_range)), wrap_scaler ? scaler_face0 : TT_OP_NOP));
}

/**
 * @brief Initialize the unpacker for reduce operations
 *
 * Configures haloize (transpose) for row reductions and each unpacker's datum count, then programs the MOP.
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param tensor_shape: Shape of the data operand (operand A).
 * @param tensor_shape_scaler: Shape of the scaler operand (operand B), e.g. [1,32].
 *
 * @note For SUM/AVG REDUCE_ROW, operands are swapped: scaler→SrcA, data→SrcB. Configure unpacker 0 for the scaler
 *       and unpacker 1 for the data before this init, face geometry as well as formats.
 * @note For MAX REDUCE_ROW, original layout is kept: data→SrcA (transposed via haloize), scaler→SrcB.
 * @ref _llk_unpack_AB_reduce_ is the matching execute call.
 * @ref _llk_math_reduce_init_ is the matching init on the math thread (this is the scaler operand unpack pairing).
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_init_(const ckernel::TensorShape tensor_shape, const ckernel::TensorShape tensor_shape_scaler)
{
    // The partial-face clear picks CLR_SRC_NEGINF only for MAX, so MIN would pad SrcA with zero where it needs +inf.
    static_assert(
        pool_type != PoolType::MIN,
        "The FPU reduce has no MIN: the hardware provides GMPOOL (max) and GAPOOL (average) only. "
        "Use the SFPU reduce instead (ckernel_sfpu_reduce.h::calculate_reduce).");

    // Validate tensor shape for tile-dependent operations
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_reduce_init_", tensor_shape);
    LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_unpack_AB_reduce_init_", tensor_shape_scaler);

    constexpr bool swap_operands       = _llk_unpack_AB_reduce_swaps_operands_<pool_type, reduce_dim>();
    constexpr std::uint32_t data_adc   = swap_operands ? p_setadc::UNP_B : p_setadc::UNP_A;
    constexpr std::uint32_t scaler_adc = swap_operands ? p_setadc::UNP_A : p_setadc::UNP_B;

    namespace cfg = hal::cfg;

    // Haloize (transpose) for row reductions; the value is a template constant, so the RMWCIB is an immediate.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0, (reduce_dim == ReduceDim::REDUCE_ROW) ? 1 : 0>();

    if (swap_operands && tensor_shape.total_num_faces() == MAX_NUM_FACES)
    {
        // TT_SETADCXX(data_adc, MAX_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1, 0x0), as an immediate
        hal_port::set_unpacker_x_end<data_adc, MAX_NUM_FACES * FACE_R_DIM>();
    }
    else
    {
        hal_port::config_unpacker_x_end<data_adc>(tensor_shape.face_r_dim);
    }
    // The scaler reads one row per face, unless the swapped path reads it whole.
    if (swap_operands && _llk_unpack_AB_reduce_scaler_whole_tile_(tensor_shape, tensor_shape_scaler))
    {
        // TT_SETADCXX(scaler_adc, MAX_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1, 0x0), as an immediate
        hal_port::set_unpacker_x_end<scaler_adc, MAX_NUM_FACES * FACE_R_DIM>();
    }
    else
    {
        hal_port::config_unpacker_x_end<scaler_adc>(swap_operands && !_llk_unpack_AB_reduce_scaler_row_only_(tensor_shape_scaler) ? FACE_R_DIM : 1);
    }

    _llk_unpack_AB_reduce_mop_config_<pool_type, reduce_dim>(tensor_shape, tensor_shape_scaler);
}

/**
 * @brief Initialize the unpacker for reduce operations with a scaler shaped like the data.
 *
 * Forwards to the two-shape init with the scaler's shape equal to the data's.
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param tensor_shape: Shape of both the data and the scaler.
 * @ref _llk_unpack_AB_reduce_ is the matching execute call.
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_init_(const ckernel::TensorShape tensor_shape)
{
    _llk_unpack_AB_reduce_init_<pool_type, reduce_dim>(tensor_shape, tensor_shape);
}

/**
 * @brief Execute the unpacker for reduction operations
 *
 * Performs the actual unpacking of data for reduction operations by:
 * 1. Resetting address counters
 * 2. Programming source A and B base addresses in hardware registers
 * 3. Synchronizing with Trisc using semaphores
 * 4. Running the configured MOP
 * 5. Switching unpacker configuration context
 *
 * @tparam pool_type: Type of pooling operation, values = <SUM/AVG/MAX>
 * @tparam reduce_dim: Dimension along which to reduce, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @param address_a: Base address for source A data in L1 memory.
 * @param address_b: Base address for source B data in L1 memory.
 *
 * @note Call @ref _llk_unpack_AB_reduce_init_ with matching template args before this function.
 * @note This function manages dual-context switching for pipelined execution.
 * @note Semaphores ensure proper synchronization between Trisc and unpacker.
 */
template <PoolType pool_type, ReduceDim reduce_dim>
inline void _llk_unpack_AB_reduce_(const std::uint32_t address_a, const std::uint32_t address_b)
{
    namespace adc = hal::counters::adc;

    // Reset address counters for both unpackers: TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();

    // Program srcA and srcB base addresses
    // Get pointer to configuration registers for current state ID
    volatile std::uint32_t tt_reg_ptr *cfg = hal::cfg::detail::state_cfg_bank();

    // Wait for free context
    hal_port::wait_for_next_context(2);

    constexpr bool swap_operands = _llk_unpack_AB_reduce_swaps_operands_<pool_type, reduce_dim>();

    // SUM/AVG REDUCE_ROW swaps operands (scaler→SrcA, data→SrcB), no transpose needed.
    // MAX REDUCE_ROW keeps original layout (data→SrcA transposed, scaler→SrcB) — GMPOOL only reads SrcA.
    const std::uint32_t addr_unp_a = swap_operands ? address_b : address_a;
    const std::uint32_t addr_unp_b = swap_operands ? address_a : address_b;
    _llk_unpack_configure_addresses_(addr_unp_a, addr_unp_b, cfg);

    // Trisc::SEMPOST for context acquire (MMIO)
    hal::sync::semaphore::post<hal::sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();

    // Stall unpacker until pending CFG writes from Trisc have completed
    hal::sync::wait::stall<hal::sync::StallTarget::Unpack, hal::sync::StallCondition::RiscvAccessProcessed>();

    // Execute the configured MOP
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    // T6::SEMGET for context release
    hal::sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();

    // Switch unpacker config context
    hal_port::switch_config_context(unp_cfg_context);
}
