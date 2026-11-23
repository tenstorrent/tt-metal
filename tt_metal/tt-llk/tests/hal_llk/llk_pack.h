// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_pack.h for PackMode::Default (the mode the eltwise binary test uses).
// Tilize and Untilize paths are static_assert'ed out. Reconfig (`_llk_pack_reconfig_data_format_`) is not reached
// by the test and is not ported.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "hal/mop.h"
#include "hal/pack.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_pack_common.h"

using namespace ckernel;
using namespace ckernel::packer;

namespace hal_port
{

// One packer ADDR_MOD slot S (thread CFG word, SETC16). Every field of the 16-bit word is supplied, as
// addr_mod_pack_t::pack_val() does, so the immediate SETC16 carries the whole word.
template <cfg::Sec S, std::uint32_t y_src_incr, std::uint32_t y_src_clr, std::uint32_t y_src_cr, std::uint32_t z_src_incr, std::uint32_t z_src_clr>
inline void set_addr_mod_pack()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::Src][cfg::Y].Incr, S, y_src_incr>(),
        cfg::set<cfg::AddrMod[cfg::Src][cfg::Y].CR, S, y_src_cr>(),
        cfg::set<cfg::AddrMod[cfg::Src][cfg::Y].Clear, S, y_src_clr>(),
        cfg::set<cfg::AddrMod[cfg::Dest][cfg::Y].Incr, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest][cfg::Y].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest][cfg::Y].Clear, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Src][cfg::Z].Incr, S, z_src_incr>(),
        cfg::set<cfg::AddrMod[cfg::Src][cfg::Z].Clear, S, z_src_clr>(),
        cfg::set<cfg::AddrMod[cfg::Dest][cfg::Z].Incr, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest][cfg::Z].Clear, S, 0>());
}

} // namespace hal_port

/**
 * @brief Configure the packer address-modification (ADDR_MOD) slots for the selected pack mode.
 *
 * Programs ADDR_MOD_0/1/2 with the src Y and Z increment/clear patterns the pack MOP relies
 * on to traverse the destination register and step through faces for the given layout.
 * The dest (channel-1) counters are left alone: output placement comes from L1_Dest_addr and
 * the packer's own write pointer, and channel-1 strides are zero outside fast-untilize.
 *
 * @tparam pack_mode: Packing layout, values = <Default> (HAL port)
 */
template <PackMode pack_mode = PackMode::Default>
inline void _llk_pack_configure_addrmod_()
{
    static_assert(pack_mode == PackMode::Default, "HAL port: only PackMode::Default is ported");

    // ADDR_MOD_0: Y src incr 4
    hal_port::set_addr_mod_pack<hal::cfg::Sec::S0, 4 /*y_src_incr*/, 0 /*y_src_clr*/, 0 /*y_src_cr*/, 0 /*z_src_incr*/, 0 /*z_src_clr*/>();
    // ADDR_MOD_1: Y src clear, Z src clear
    hal_port::set_addr_mod_pack<hal::cfg::Sec::S1, 0 /*y_src_incr*/, 1 /*y_src_clr*/, 0 /*y_src_cr*/, 0 /*z_src_incr*/, 1 /*z_src_clr*/>();
    // ADDR_MOD_2: Y src clear, Z src incr 1
    hal_port::set_addr_mod_pack<hal::cfg::Sec::S2, 0 /*y_src_incr*/, 1 /*y_src_clr*/, 0 /*y_src_cr*/, 1 /*z_src_incr*/, 0 /*z_src_clr*/>();
}

/**
 * @brief Build and program the packer MOP template for the selected pack mode.
 *
 * Programs the ckernel MOP with the PACR instruction sequence that packs one tile worth of data,
 * selecting packer interfaces and ADDR_MODs per the layout.
 *
 * @tparam pack_mode: Packing layout, values = <Default> (HAL port)
 * @tparam zero_output: When true, packer emits zeros instead of dest data.
 * @param face_r_dim: Number of rows per face.
 * @param tile_c_dim: Tile column dimension (datums).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param num_tiles: Number of tiles processed per MOP run.
 * @note @ref _llk_pack_configure_addrmod_ must have programmed the ADDR_MOD slots for the same pack_mode.
 */
template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t face_r_dim                  = FACE_R_DIM,
    [[maybe_unused]] const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces                   = 4,
    const std::uint32_t num_tiles                   = 1)
{
    static_assert(pack_mode == PackMode::Default, "HAL port: only PackMode::Default is ported");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    const std::uint32_t PACK_INTF_SEL = face_r_dim == 1 ? p_pacr::SINGLE_INTF_ACTIVE : (face_r_dim == 2 ? p_pacr::TWO_INTFS_ACTIVE : p_pacr::ALL_INTF_ACTIVE);

    const std::uint32_t MOP_INNER_LOOP = (face_r_dim < 4) ? 1 : face_r_dim >> 2;
    const std::uint32_t MOP_OUTER_LOOP = num_faces * num_tiles;

    hal::pack::DataTransfer transfer {
        .address_modifier = ADDR_MOD_0,
        .interfaces       = static_cast<std::uint8_t>(PACK_INTF_SEL),
        .datum_override   = zero_output ? hal::pack::DatumOverride::Zero : hal::pack::DatumOverride::None,
    };
    const std::uint32_t rows = transfer.get_operation();

    transfer.address_modifier     = ADDR_MOD_2;
    const std::uint32_t last_row  = transfer.get_operation(); // last inner iteration
    transfer.address_modifier     = ADDR_MOD_1;
    transfer.boundary             = hal::pack::TileBoundary::Last;
    const std::uint32_t last_face = transfer.get_operation(); // last outer iteration: close the tile

    using namespace hal::mop;
    // Explicit NOP end ops keep production's MOP_CFG stores (end_op0/end_op1 = NOP) unchanged.
    const auto config = loop(MOP_OUTER_LOOP, loop(MOP_INNER_LOOP, rows), TT_OP_NOP, TT_OP_NOP).patch(last_outer, last_face);
    // The last-inner op is never issued with an outer count of 1 (HAL requires the patch to be skipped then).
    program(MOP_OUTER_LOOP > 1 ? config.patch(last_inner, last_row) : config);
}

namespace llk_pack_internal_bh
{
/**
 * @brief Shared packer-init body behind the @ref _llk_pack_init_ overloads.
 *
 * Runs the common init sequence with each stage individually skippable: program the ADDR_MOD slots,
 * build the MOP template, set the packer strides, and program the packer X (datum) counter. The
 * skip_* template flags let a caller reuse state already established by a prior init or hw-configure.
 *
 * @tparam pack_mode: Packing layout, values = <Default> (HAL port)
 * @tparam zero_output: When true, the packer emits zeros instead of dest data.
 * @tparam skip_addrmod_config: When true, leave the ADDR_MOD slots untouched.
 * @tparam skip_packer_strides: When true, do not re-program the packer strides.
 * @tparam mutex_ADC: When true, serialize the SETADCXX issue against mutex::THREAD2_ADC. Needed only
 *         when another thread borrows the pack thread's ADCs.
 * @param pack_src_format: Source (dest register) data format; only used when programming strides.
 * @param face_r_dim: Number of rows per face.
 * @param tile_c_dim: Tile column dimension (datums).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param num_tiles: Number of tiles processed per MOP run.
 */
template <PackMode pack_mode, bool zero_output, bool skip_addrmod_config, bool skip_packer_strides, bool mutex_ADC>
inline void pack_init_apply(
    const std::uint32_t pack_src_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles)
{
    static_assert(pack_mode == PackMode::Default, "HAL port: only PackMode::Default is ported");

    if constexpr (!skip_addrmod_config)
    {
        _llk_pack_configure_addrmod_<pack_mode>();
    }
    _llk_pack_mop_config_<pack_mode, zero_output>(face_r_dim, tile_c_dim, num_faces, num_tiles);
    if constexpr (!skip_packer_strides)
    {
        hal_port::set_packer_strides<pack_mode>(pack_src_format, tile_c_dim);
    }

    // Program the packer X (datum) counter. Per the "inits own SETADCXX" contract, every init sets its
    // own value; on Blackhole x_start/x_end must stay within a single row (0..FACE_C_DIM-1).
    {
        hal_port::AdcMutexGuard<mutex_ADC> guard;
        hal::counters::adc::set<hal::counters::adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = FACE_C_DIM - 1}}
                                    .for_clients<hal::counters::adc::Client::Packers>()>(); // SETADCXX(PAC, FACE_C_DIM - 1, 0)
    }
}
} // namespace llk_pack_internal_bh

/**
 * @brief One-time hardware configuration of the packer for a given data format and tile geometry.
 *
 * Programs the packer config registers (formats, strides, relu) for the chosen pack mode. Call once
 * before the init/execute sequence, or again when a later op needs a different packer configuration.
 *
 * @note Drains the packer and blocks this RISC until it is idle before any register is rewritten, so it
 *       is safe to call while a pack from the previous op is still in flight.
 *
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam pack_mode: Packing layout, values = <Default> (HAL port)
 * @param pack_src_format: Source (dest register) data format.
 * @param pack_dst_format: Destination (L1) data format.
 * @param tile_size: Size of one output tile in bytes.
 * @param face_r_dim: Number of rows per face.
 * @param tile_c_dim: Tile column dimension (datums).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param partial_face: True if packing a partial (sub-face-row) face.
 * @param relu_config: Packed relu mode and threshold configuration (0 disables relu).
 */
template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void _llk_pack_hw_configure_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim  = FACE_R_DIM,
    const std::uint32_t tile_c_dim  = TILE_C_DIM,
    const std::uint32_t num_faces   = 4,
    const bool partial_face         = false,
    const std::uint32_t relu_config = 0)
{
    static_assert(pack_mode == PackMode::Default, "HAL port: only PackMode::Default is ported");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    LLK_ASSERT_BLOCK(llk_assert_fp32_dest_acc_en = is_fp32_dest_acc_en);
    hal_port::configure_pack<is_fp32_dest_acc_en, pack_mode>(
        pack_src_format, pack_dst_format, tile_size, face_r_dim, tile_c_dim, num_faces, partial_face, relu_config);
}

/**
 * @brief Initialize the packer (addrmod + MOP + strides) for a pack op given a source format.
 *
 * The single Blackhole pack-init entry point: programs ADDR_MODs, the MOP template, (unless skipped)
 * the packer strides, and the packer X (datum) counter. Per the "inits own SETADCXX" contract, every
 * init sets its own X-counter value (FACE_C_DIM - 1 on Blackhole).
 *
 * @tparam pack_mode: Packing layout, values = <Default> (HAL port)
 * @tparam zero_output: When true, packer emits zeros instead of dest data.
 * @tparam skip_addrmod_config: When true, leave ADDR_MOD slots untouched (assume already programmed).
 * @tparam skip_packer_strides: When true, do not re-program the packer strides.
 * @tparam mutex_ADC: When true, serialize the SETADCXX issue against mutex::THREAD2_ADC.
 * @param pack_src_format: Source (dest register) data format. Only consulted when programming strides.
 * @param face_r_dim: Number of rows per face.
 * @param tile_c_dim: Tile column dimension (datums).
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param num_tiles: Number of tiles processed per MOP run.
 * @param skip_bh_tilize_workaround: Only meaningful for PackMode::Tilize, which is not ported.
 * @note Pair with @ref _llk_pack_uninit_ after the matching @ref _llk_pack_ execute calls.
 */
template <
    PackMode pack_mode       = PackMode::Default,
    bool zero_output         = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false,
    bool mutex_ADC           = false>
inline void _llk_pack_init_(
    const std::uint32_t pack_src_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles,
    [[maybe_unused]] const bool skip_bh_tilize_workaround)
{
    static_assert(pack_mode == PackMode::Default, "HAL port: only PackMode::Default is ported");

    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    const DataFormat src_format = static_cast<DataFormat>(pack_src_format);
    if (src_format == DataFormat::Float32)
    {
        LLK_ASSERT(num_tiles <= 4, "Max supported num_tiles for FLOAT32 is 4.");
    }
    else if ((src_format == DataFormat::Float16) || (src_format == DataFormat::Float16_b))
    {
        LLK_ASSERT(num_tiles <= 8, "Max supported num_tiles for FLOAT16 or FLOAT16_B is 8.");
    }

    llk_pack_internal_bh::pack_init_apply<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides, mutex_ADC>(
        pack_src_format, face_r_dim, tile_c_dim, num_faces, num_tiles);
}

/**
 * @brief Tear down the packer after a pack op (no-op on Blackhole).
 *
 * @note Pairs with @ref _llk_pack_init_.
 */
inline void _llk_pack_uninit_()
{
    // No state to restore - Blackhole pack_init sets PAC X counter to FACE_C_DIM - 1 which is the default.
}

/**
 * @brief Pack one tile from the destination register to an L1 address.
 *
 * Selects the source dest tile, programs the L1 destination address, runs the packer MOP, and resets
 * the Z counters afterward.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam pack_mode: Packing layout, values = <Default> (HAL port)
 * @tparam mutex_ADC: When true, serialize the per-tile SETADC issues against mutex::THREAD2_ADC.
 * @param tile_index: Index of the source tile in the destination register.
 * @param address: L1 destination address for the packed tile.
 * @note Call @ref _llk_pack_init_ with matching template/runtime args before this function, and
 *       @ref _llk_pack_uninit_ once all pack calls are complete.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool mutex_ADC = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    static_assert(pack_mode == PackMode::Default, "HAL port: only PackMode::Default is ported");

    hal_port::set_dst_write_addr<mutex_ADC>(tile_index);

    hal_port::program_packer_destination(address);

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    {
        hal_port::AdcMutexGuard<mutex_ADC> guard;
        hal::counters::adc::set<hal::counters::adc::AddrCounter {.channel0 = {.z = 0}, .channel1 = {.z = 0}}
                                    .for_clients<hal::counters::adc::Client::Packers>()>(); // reset z counters: SETADCZW(PAC, mask 0b0101)
    }
}
