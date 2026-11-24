// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_pack_untilize.h: same Tensix instruction sequence and access kinds as
// production, except the listed HAL-GAP (CFGSHIFTMASK).

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h" // not used here: kept for production-copy headers that include llk_pack_untilize.h
#include "hal/cfg.h"
#include "hal/counters/adc.h"
#include "hal/mop.h"
#include "hal/nop.h"
#include "hal/pack.h"
#include "hal/replay.h"
#include "hal/scalar.h"
#include "hal/sync.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_pack.h" // hal_port::set_addr_mod_pack
#include "llk_pack_common.h"

using namespace ckernel;
using namespace ckernel::packer;

/**
 * @brief Configure the ADDR_MOD slots used by the untilize pack MOP.
 *
 * ADDR_MOD_0 keeps y_src on the current Dest face-row (used by every inner-loop PACR); ADDR_MOD_1
 * advances y_src by one row and is used by the row-closing PACR.
 */
inline void _llk_pack_untilize_configure_addrmod_()
{
    // In DST_STRIDED_MODE, y_src tracks the row within each Dest face and W tracks
    // the tile within Dest.
    // ADDR_MOD_0: used by every inner-loop PACR. y_src stays on the current row;
    // W advances via INCADCZW between tiles.
    hal_port::set_addr_mod_pack<hal::cfg::Sec::S0, 0 /*y_src_incr*/, 0 /*y_src_clr*/, 0 /*y_src_cr*/, 0 /*z_src_incr*/, 0 /*z_src_clr*/>();

    // ADDR_MOD_1: used by the row-closing PACR (set_last_inner_loop_instr).
    // y_src.incr=1 advances to the next Dest face-row after packing, folding the
    // explicit INCADCXY end_op into the PACR itself.
    hal_port::set_addr_mod_pack<hal::cfg::Sec::S1, 1 /*y_src_incr*/, 0 /*y_src_clr*/, 0 /*y_src_cr*/, 0 /*z_src_incr*/, 0 /*z_src_clr*/>();
}

/*
block_ct_dim represents the number of input tiles in a block.
dense is used with num_faces == 2 and even block_ct_dim, where two 16x32 (or smaller) tiles are packed in a single 32x32 tile region in dest.
*/
/**
 * @brief Build and program the packer MOP template for an untilize (tilized -> row-major) pack.
 *
 * Programs a MOP that walks face rows in the outer loop and tiles within the block in the inner loop,
 * using DST_STRIDED_MODE so each PACR packs a row from each tile, plus a replay buffer that advances
 * the L1 destination address by the per-row stride.
 *
 * @tparam block_ct_dim: Number of input tiles per block.
 * @tparam narrow_row: True when faces occupy only the first column of the tile (single packer interface).
 * @tparam dense: True to pack two tiles into one 32x32 dest region using all interfaces; requires num_faces == 2 and even block_ct_dim.
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @note @ref _llk_pack_untilize_configure_addrmod_ must have programmed the ADDR_MOD slots.
 */
template <std::uint32_t block_ct_dim, bool narrow_row = false, bool dense = false>
inline void _llk_pack_untilize_mop_config_(const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    static_assert(!dense || (block_ct_dim % 2 == 0), "block_ct_dim must be even when dense");
    static_assert(!dense || (!narrow_row), "narrow_row must be false when dense");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(!dense || (num_faces == 2), "num_faces must be 2 when dense");
    /*
    Outer loop iterates over the rows in the block, while the inner loop iterates
    over each tile in the block.
    When dense, we use all 4 interfaces to pack out a row each from 4 faces (2 tiles) that end up contiguous in L1
    because offsets align well and it improves perf, thus we halve the number of mop inner loops.
    */
    constexpr std::uint32_t MOP_INNER_LOOP = dense ? block_ct_dim / 2 : block_ct_dim;
    const std::uint32_t MOP_OUTER_LOOP     = face_r_dim;

    // For narrow row, the faces are stored in the first column of the tile, therefore requiring only one packer interface.
    const std::uint32_t PACK_INTF_SEL = (dense)                          ? p_pacr::ALL_INTF_ACTIVE
                                        : (narrow_row || num_faces == 1) ? p_pacr::SINGLE_INTF_ACTIVE
                                                                         : p_pacr::TWO_INTFS_ACTIVE;
    /*
    When using DST_STRIDED_MODE, each packer interface has a stride of 16*block_size,
    where block_size is set to be the size of a row within face.
    Each PACR instruction packs 2x16 datums if (num_faces>1), meaning that it would
    pack out one row for each tile in the block.
    In the inner loop, for each tile, the rows that get packed from dest register
    in the first outer loop iteration are:
    tile 0: row 0, row 16
    tile 1: row 64, row 80
    tile block_ct_dim-1: row 64*(block_ct_dim-1), row 64*(block_ct_dim-1)+16
    This processes is repeated for each row of the block in dest.
    */
    using namespace hal::mop;
    namespace adc = hal::counters::adc;
    using hal::pack::DataTransfer;
    using hal::pack::DestAccess;
    using hal::pack::TileBoundary;

    // w cnt points to the next tile: INCADCZW(PAC, 0, 0, 1, 0)
    constexpr std::uint32_t next_tile =
        adc::AddrCounter {.action = adc::Action::Increment, .channel0 = {.w = 1}}.for_clients<adc::Client::Packers>().operation();

    DataTransfer transfer {
        .address_modifier = ADDR_MOD_0,
        .interfaces       = static_cast<std::uint8_t>(PACK_INTF_SEL),
        .dest_access      = DestAccess::Strided,
    };
    const std::uint32_t row = transfer.get_operation();

    /*
    Since there are two inner loop operations, the instruction set by set_last_inner_loop_instr
    will replace the second inner loop operation (in the last iteration, call the PACR instruction
    with the Last bit set to 1 instead of 0 to close the row).
    The W counter CR shadow (W_Cr) is established by the SETADC of W in _llk_pack_untilize_
    before run() is called; the SETADCZW there only initializes Z.
    ADDRCRZW with increment 0 resets W to the stored W_Cr value at the start of each outer loop
    iteration (row), without needing tile_dst_offset baked into this MOP template.
    */
    // W = W_Cr (restore W to start of block): ADDRCRZW(PAC, 0, 0, 0, 0, 0b0010 /*CH0_W*/)
    constexpr std::uint32_t restore_w =
        adc::AddrCounter {.action = adc::Action::AdvanceAndReset, .channel0 = {.w = 0}}.for_clients<adc::Client::Packers>().operation();

    constexpr hal::replay::BufferRange l1_advance {ckernel::packer::replay_buf_offset, 2};
    hal::replay::record<l1_advance>(
        []
        {
            // THCON_SEC0_REG1_L1_Dest_addr_ADDR32 += SCRATCH_SEC[CurrentThread].val
            // Scratch slot loaded in _llk_pack_untilize_init_ holds the per-row L1 stride.
            // Replaces ADDDMAREG + STALLWAIT + WRCFG + NOP — saves ~3 cyc + 1 STALLWAIT per row.
            // Mirrors llk_unpack_tilize.h:285 precedent.
            TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG1_L1_Dest_addr_ADDR32); // HAL-GAP: no HAL API for CFGSHIFTMASK
            hal::nop::thread();
        });

    /*
    Close the row in the block by setting the Last bit to 1 in the last inner loop instruction.
    Use ADDR_MOD_1 so the packer auto-advances y_src by 1 (next row in face) post-PACR.
    */
    transfer.address_modifier        = ADDR_MOD_1;
    transfer.boundary                = TileBoundary::Last;
    const std::uint32_t last_loop_op = transfer.get_operation();

    // After the inner loop finishes, update L1 address (replay). The "advance Dst face-row" is folded into the
    // row-closing PACR's AddrMod (ADDR_MOD_1), so no INCADCXY end_op is needed. The explicit NOP second end op keeps
    // production's MOP_CFG store (end_op1 = NOP) unchanged.
    const auto config = loop(MOP_OUTER_LOOP, restore_w, loop(MOP_INNER_LOOP, next_tile, row), hal::replay::get_operation<l1_advance>(), TT_OP_NOP)
                            .patch(last_outer, last_loop_op);
    // The last-inner op is never issued with an outer count of 1 (HAL requires the patch to be skipped then).
    program(MOP_OUTER_LOOP > 1 ? config.patch(last_inner, last_loop_op) : config);
}

/**
 * @brief Initialize the packer for an untilize pack op.
 *
 * Configures ADDR_MODs and the untilize MOP, programs the Z stride, and stores the per-row L1
 * destination address offset into a scratch config slot so the MOP can advance the L1 address per row.
 *
 * @tparam block_ct_dim: Number of input tiles per block.
 * @tparam full_ct_dim: Total number of input tiles across all blocks (must be divisible by block_ct_dim).
 * @tparam narrow_row: True when packing fewer than TILE_C_DIM datums per row.
 * @tparam row_num_datums: Number of datums per output row when narrow_row is set.
 * @tparam dense: True to pack two tiles into one dest region; requires num_faces == 2 and even block_ct_dim.
 * @param pack_src_format: Source (dest register) data format.
 * @param pack_dst_format: Destination (L1) data format.
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @note On the math thread, @ref _llk_math_eltwise_unary_datacopy_ (A2D) populates the dest register this packer reads.
 * @note Pair with @ref _llk_pack_untilize_uninit_ after the matching @ref _llk_pack_untilize_ execute calls.
 */
template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim    = block_ct_dim,
    bool narrow_row              = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense                   = false>
inline void _llk_pack_untilize_init_(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    static_assert(block_ct_dim <= (dense ? 16 : 8), "block_ct_dim must be <= 8 when not dense, <= 16 when dense");
    static_assert(!dense || (block_ct_dim % 2 == 0), "block_ct_dim must be even when dense");
    static_assert(!dense || (!narrow_row), "narrow_row must be false when dense");
    static_assert(full_ct_dim % block_ct_dim == 0, "full_ct_dim must be divisible by block_ct_dim");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(!dense || (num_faces == 2), "num_faces must be 2 when dense");

    if constexpr (narrow_row)
    {
        // Changed to check against TILE_C_DIM instead of FACE_C_DIM until tt-metal#24095 is investigated.
        static_assert(row_num_datums < TILE_C_DIM, "row_num_datums must be set to less than TILE_C_DIM for narrow_row packing");
    }

    _llk_pack_untilize_configure_addrmod_();

    _llk_pack_untilize_mop_config_<block_ct_dim, narrow_row, dense>(face_r_dim, num_faces);

    // Set CH0 Zstride = 2x16x16 faces, .z_src = {.incr = 1} jumps 2 faces
    std::uint32_t x_stride       = datum_size_in_bytes(pack_src_format);
    std::uint32_t y_stride       = FACE_C_DIM * x_stride;
    const std::uint32_t z_stride = 2 * face_r_dim * y_stride;
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::Packer[0].AddrCtrl[hal::cfg::PackerReg::Reg0].Zstride, hal::cfg::Sec::S0>(z_stride);

    std::uint32_t output_addr_offset;
    if constexpr (narrow_row)
    {
        output_addr_offset = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * row_num_datums);
    }
    else
    {
        output_addr_offset = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * ((num_faces == 1) ? 1 : 2) * FACE_C_DIM);
    }

    // Store 16B aligned row offset into a scratch cfg slot so the MOP replay buf can use
    // CFGSHIFTMASK to do `THCON_SEC0_REG1_L1_Dest_addr += SCRATCH` per row.
    // ScratchIndex=0b11 in the CFGSHIFTMASK selects SCRATCH_SEC[CurrentThread]; pack thread
    // is TRISC2, so this slot is SCRATCH_SEC2.
    constexpr auto row_offset = hal::gpr<p_gpr_pack::OUTPUT_ADDR_OFFSET>();
    hal::scalar::set_low(row_offset, LOWER_HALFWORD(output_addr_offset / 16));
    hal::scalar::set_high(row_offset, UPPER_HALFWORD(output_addr_offset / 16));
    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::ScalarIdle>();
    static_assert(hal::cfg::Scratch::val.addr32(hal::cfg::Sec::S2) == SCRATCH_SEC2_val_ADDR32, "pack thread scratch word");
    hal::cfg::write<
        hal::cfg::Access::TensixCfgUnit,
        hal::cfg::Scratch::val,
        hal::cfg::Sec::S2,
        hal::cfg::GprTransferSize::Bits32,
        hal::cfg::WrcfgCompletion::Deferred>(row_offset);
    hal::nop::thread();

    // Always include setup calls for safety (as recommended by maintainer)
    // Program packer to pack out the correct number of datums per row
    // SETADCXX(PAC, x_end, 0)
    namespace adc                 = hal::counters::adc;
    constexpr std::uint32_t x_end = narrow_row ? row_num_datums - 1 : FACE_C_DIM - 1;
    adc::set<adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = x_end}}.for_clients<adc::Client::Packers>()>();
}

/**
 * @brief Untilize-pack one block of tiles from the destination register to L1.
 *
 * Programs the L1 destination address and the packer Z/W/XY counters (establishing the W_Cr shadow so
 * each MOP row restores W), then runs the MOP once per face group, advancing the Z counter between
 * face groups and resetting counters afterward.
 *
 * @tparam block_ct_dim: Number of input tiles per block.
 * @tparam full_ct_dim: Total number of input tiles across all blocks.
 * @tparam narrow_row: True when packing fewer than TILE_C_DIM datums per row.
 * @tparam tile_dst_ct_offset: Compile-time column-tile offset into the destination register.
 * @tparam dense: True to pack two tiles into one dest region; requires num_faces == 2 and even block_ct_dim.
 * @param address: L1 destination base address for the block.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param tile_dst_rt_offset: Runtime row-tile offset into the destination register.
 * @note Call @ref _llk_pack_untilize_init_ with matching template/runtime args before this function, and
 *       @ref _llk_pack_untilize_uninit_ once all untilize-pack calls are complete.
 */
template <
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim        = block_ct_dim,
    bool narrow_row                  = false,
    std::uint32_t tile_dst_ct_offset = 0,
    bool dense                       = false>
inline void _llk_pack_untilize_(const std::uint32_t address, const std::uint32_t num_faces = 4, const std::uint32_t tile_dst_rt_offset = 0)
{
    static_assert(block_ct_dim <= (dense ? 16 : 8), "block_ct_dim must be <= 8 when not dense, <= 16 when dense");
    static_assert(!dense || (block_ct_dim % 2 == 0), "block_ct_dim must be even when dense");
    static_assert(!dense || (!narrow_row), "narrow_row must be false when dense");
    static_assert(full_ct_dim % block_ct_dim == 0, "full_ct_dim must be divisible by block_ct_dim");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(!dense || (num_faces == 2), "num_faces must be 2 when dense");

    /*
    full_ct_dim represents the number of input tiles.
    For input widths greater than 8 tiles, input is split into blocks of equal sizes,
    each block the size of block_ct_dim. This function is called for each block.
    */
    // program_packer_untilized_destination<block_ct_dim, full_ct_dim, diagonal>(address, pack_dst_format);
    hal_port::program_packer_destination(address);
    const std::uint32_t num_faces_per_rdim_tile = (num_faces > 2) ? 2 : 1;

    const std::uint32_t tile_dst_offset = tile_dst_ct_offset + tile_dst_rt_offset;
    // Set W = (15 + tile_dst_offset) & 0xF, establishing the W_Cr shadow so that ADDRCRZW in the
    // MOP START_OP resets W to this value at the start of each outer loop iteration (row).
    // The first INCADCZW in the inner loop then advances W to tile_dst_offset for the first tile.
    // SETADCZW's Ch0_W field is only 3 bits (0-7), so SETADC is used to carry the full 4-bit W value.
    namespace adc = hal::counters::adc;
    // reset ch0 z counter: SETADCZW(PAC, mask 0b0001). A single counter, so the HAL planner emits SETADC(PAC, CH_0, Z, 0).
    adc::set<adc::AddrCounter {.channel0 = {.z = 0}}.for_clients<adc::Client::Packers>()>();
    // set ch0 w counter, establishing W_Cr: SETADC(PAC, CH_0, SET_W, (15 + tile_dst_offset) & 0xF), the same instruction
    // set_dst_write_addr issues.
    set_dst_write_addr((15 + tile_dst_offset) & 0xF);
    // reset ch0 xy counters: SETADCXY(PAC, mask 0b0011)
    adc::set<adc::AddrCounter {.channel0 = {.x = 0, .y = 0}}.for_clients<adc::Client::Packers>()>();

    // Iterate over top, then over bottom faces in the block (if num_faces > 2)
    for (std::uint32_t face = 0; face < num_faces_per_rdim_tile; face++)
    {
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

        // z cnt increments by 2xface_r_dimxFACE_C_DIM: INCADCZW(PAC, 0, 0, 0, 1)
        adc::set<adc::AddrCounter {.action = adc::Action::Increment, .channel0 = {.z = 1}}.for_clients<adc::Client::Packers>()>();
        // reset ch0_y counters: SETADCXY(PAC, mask 0b0010). A single counter, so the HAL planner emits SETADC(PAC, CH_0, Y, 0).
        adc::set<adc::AddrCounter {.channel0 = {.y = 0}}.for_clients<adc::Client::Packers>()>();
    }

    // reset z counters: SETADCZW(PAC, mask 0b0101)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0}, .channel1 = {.z = 0}}.for_clients<adc::Client::Packers>()>();
    set_dst_write_addr(tile_dst_offset);             // reset w counter
}

/**
 * @brief Restore the packer Z stride after an untilize pack op.
 *
 * Stalls on the pack pipe and reprograms the Z stride to its default (single face) value, undoing the
 * strided-mode stride set in @ref _llk_pack_untilize_init_.
 *
 * @param pack_src_format: Source (dest register) data format used to size the default Z stride.
 * @note Pairs with @ref _llk_pack_untilize_init_.
 */
inline void _llk_pack_untilize_uninit_(const std::uint32_t pack_src_format)
{
    hal::sync::wait::stall<hal::sync::StallTarget::Config, hal::sync::StallCondition::PackerIdle>();
    const std::uint32_t z_stride = SCALE_DATUM_SIZE(pack_src_format, FACE_R_DIM * FACE_C_DIM);
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::Packer[0].AddrCtrl[hal::cfg::PackerReg::Reg0].Zstride, hal::cfg::Sec::S0>(z_stride);
}
