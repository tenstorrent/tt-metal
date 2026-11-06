// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared pieces of the Quasar 32-bit Dest write-path kernels, which fill each Dest bank one of two ways:
//   A: TRISC0 unpacks L1 straight into Dest (UNPACR_DEST_TILE_INC).
//   B: TRISC3 streams L1 through Unpacker 2 into SrcS, and the SFPU moves each SrcS slice into Dest.
// Each path can hand the bank on with a drain (STALLWAIT + CLEARDVALID) or with the hardware done bit on its
// last write (A: SetDatValid on the last UNPACR_DEST, B: done on the last SFPSTORE / SFPNOP).

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_trisc_common.h"
#include "llk_defs.h"

namespace dwp
{

// DWP_KERNEL_OP values: what the Dest writer does to each tile on its way into Dest.
constexpr std::uint32_t OP_COPY   = 0;
constexpr std::uint32_t OP_SQUARE = 1;
constexpr std::uint32_t OP_EXP    = 2;
constexpr std::uint32_t OP_CHAIN  = 4;

// Per-bank timestamp record the pack thread writes into buffer_C (32-bit words):
//   [0] bank count N, [1] tile-loop start, [2 .. N+1] one stamp per bank, [N+2] tile-loop end.
constexpr std::uint32_t STAMP_WORD_COUNT = 0;
constexpr std::uint32_t STAMP_WORD_START = 1;
constexpr std::uint32_t STAMP_WORD_BANK0 = 2;
constexpr std::uint32_t STAMP_MAX_BANKS  = 1000; // record must fit one 32-bit buffer_C tile (1024 words)

inline std::uint32_t wall_clock_lo()
{
    return static_cast<std::uint32_t>(ckernel::read_wall_clock());
}

// 32-bit SrcS: a 32x32 tile is 16 slices of 4 rows x 16 datums, each moved by one even- and one odd-column access.
constexpr std::uint32_t SRCS_SLICES_PER_TILE = ckernel::trisc::srcs_dims::slice_count(true);

/**
 * @brief Who issues the UNPACR2 auto-loops that feed path B's srcs_bank_to_dest.
 *
 * Caller: the caller issues one UNPACR2 ahead of several banks (@ref srcs_banks_per_unpack). Unpacker 2 then refills
 * SrcS across the bank hand-off instead of restarting. Safe because nothing on TRISC3 waits for Unpacker 2 mid-run:
 * the drain hand-off (_llk_math_set_dvalid_) waits on the SFPU only, and a full SrcS stalls Unpacker 2, not the
 * thread, once the UNPACR2 has been accepted.
 */
enum class SrcsUnpack
{
    PerTile,
    PerBank,
    Caller,
};

/**
 * @brief Banks one UNPACR2 can cover: the largest power of two up to 8 whose slices fit the 8-bit auto-loop count.
 *
 * A power of two keeps a group inside the L1 blocks (a power-of-two count) so its tiles stay contiguous.
 */
constexpr std::uint32_t srcs_banks_per_unpack(const std::uint32_t tiles_per_bank)
{
    std::uint32_t banks = 8;
    while (banks > 1 && SRCS_SLICES_PER_TILE * tiles_per_bank * banks > 255)
    {
        banks /= 2;
    }
    return banks;
}

// Controlled issue delay, without draining prior engine work.
inline void issue_gap(const std::uint32_t cycles)
{
    if (cycles == 0)
    {
        return;
    }
    const auto start = wall_clock_lo();
    while (static_cast<std::uint32_t>(wall_clock_lo() - start) < cycles)
    {
    }
}
} // namespace dwp

#ifdef LLK_TRISC_UNPACK

#include "ckernel_template.h"
#include "llk_unpack_common.h"

namespace dwp
{

/**
 * @brief Programs a MOP that unpacks one Dest bank: tiles_per_bank UNPACR_DEST_TILE_INCs from the current L1/Dest tile.
 *
 * @tparam HW_BANK_SWITCH: Set SetDatValid on the bank's last UNPACR_DEST so the hardware flips the unpack Dest bank on
 *         its final write beat; the caller must then skip @ref _llk_unpack_dest_dvalid_section_done_ for that bank.
 */
template <bool HW_BANK_SWITCH>
inline void program_u2d_bank_mop(const std::uint32_t buf_desc_id, const std::uint32_t tiles_per_bank)
{
    const std::uint32_t unpack_tile = TT_OP_UNPACR_DEST_TILE_INC(1 /*Dst_Tile_Idx_Inc*/, 1 /*Src_Tile_Idx_Inc*/, buf_desc_id, 0 /*SetDatValid*/);
    ckernel::ckernel_template mop(1, tiles_per_bank, unpack_tile);
    if constexpr (HW_BANK_SWITCH)
    {
        const std::uint32_t unpack_last_tile = TT_OP_UNPACR_DEST_TILE_INC(1, 1, buf_desc_id, 1 /*SetDatValid*/);
        mop.set_last_inner_loop_instr(unpack_last_tile);
        mop.set_last_outer_loop_instr(unpack_last_tile);
    }
    mop.program_bank0_sw_cntl(ckernel::instrn_buffer);
}

/**
 * @brief Unpacks one Dest bank starting at L1 tile l1_tile_idx with the MOP from @ref program_u2d_bank_mop.
 *
 * Dest tile 0 is always the start of the current hardware bank under the dest-dvalid scheme.
 */
template <bool HW_BANK_SWITCH>
inline void unpack_bank_to_dest(const std::uint32_t l1_tile_idx)
{
    TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, l1_tile_idx);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    ckernel::ckernel_template::run_bank0_sw_cntl(ckernel::instrn_buffer);
    if constexpr (!HW_BANK_SWITCH)
    {
        _llk_unpack_dest_dvalid_section_done_<ckernel::DstSync::SyncHalf>();
    }
}

} // namespace dwp

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "cmath_common.h"
#include "llk_bfd_alloc.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_sfpu/ckernel_sfpu_srcs.h"
#include "llk_srcs.h"
#include "llk_unpack_common.h"
#include "tensor_shape.h"

namespace dwp
{

constexpr std::uint32_t SRCS_SLICE_ROWS  = ckernel::trisc::srcs_dims::ydim(true);
constexpr std::uint32_t SRCS_ODD_COLUMNS = ckernel::math::SFP_ROWS;
// Programmed with dest.incr = SRCS_SLICE_ROWS when the copy's odd-column SFPSTORE steps Dest (ADDR_MOD_STEP).
constexpr std::uint32_t SRCS_SLICE_STEP_ADDR_MOD = ADDR_MOD_6;

/**
 * @brief Programs Unpacker 2 to fill 32-bit SrcS slices from an L1 tile buffer, one auto-looped UNPACR2 per
 *        TILES_PER_UNPACK tiles.
 *
 * SrcS is only ever fed by the auto-loop: per-slice UNPACR2 issue hangs once the instruction stream runs dry.
 * PACK1 is left unconfigured; path B consumes SrcS into Dest, never back to L1.
 *
 * @tparam TILES_PER_UNPACK: Tiles one UNPACR2 auto-loop covers. More than 1 relies on the L1 tiles being contiguous,
 *         so the slice increment walks straight from one tile into the next.
 * @return The Unp2_Slice0 buffer descriptor id to pass to @ref srcs_bank_to_dest.
 */
template <std::uint32_t TILES_PER_UNPACK = 1>
inline std::uint8_t init_srcs_unpack(const std::uint32_t l1_addr_16B, const DataFormat l1_format, const DataFormat srcs_format)
{
    static_assert(TILES_PER_UNPACK >= 1 && SRCS_SLICES_PER_TILE * TILES_PER_UNPACK <= 255, "UNPACR2 auto-loop count is 8 bits");
    const ckernel::TensorShape slice_shape =
        ckernel::make_tensor_shape(static_cast<std::uint8_t>(SRCS_SLICE_ROWS), static_cast<std::uint8_t>(ckernel::trisc::srcs_dims::XDIM), 1, 1);
    const std::uint8_t bfd =
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp2_Slice0>(slice_shape, l1_addr_16B, static_cast<std::uint32_t>(l1_format));
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(srcs_format);
    _llk_unpack_srcs_config_<1 /*INSTRN_COUNT*/, SRCS_SLICES_PER_TILE * TILES_PER_UNPACK>();
    return bfd;
}

/**
 * @brief Raw 32-bit copy of one SrcS slice into the four Dest rows at RWC_D, then RWC_D += 4.
 *
 * INT32 mode moves raw bits both ways, so Float32 data is bit-exact too (FP32 mode would flush denormals and
 * canonicalise NaNs). The odd-column load carries the SrcS read done, which frees the slice like SFPNOP rd_done.
 *
 * @tparam DEST_DONE: Set done on the odd-column store, flipping the SFPU Dest bank. Use only on a bank's last slice,
 *         and never together with @ref _llk_math_set_dvalid_ for the same bank.
 * @tparam ADDR_MOD_STEP: Step RWC_D through the odd-column store's address mode instead of a separate INCRWC, saving
 *         one issue slot per slice. Needs @ref init_srcs_to_dest with the same flag.
 */
template <bool DEST_DONE, bool ADDR_MOD_STEP = false>
inline void copy_srcs_slice_to_dest()
{
    constexpr std::uint32_t SRCS            = ckernel::math::SFPU_SRCS_BASE_ADDR;
    constexpr std::uint32_t DEST            = ckernel::math::SFPU_DEST_BASE_ADDR;
    constexpr std::uint32_t LAST_STORE_MODE = ADDR_MOD_STEP ? SRCS_SLICE_STEP_ADDR_MOD : ADDR_MOD_7;
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, SRCS);
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0, DEST);
    TTI_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 1 /*SrcS rd done*/, SRCS + SRCS_ODD_COLUMNS);
    TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::INT32, LAST_STORE_MODE, DEST_DONE ? 1 : 0, DEST + SRCS_ODD_COLUMNS);
    if constexpr (!ADDR_MOD_STEP)
    {
        ckernel::math::_incr_counters_<0, 0, SRCS_SLICE_ROWS, 0>();
    }
}

/**
 * @brief Dest[RWC_D .. +3] = MATH(SrcS slice) through SFPI, then release the slice and RWC_D += 4.
 *
 * @tparam DEST_DONE: Put the Dest done on the trailing SFPNOP (bank's last slice only, see @ref copy_srcs_slice_to_dest).
 */
template <typename MATH, bool DEST_DONE>
inline void apply_srcs_slice_to_dest()
{
    using Layout  = ckernel::sfpu::SrcsLayout<sfpi::DataLayout::F32>;
    using SrcsIn  = ckernel::sfpu::SfpuOperand<ckernel::sfpu::SfpuReg::SrcS, ckernel::sfpu::SfpiFormat<sfpi::DataLayout::F32, sfpi::vFloat>>;
    using DestOut = ckernel::sfpu::SfpuOperand<ckernel::sfpu::SfpuReg::Dest, ckernel::sfpu::SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat>>;
    ckernel::sfpu::calculate_unary_operands<MATH, Layout::ops>(SrcsIn {Layout::in0}, DestOut {0});
    TTI_SFPNOP(0 /*srcs_wr_done*/, 1 /*srcs_rd_done*/, DEST_DONE ? 1 : 0);
    ckernel::math::_incr_counters_<0, 0, SRCS_SLICE_ROWS, 0>();
}

// Identical arithmetic and vector grouping for Dest and SrcS input.
template <unsigned DEPTH>
struct AffineChain
{
    using value_type = sfpi::vFloat;

    sfpi_inline static sfpi::vFloat apply(sfpi::vFloat x)
    {
#pragma GCC unroll 1
        for (unsigned i = 0; i < DEPTH; ++i)
        {
            x = x * 0.5f + 0.125f;
        }
        return x;
    }
};

template <typename MATH>
inline void apply_dest_slice()
{
    using Operand = ckernel::sfpu::SfpuOperand<ckernel::sfpu::SfpuReg::Dest, ckernel::sfpu::SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat>>;
    ckernel::sfpu::calculate_unary_operands<MATH, 2>(Operand {0}, Operand {0});
    ckernel::math::_incr_counters_<0, 0, SRCS_SLICE_ROWS, 0>();
}

/**
 * @brief Fills one Dest bank from tiles_per_bank L1 tiles via Unpacker 2 -> SrcS -> SFPU.
 *
 * @tparam UNPACK: PerTile issues one UNPACR2 per tile. PerBank issues one for the whole bank, so the next tile's slices
 *         are not held behind this tile's SFPU instructions. Caller issues none (see @ref SrcsUnpack). PerBank and
 *         Caller need @ref init_srcs_unpack with TILES_PER_UNPACK = the tiles each UNPACR2 covers.
 * @param slice: Called once per SrcS slice with std::true_type on the bank's last slice when HW_BANK_SWITCH is set
 *        (std::false_type otherwise); it must consume one slice and advance RWC_D by @ref SRCS_SLICE_ROWS.
 */
template <bool HW_BANK_SWITCH, SrcsUnpack UNPACK = SrcsUnpack::PerTile, typename SliceFn>
inline void srcs_bank_to_dest(
    const std::uint8_t bfd, const std::uint32_t l1_tile_idx, const std::uint32_t tiles_per_bank, SliceFn&& slice, const std::uint32_t input_stride = 1)
{
    ckernel::math::_reset_counters_<p_setrwc::SET_D>();
    if constexpr (UNPACK == SrcsUnpack::PerBank)
    {
        _llk_unpack_srcs_<1>(bfd, l1_tile_idx * SRCS_SLICES_PER_TILE);
    }
    for (std::uint32_t t = 0; t < tiles_per_bank; ++t)
    {
        if constexpr (UNPACK == SrcsUnpack::PerTile)
        {
            _llk_unpack_srcs_<1>(bfd, (l1_tile_idx + t * input_stride) * SRCS_SLICES_PER_TILE);
        }
        for (std::uint32_t s = 0; s + 1 < SRCS_SLICES_PER_TILE; ++s)
        {
            slice(std::false_type {});
        }
        if constexpr (HW_BANK_SWITCH)
        {
            if (t + 1 == tiles_per_bank)
            {
                slice(std::true_type {});
            }
            else
            {
                slice(std::false_type {});
            }
        }
        else
        {
            slice(std::false_type {});
        }
    }
    if constexpr (!HW_BANK_SWITCH)
    {
        _llk_math_set_dvalid_<p_cleardvalid::SFPU, ckernel::DstSync::SyncHalf>();
    }
}

/**
 * @brief Path B setup common to every kernel: SrcS unpack, SFPU state, and zero Dest base / RWC_D.
 *
 * The thread's Dest section base and the low bits of RWC_D are added to SrcS addresses too, so both stay 0 here.
 */
template <std::uint32_t TILES_PER_UNPACK = 1, bool ADDR_MOD_STEP = false>
inline std::uint8_t init_srcs_to_dest(const std::uint32_t l1_addr_16B, const DataFormat l1_format, const DataFormat srcs_format)
{
    const std::uint8_t bfd = init_srcs_unpack<TILES_PER_UNPACK>(l1_addr_16B, l1_format, srcs_format);
    _llk_math_eltwise_sfpu_init_();
    if constexpr (ADDR_MOD_STEP)
    {
        // The shared SFPU init only programs ADDR_MOD_7 (no increment); this one rides the copy's last store.
        ckernel::addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = SRCS_SLICE_ROWS},
        }
            .set(SRCS_SLICE_STEP_ADDR_MOD);
    }
    ckernel::math::_set_dst_write_addr_<ckernel::trisc::DstTileShape::Tile32x32>(0);
    return bfd;
}

} // namespace dwp

#endif // LLK_TRISC_ISOLATE_SFPU

#if defined(LLK_TRISC_PACK) || defined(LLK_TRISC_ISOLATE_SFPU)

namespace dwp
{

/**
 * @brief Writes the per-bank wall-clock record (layout above) into L1 from the pack thread.
 *
 * @tparam SYNC_BEFORE_STAMP: Block on tensix_sync() before each bank stamp so it marks the bank's pack completion
 *         rather than when TRISC2 issued it (TRISC2 can run several banks ahead of the packer otherwise).
 */
// Run in INIT before the rendezvous, so metadata reads cannot let producers
// prefill the pipeline before the timestamp window starts.
inline void snapshot_hash(const std::uint32_t l1_byte_addr)
{
    auto* words      = reinterpret_cast<volatile std::uint32_t*>(l1_byte_addr);
    const auto* hash = reinterpret_cast<const volatile std::uint32_t*>(0x0080A000);
    for (unsigned i = 0; i < 11; ++i)
    {
        words[1004 + i] = hash[i];
    }
}

template <bool SYNC_BEFORE_STAMP, unsigned STAMP_EVERY = 1>
class BankStamps
{
public:
    BankStamps(const std::uint32_t l1_byte_addr, const std::uint32_t num_banks) :
        words_(reinterpret_cast<volatile std::uint32_t*>(l1_byte_addr)), num_banks_(num_banks)
    {
        words_[STAMP_WORD_COUNT] = num_banks;
    }

    void start()
    {
        words_[STAMP_WORD_START] = wall_clock_lo();
    }

    void bank_done(const std::uint32_t bank)
    {
        if constexpr (STAMP_EVERY == 0)
        {
            return;
        }
        else if ((bank + 1) % STAMP_EVERY != 0)
        {
            return;
        }
        if constexpr (SYNC_BEFORE_STAMP)
        {
            ckernel::tensix_sync();
        }
        words_[STAMP_WORD_BANK0 + bank] = wall_clock_lo();
    }

    void end()
    {
        ckernel::wait_pack_idle();
        words_[STAMP_WORD_BANK0 + num_banks_] = wall_clock_lo();
    }

private:
    volatile std::uint32_t* words_;
    std::uint32_t num_banks_;
};

} // namespace dwp

#endif // LLK_TRISC_PACK
