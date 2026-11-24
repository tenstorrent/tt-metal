// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_custom_mm.h. Same instruction sequence and access
// kinds as production, expressed through the HAL, except the HAL-GAPs marked below.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "experimental/llk_custom_mm_common.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/nop.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// The custom_mm UNPACR encodings and the counter reset, shared with llk_unpack_AB_compressed_custom_mm.h; the guard
// keeps a translation unit that includes both to one definition.
#ifndef HAL_PORT_CUSTOM_MM_UNPACK_OPS_DEFINED_
#define HAL_PORT_CUSTOM_MM_UNPACK_OPS_DEFINED_

namespace hal_port
{

// TT_OP_UNPACR_COMMON(SrcA, 0b00000000, 1): a full weight tile into SrcA, no counter increments, set dvalid.
inline constexpr std::uint32_t custom_mm_unpack_a = hal::unpack::get_operation(hal::unpack::DataTransfer {
    .engine  = hal::unpack::Engine::Unpacker0,
    .context = hal::unpack::ContextSelection::explicit_context(0, 0),
    .handoff = hal::unpack::SourceHandoff::FlipAndSetDataValid,
});
static_assert(custom_mm_unpack_a == TT_OP_UNPACR_COMMON(SrcA, 0b00000000, 1), "UNPACR SrcA encoding");

// TT_OP_UNPACR_COMMON(SrcB, 0b00010001, 0): first SrcB face, CH0 Z += 1 and CH1 Z += 1, keep the bank.
inline constexpr std::uint32_t custom_mm_unpack_b_face0 = hal::unpack::get_operation(hal::unpack::DataTransfer {
    .engine     = hal::unpack::Engine::Unpacker1,
    .increments = {.channel0 = {.z = 1}, .channel1 = {.z = 1}},
    .context    = hal::unpack::ContextSelection::explicit_context(0, 0),
    .handoff    = hal::unpack::SourceHandoff::Keep,
});
static_assert(custom_mm_unpack_b_face0 == TT_OP_UNPACR_COMMON(SrcB, 0b00010001, 0), "UNPACR SrcB face 0 encoding");

// TT_OP_UNPACR_COMMON(SrcB, 0b00110100, 1): second SrcB face, CH0 Y += 1 and CH1 Z += 3 (wraps back to row 0), set
// dvalid.
inline constexpr std::uint32_t custom_mm_unpack_b_face1 = hal::unpack::get_operation(hal::unpack::DataTransfer {
    .engine     = hal::unpack::Engine::Unpacker1,
    .increments = {.channel0 = {.y = 1}, .channel1 = {.z = 3}},
    .context    = hal::unpack::ContextSelection::explicit_context(0, 0),
    .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
});
static_assert(custom_mm_unpack_b_face1 == TT_OP_UNPACR_COMMON(SrcB, 0b00110100, 1), "UNPACR SrcB face 1 encoding");

// TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111) then TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010): Z and W of both channels, then
// Y of both channels, for both unpackers.
inline __attribute__((always_inline)) void custom_mm_reset_unpack_counters()
{
    namespace adc = hal::counters::adc;
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
    adc::set<adc::AddrCounter {.channel0 = {.y = 0}, .channel1 = {.y = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
}

// TTI_STALLWAIT(STALL_UNPACK, UNPACK), TTI_UNPACR_NOP(SrcB, ..., CLR_SRC), TTI_STALLWAIT(STALL_UNPACK, UNPACK1): zero
// both SrcB banks once both are free. It must run alone: a both-bank SrcB clear drops SrcA writes that unpacker 0
// makes while it runs.
inline __attribute__((always_inline)) void custom_mm_clear_srcb_both_banks()
{
    namespace sync = hal::sync;
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();
    // HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the clear-source flavor (CLR_SRC) has no HAL
    // descriptor, so the production encoding is kept.
    TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::Unpacker1Idle>();
}

} // namespace hal_port
#endif // HAL_PORT_CUSTOM_MM_UNPACK_OPS_DEFINED_

// CUSTOM_MM
// Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
// in0 tile shape: [{1, 2, 4, 8}, 32]
// in1 tile shape: [32, 32] or [32, 16] ([16, 32] with transpose), giving [{1, 2, 4, 8}, 32] or [{1, 2, 4, 8}, 16] output tiles
// rt_dim: 1
// ct_dim: any integer from 1 to 34 (DEST capacity also bounds it, see llk_custom_mm_common.h)
// kt_dim: any integer of 1 or more
// fidelity: LoFi only
// throttle: not supported

inline void _llk_unpack_AB_custom_mm_iter_insns(const bool post1)
{
    if (post1)
    {
        // This nop is not actually used, it just ensures both tunings are of the same length when recording
        // When playing back the replay it is omitted
        hal::nop::thread();
    }

    // Unpack SrcA (in1, full tile, uses CFGSHIFTMASK to manipulate L1 addr and keeps SrcA addr fixed at 0)
    TTI_INSN(hal_port::custom_mm_unpack_a); // Also set dvalid

    // Unpack SrcB (in0, one instruction per face, uses counters to manipulate addresses for both L1 and SrcB)
    // Counters of interest are CH0 Y and Z (both with a stride of a single face in L1
    // (datum_size * 16 * face_r_dim) and CH1 Y with a stride of a face in SrcB (16 rows)
    // Each unpack instruction moves to the next face in L1, split among the Y and Z counters.
    // The L1 face index is (Y & 0xff) + (Z & 0xff), so one base address covers CUSTOM_MM_MAX_KT_PER_WALK k-tiles.
    // For SrcB first unpack has to land at index 0 and second one at index 16,
    // increment from first to second is straightforward, but moving back to 0 after second requires us to exploit
    // counter overflow as CH1 counters only use low 6 bits and wrap at the number of rows in a Src reg (64)
    // thus an increment of 48 rows or 3 CH1 Y increments wraps us back to 0
    // (which is conveniently the max we can increment in a single instruction)

    TTI_INSN(hal_port::custom_mm_unpack_b_face0);
    if (!post1)
    {
        // Unpack second face of SrcB before CFGSHIFTMASK in post0 tuning
        TTI_INSN(hal_port::custom_mm_unpack_b_face1); // Also set dvalid
    }

    // Increment UnpA L1 address
    // HAL-GAP: CFGSHIFTMASK (Base_address += SCRATCH_SEC0_val) has no HAL API; production encoding kept.
    TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
    if (post1)
    {
        // Unpack second face of SrcB after CFGSHIFTMASK in post1 tuning
        TTI_INSN(hal_port::custom_mm_unpack_b_face1); // Also set dvalid;
    }
    else
    {
        // This nop is required in post0 as CFGSHIFTMASK is a 2 cycle instruction
        // And first instruction in next iteration is unpack into SrcA which uses this value
        hal::nop::thread();
    }
}

// TT_MOP has a 7-bit loop count.
constexpr std::uint32_t CUSTOM_MM_MAX_MOP_ITERATIONS = 128;

inline void _llk_unpack_AB_custom_mm_mop_config_(const std::uint32_t ct_dim, const bool post1)
{
    LLK_ASSERT(ct_dim >= 1 && ct_dim <= CUSTOM_MM_MAX_CT_DIM, "custom_mm: ct_dim must be 1 to 34");

    hal::replay::record<hal::replay::BufferRange {0 /*start*/, 32 /*count*/}>(
        [post1]
        {
            // Full unpack (both SrcA and SrcB)
            _llk_unpack_AB_custom_mm_iter_insns(post1);

            // Loop 8 times to fill up the replay buffer
            for (std::uint32_t i = 0; i < 8; i++)
            {
                // Reuse unpack (unpacks only SrcA, SrcB is reused across width dim)
                TTI_INSN(hal_port::custom_mm_unpack_a); // Also set dvalid
                // HAL-GAP: CFGSHIFTMASK has no HAL API; production encoding kept.
                TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
                hal::nop::thread();
            }

            // Reuse unpack (unpacks only SrcA, SrcB is reused across width dim)
            TTI_INSN(hal_port::custom_mm_unpack_a); // Also set dvalid
            // This last iteration uses inner_increment instead of block_increment
            // HAL-GAP: CFGSHIFTMASK has no HAL API; production encoding kept.
            TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 1, THCON_SEC0_REG3_Base_address_ADDR32);
            hal::nop::thread();
        });

    // Each mop iteration covers one kt_dim loop: the ct_dim output tiles of one k-tile, all sharing its SrcB tile.
    // To usefully issue up to 128 mop iterations we're limited to only using 0s in zmask
    // (not using SKIP_A/B instructions) since iterations beyond 32 always use 0s for zmask
    //
    // Replay buffer layout (always 32 instructions):
    // post0: [0-4]: full unpack, [5-28]: 8 reuse blocks (3 insns each), [29-31]: final reuse w/ inner_increment
    // post1: [0]: NOP (skipped), [1-4]: full unpack, [5-28]: 8 reuses, [29-31]: final reuse w/ inner_increment
    // Where:
    //   - full unpack = SrcA + SrcB + address increment (new k-tile)
    //   - reuse = SrcA only + address increment (same k-tile, next c-tile)
    //   - final reuse = uses inner_increment to jump to next k-row instead of block_increment
    //
    // Sequence per mop iteration: [full, reuse x (ct_dim - 2), final reuse], or [full] for ct_dim == 1
    // (block_increment == inner_increment then, so no final reuse is needed).
    //
    // The ct_dim - 1 reuses split over the template slots, each slot a replay range:
    //   the first slot starts at the buffer beginning (offset by post1) and holds the full unpack plus some reuses,
    //   the last slot counts backwards from the buffer end so it always ends in the final reuse,
    //   any slots between replay reuse blocks from the start of the reuse range, since all reuses but the last
    //   are the same instructions.
    // Every slot the template issues must replay something, so the configuration follows the reuse count:
    //   ct_dim == 1:       A0 = full                                       (no reuse for a B slot)
    //   ct_dim 2 to 3:     A0 = full + a reuses, B = d reuses              (too few reuses for A1..A3)
    //   ct_dim 4 to 34:    A0 = full + a reuses, A1 = b, A2 = c, A3 = d    (a, b, c <= 8, d <= 9)
    // The reuses split evenly; the remainder goes to A3, A2 and A1 before A0, which already holds the full unpack,
    // so A1..A3 get a reuse each from ct_dim 4 on.
    const std::uint32_t reuses          = ct_dim - 1;
    const auto replay_insn              = [](const std::uint32_t start, const std::uint32_t count)
    { return hal::replay::get_operation(hal::replay::BufferRange {start, count}); };
    const auto full_and                 = [post1, replay_insn](const std::uint32_t n) { return replay_insn(post1 ? 1 : 0, (post1 ? 1 : 2) + (1 + n) * 3); };
    const auto final_reuses             = [replay_insn](const std::uint32_t n) { return replay_insn(32 - n * 3, n * 3); };
    constexpr std::uint32_t reuse_start = 5;

    std::uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b = 0;
    bool use_b = false, use_halo = false;
    if (reuses == 0)
    {
        a0 = full_and(0);
    }
    else if (reuses < 3)
    {
        use_b = true;
        a0    = full_and(reuses >> 1);
        b     = final_reuses(reuses - (reuses >> 1));
    }
    else
    {
        use_halo                 = true;
        const std::uint32_t base = reuses >> 2;
        const std::uint32_t rem  = reuses & 3;
        a0                       = full_and(base);
        a1                       = replay_insn(reuse_start, (base + (rem > 2 ? 1 : 0)) * 3);
        a2                       = replay_insn(reuse_start, (base + (rem > 1 ? 1 : 0)) * 3);
        a3                       = final_reuses(base + (rem > 0 ? 1 : 0));
    }

    // ckernel_unpack_template(use_b, use_halo, a0, a1, a2, a3, 0 /*skip A*/, b, 0 /*skip B*/).program(). The HAL picks
    // the flags from the shape and leaves the words the expander does not read for that shape (A1-3 without halo, B and
    // skip B without B) unwritten. The skip words keep production's 0: zmask is always 0, so they are never issued.
    using namespace hal::mop;
    if (use_halo)
    {
        program(foreach_mask_bit(if_clear(a0, a1, a2, a3), otherwise(0 /*skip A*/)));
    }
    else if (use_b)
    {
        program(foreach_mask_bit(if_clear(a0, b), otherwise(0 /*skip A*/, 0 /*skip B*/)));
    }
    else
    {
        program(foreach_mask_bit(if_clear(a0), otherwise(0 /*skip A*/)));
    }
    // TTI_MOP_CFG(0)
    Runner<MopTemplate::Template0>::configure_mask_high<0>();
}

/**
 * @brief Configure the unpack thread for a custom_mm block matmul.
 *
 * Sets the unpacker X ends, records the MOP and replay buffer the execute runs, and resets the counters.
 *
 * @tparam transpose: Transpose the SrcA read, values = <true/false>
 * @tparam clear_src: Zero both SrcB banks once here, values = <true/false>. Only unpB_face_r_dim rows of each
 *                    SrcB face are unpacked, so zeroing the rest saves FPU power.
 * @param unpB_face_r_dim: Activation rows per face, 1, 2, 4 or 8. Sets unpacker 1's X end.
 * @param unpA_dst_format: Unpack destination format of the weights (SrcA); Bfp4_b selects a tuned sequence.
 * @param ct_dim: Output width in tiles, 1 to 34.
 * @param unpA_num_faces: Faces in a weight tile: 4 for [32, 32], 2 for [32, 16] ([16, 32] with transpose). Sets
 *                        unpacker 0's X end.
 * @note Call this before @ref _llk_unpack_AB_custom_mm_, and again after any other op has run, in particular one
 *       that writes SrcB: the execute does not reprogram the MOP, and the SrcB clear happens only here.
 * @note On the math thread, pair with @ref _llk_math_custom_mm_init_, giving it the same face count.
 */
template <bool transpose = false, bool clear_src = true>
inline void _llk_unpack_AB_custom_mm_init_(
    const std::uint32_t unpB_face_r_dim,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t unpA_num_faces = TILE_NUM_FACES)
{
    // Two-context handshake op: an earlier op's init may have left Auto TTSync on, and tt-metal calls no uninit.
    hal_port::unpack_sync_mode<hal_port::UnpackSyncMode::Handshake>();
    LLK_ASSERT(custom_mm_legal_in1_num_faces(unpA_num_faces), "custom_mm: in1 must have 4 faces, or 2 for a [32, 16] tile");
    namespace cfg = hal::cfg;
    namespace adc = hal::counters::adc;
    // cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose ? 1 : 0): one RMWCIB, compile-time value
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0, (transpose ? 1u : 0u)>();

    // UnpA unpacks full tiles, of 4 faces or 2 for a [32, 16] tile
    const std::uint32_t unpA_x_end = unpA_num_faces * FACE_R_DIM * FACE_C_DIM - 1;
    // UnpB unpacks [{1, 2, 4, 8}, 32] tiles which only have top two faces and only a single face per instruction
    // so a single instruction only unpacks unpB_face_r_dim rows
    const std::uint32_t unpB_x_end = unpB_face_r_dim * FACE_C_DIM - 1;
    // TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0), then the same for UNP_B
    adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = unpA_x_end}}.for_clients<adc::Client::Unpacker0>());
    adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = unpB_x_end}}.for_clients<adc::Client::Unpacker1>());

    // This is a profiling guided heuristic that slightly tunes the instruction sequence
    // More details under tt-metal#38518
    const bool post1 = unpA_dst_format == to_underlying(DataFormat::Bfp4_b);

    _llk_unpack_AB_custom_mm_mop_config_(ct_dim, post1);

    if constexpr (clear_src)
    {
        // Clear SrcB as we only unpack into 1/8 FPU rows so zeroing them gives power savings
        // This particular instruction clears both banks after waiting for both of them to be free
        // It must run alone: a both-bank SrcB clear drops SrcA writes that unpacker 0 makes while it runs
        hal_port::custom_mm_clear_srcb_both_banks();
    }

    // Reset counters here since reset in the execute API is at the end
    hal_port::custom_mm_reset_unpack_counters();
}

// Most k-tiles one SrcB base address covers: k-tile k reads faces k and k + 1 of an 8-bit Y + 8-bit Z face index,
// which reaches face 255 + 255 = 510, so k-tile 255 would need face 511 and alias.
constexpr std::uint32_t CUSTOM_MM_MAX_KT_PER_WALK = 255;

/**
 * @brief Issue the SrcA/SrcB unpack of a kt_dim x ct_dim block with the MOP that @ref _llk_unpack_AB_custom_mm_init_
 *        recorded. Shared by @ref _llk_unpack_AB_custom_mm_ and @ref _llk_unpack_AB_sdpa_custom_mm_.
 *
 * Splits kt_dim into ceil(kt_dim / CUSTOM_MM_MAX_KT_PER_WALK) SrcB face walks whose lengths differ by at most one.
 * Each walk sets the SrcB base, waits for its UNPACRs to issue and resets the counters. The SrcA base carries over.
 *
 * @param cfg: Config register pointer of the unpack thread.
 * @param address_a: L1 address of the first weight tile (SrcA), in 16-byte words.
 * @param address_b: L1 address of the first activation tile (SrcB), in 16-byte words.
 * @param tile_size_b: L1 stride between activation k-tiles, in 16-byte words. Above CUSTOM_MM_MAX_KT_PER_WALK k-tiles
 *                     it must equal the dense stride the counters step, two in0 faces (4 * face_r_dim words for
 *                     16-bit in0), since later walks are placed with it.
 * @param block_increment: SrcA address step to the next tile of a k-row, in 16-byte words.
 * @param inner_increment: SrcA address step from the last tile of a k-row to the next k-row, in 16-byte words.
 * @param kt_dim: Inner dimension in tiles, 1 or more.
 */
inline void _llk_unpack_AB_custom_mm_run_(
    volatile std::uint32_t* cfg,
    std::uint32_t address_a,
    const std::uint32_t address_b,
    const std::uint32_t tile_size_b,
    const std::uint32_t block_increment,
    const std::uint32_t inner_increment,
    const std::uint32_t kt_dim)
{
    LLK_ASSERT(kt_dim >= 1, "custom_mm: kt_dim must be at least 1");
    namespace hcfg = hal::cfg;
    namespace sync = hal::sync;

    // Whole-word MMIO stores into the state-CFG bank, as production.
    constexpr std::uint32_t base_a_addr   = hcfg::Unpacker[0].Cntx[0].Base.addr32(hcfg::Sec::S0);
    constexpr std::uint32_t base_b_addr   = hcfg::Unpacker[1].Cntx[0].Base.addr32(hcfg::Sec::S0);
    constexpr std::uint32_t scratch0_addr = hcfg::Scratch::val.addr32(hcfg::Sec::S0);
    constexpr std::uint32_t scratch1_addr = hcfg::Scratch::val.addr32(hcfg::Sec::S1);
    static_assert(base_a_addr == THCON_SEC0_REG3_Base_address_ADDR32 && base_b_addr == THCON_SEC1_REG3_Base_address_ADDR32, "unpacker base words");
    static_assert(scratch0_addr == SCRATCH_SEC0_val_ADDR32 && scratch1_addr == SCRATCH_SEC1_val_ADDR32, "scratch words");

    // Program the SrcA base once; CFGSHIFTMASK advances it across all walks
    cfg[base_a_addr] = address_a;
    // Setup CFGSHIFTMASK increments
    // block_increment moves to next tile in tile_row
    cfg[scratch0_addr] = block_increment;
    // inner_increment moves to next tile_row (next inner_dim)
    cfg[scratch1_addr] = inner_increment;

    using Mop = hal::mop::Runner<hal::mop::MopTemplate::Template0>;

    // Split kt_dim into near-equal walks; the first kt_dim % num_walks take one extra k-tile
    const std::uint32_t num_walks  = (kt_dim + CUSTOM_MM_MAX_KT_PER_WALK - 1) / CUSTOM_MM_MAX_KT_PER_WALK;
    const std::uint32_t base_kt    = kt_dim / num_walks;
    const std::uint32_t long_walks = kt_dim % num_walks;
    std::uint32_t walk_kt          = 0;
    std::uint32_t walk             = 0;
    for (std::uint32_t walk_start = 0; walk_start < kt_dim; walk_start += walk_kt, walk++)
    {
        walk_kt = base_kt + (walk < long_walks ? 1 : 0);

        // Program the SrcB base for this walk; the counters advance it within the walk
        cfg[base_b_addr] = address_b + walk_start * tile_size_b;

        // semaphore_post(semaphore::UNPACK_SYNC): MMIO
        sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();

        // TTI_STALLWAIT(STALL_UNPACK, TRISC_CFG)
        sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

        // One mop iteration per kt_dim loop, at most CUSTOM_MM_MAX_MOP_ITERATIONS per TT_MOP
        for (std::uint32_t kt_done = 0; kt_done < walk_kt; kt_done += CUSTOM_MM_MAX_MOP_ITERATIONS)
        {
            const std::uint32_t kt_left    = walk_kt - kt_done;
            const std::uint32_t iterations = kt_left < CUSTOM_MM_MAX_MOP_ITERATIONS ? kt_left : CUSTOM_MM_MAX_MOP_ITERATIONS;
            // TT_MOP(0, iterations - 1, 0): zmask 0, whose high half init wrote with MOP_CFG
            Mop::run<0 /*mask*/, hal::mop::MaskHigh::Preset>(iterations);
        }

        // Wait until every UNPACR of this walk has issued
        sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();

        // Wait for all contexts to be free
        hal_port::wait_for_next_context(1);
        hal_port::reset_config_context();

        // Reset counters at the end of each walk
        hal_port::custom_mm_reset_unpack_counters();
    }
}

/**
 * @brief Unpack a kt_dim x ct_dim block of weight tiles into SrcA and the matching activation tiles into SrcB.
 *
 * @tparam read_transposed: Walk the weight tiles column by column (ct_dim tiles with a stride of kt_dim, then the
 *                          next tile) instead of row by row, values = <true/false>
 * @param base_address_a: L1 address of the weights (SrcA), in the 16-byte-word encoding of L1_ADDRESS().
 * @param base_address_b: L1 address of the activations (SrcB), in the same encoding.
 * @param tile_index_a: First weight tile to read.
 * @param tile_index_b: First activation tile to read.
 * @param tile_size_a: Weight tile size, in 16-byte words.
 * @param tile_size_b: L1 stride between activation k-tiles, in 16-byte words; above CUSTOM_MM_MAX_KT_PER_WALK k-tiles
 *                     the dense two-face in0 stride, see @ref _llk_unpack_AB_custom_mm_run_.
 * @param kt_dim: Inner dimension in tiles, 1 or more.
 * @param ct_dim: Output width in tiles; must match the init, whose MOP bakes it in.
 * @note Call @ref _llk_unpack_AB_custom_mm_init_ first.
 * @note On the math thread, pair with @ref _llk_math_custom_mm_.
 */
template <bool read_transposed = false>
inline void _llk_unpack_AB_custom_mm_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim)
{
    volatile std::uint32_t* cfg = hal::cfg::detail::state_cfg_bank(); // get_cfg_pointer()

    const std::uint32_t block_increment = read_transposed ? kt_dim * tile_size_a : tile_size_a;
    const std::uint32_t inner_increment = read_transposed ? -(((ct_dim - 1) * kt_dim) - 1) * tile_size_a : tile_size_a;

    const std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    const std::uint32_t address_b = base_address_b + tile_size_b * tile_index_b;

    // Wait for all contexts to be free
    hal_port::wait_for_next_context(1);
    hal_port::reset_config_context();

    _llk_unpack_AB_custom_mm_run_(cfg, address_a, address_b, tile_size_b, block_increment, inner_increment, kt_dim);
}
