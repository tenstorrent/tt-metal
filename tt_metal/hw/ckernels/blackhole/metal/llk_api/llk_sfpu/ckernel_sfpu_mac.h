// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "llk_defs.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Replay slot and length shared by mac_init() (which records) and calculate_mac()
// (which replays).  lltt::record() captures exactly `length` instructions, so the two
// must agree: an under-count leaves the tail of the body out of the buffer, an
// over-count swallows whatever the caller emits next (which is why the body below is
// written with explicit TTI_* intrinsics rather than sfpi C++ - the instruction count
// then follows from the source and does not depend on how sfpi schedules the block).
//
// The body is 3x SFPLOAD, SFPMAD, [SFP_STOCH_RND when DST is 16-bit,] SFPSTORE.
inline constexpr int MAC_REPLAY_SLOT = 0;
template <bool is_fp32_dest_acc_en>
inline constexpr int mac_replay_len = is_fp32_dest_acc_en ? 5 : 6;

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, DataFormat data_format>
inline void mac_init() {
    // eltwise_ternary_sfpu_configure_addrmod only sets ADDR_MOD_6 (dest.incr=2)
    // for SfpuType::where.  mac's replay sequence uses ADDR_MOD_6 on SFPSTORE
    // (physical slot 6, direct on Blackhole since there is no addr_mod_base
    // offset), so we must configure it explicitly here.
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);

    // Record the replay sequence once at init time with fixed dest offsets.
    // All callers use tile indices (0, 1, 2, 0); a tile is 32 rows and the dest
    // offsets are in half-rows, so the operands sit at 0, 64 and 128.
    //
    // The SFPLOAD/SFPSTORE instruction mod describes the DST register layout, which is
    // 32-bit only when fp32 dest accumulation is enabled - it is not a property of the
    // input tensor's data format.  Keying it off data_format would issue FP32-mode
    // accesses against a 16-bit DST whenever the inputs are fp32 but the output dtype
    // (which is what drives fp32_dest_acc_en) is bf16, and 16-bit accesses against a
    // 32-bit DST in the mirrored case.  InstrModLoadStore::DEFAULT is mod 0, i.e.
    // "whatever format DST is configured with".
    //
    // The loads use ADDR_MOD_7 (no increment); only the store overrides it, with the
    // auto-advancing mod set above.
    constexpr std::uint32_t dst_mod =
        is_fp32_dest_acc_en ? static_cast<std::uint32_t>(InstrModLoadStore::FP32)
                            : static_cast<std::uint32_t>(InstrModLoadStore::DEFAULT);

    lltt::record<lltt::NoExec>(MAC_REPLAY_SLOT, mac_replay_len<is_fp32_dest_acc_en>);
    TTI_SFPLOAD(p_sfpu::LREG0, dst_mod, ADDR_MOD_7, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, dst_mod, ADDR_MOD_7, 64);
    TTI_SFPLOAD(p_sfpu::LREG2, dst_mod, ADDR_MOD_7, 128);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    if constexpr (!is_fp32_dest_acc_en) {
        // SFPMAD accumulates in fp32; an SFPSTORE into a 16-bit DST truncates the
        // mantissa, so round to nearest-even on the way down to bf16 instead.
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            0,
            p_sfpu::LREG0,
            p_sfpu::LREG0,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    }
    TTI_SFPSTORE(p_sfpu::LREG0, dst_mod, ADDR_MOD_6, 0);
}

// mac: out = a * b + c, computed in FP32 accumulator via SFPMAD.
//
// The replay sequence is recorded once in mac_init (with fixed dest offsets 0, 64, 128
// matching tile indices 0, 1, 2) and replayed ITERATIONS times here.
// ADDR_MOD_6 on SFPSTORE auto-advances the dest base register by 2 rows per
// replay, so the next replay's SFPLOADs read the next row group automatically.
//
// The dst_index_* parameters are accepted for signature compatibility with the
// ternary SFPU dispatch but are NOT used: the dest offsets are baked into the
// recorded replay sequence, so the operand tiles are always (0, 1, 2) and the
// result is always written to tile 0.  Passing anything else has no effect.
//
// Because the replay slots (0..6) are shared with other SFPU ops, this must be
// replayed while mac_init's recording is still the resident one - i.e. a
// mac_tile call is only valid immediately after mac_tile_init.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_mac(
    [[maybe_unused]] const uint dst_index_in0,  // input a  (fixed at 0)
    [[maybe_unused]] const uint dst_index_in1,  // input b  (fixed at 1)
    [[maybe_unused]] const uint dst_index_in2,  // input c  (fixed at 2)
    [[maybe_unused]] const uint dst_index_out) {  // output  (fixed at 0)
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_mac(). Supported data formats are: Float32, Float16_b.");

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        lltt::replay(MAC_REPLAY_SLOT, mac_replay_len<is_fp32_dest_acc_en>);
    }
}

}  // namespace ckernel::sfpu
