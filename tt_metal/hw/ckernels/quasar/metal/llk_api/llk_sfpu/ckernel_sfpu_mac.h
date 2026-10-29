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
// over-count swallows whatever the caller emits next - here that would be the
// STALL_SFPU/MATH TTI_STALLWAIT from _llk_math_eltwise_ternary_sfpu_start_, which is
// recorded under NoExec and so never runs inline, leaving the SFPU unsynchronised
// against MATH before the first replay.
//
// The lengths below are the instructions sfpi emits for mac_init()'s body:
//   3x SFPLOAD, SFPMAD, [SFPSTOCHRND for bf16,] SFPSTORE
// These differ from Wormhole, which needs one more on each path: sfpi there inserts an
// SFPNOP after SFPMAD for the LREG write latency, a hazard Blackhole does not have.
// The count is therefore a property of what sfpi emits for this arch, not of the source
// below - if mac_init()'s body changes, or sfpi changes how it schedules the block,
// re-check these against the generated kernel's disassembly.
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
    // All callers use tile indices (0, 1, 2, 0), i.e. dst_reg[0], dst_reg[32], dst_reg[64]
    // (sfpi indexes dst_reg in rows; the row stride is 2, so tile n starts at row n * 32).
    //
    // The SFPLOAD/SFPSTORE instruction mod describes the DST register layout, which is
    // 32-bit only when fp32 dest accumulation is enabled - it is not a property of the
    // input tensor's data format.  Keying it off data_format would issue FP32-mode
    // accesses against a 16-bit DST whenever the inputs are fp32 but the output dtype
    // (which is what drives fp32_dest_acc_en) is bf16, and 16-bit accesses against a
    // 32-bit DST in the mirrored case.  DataLayout::FSrcB is mod 0, i.e. "whatever
    // format DST is configured with".
    //
    // sfpi defaults the loads to the arch's no-increment addr_mod (ADDR_MOD_7 here), which
    // is what we want; only the store overrides it, with the auto-advancing mod set above.
    constexpr sfpi::DataLayout dst_fmt = is_fp32_dest_acc_en ? sfpi::DataLayout::F32 : sfpi::DataLayout::FSrcB;

    lltt::record<lltt::NoExec>(MAC_REPLAY_SLOT, mac_replay_len<is_fp32_dest_acc_en>);
    sfpi::vFloat a = sfpi::dst_reg[0].mode<dst_fmt>();
    sfpi::vFloat b = sfpi::dst_reg[32].mode<dst_fmt>();
    sfpi::vFloat c = sfpi::dst_reg[64].mode<dst_fmt>();
    sfpi::vFloat result = a * b + c;
    if constexpr (!is_fp32_dest_acc_en) {
        // SFPMAD accumulates in fp32; an SFPSTORE into a 16-bit DST truncates the
        // mantissa, so round to nearest-even on the way down to bf16 instead.
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
    }
    sfpi::dst_reg[0].mode<dst_fmt>(ADDR_MOD_6) = result;
}

// mac: out = a * b + c, computed in FP32 accumulator via SFPMAD.
//
// The replay sequence is recorded once in mac_init (with fixed dest offsets 0, 64, 128
// matching tile indices 0, 1, 2) and replayed ITERATIONS times here.
// ADDR_MOD_6 on SFPSTORE auto-advances the dest base register by 2 rows per
// replay, so the next replay's SFPLOADs read the next row group automatically.
// This avoids the explicit sfpi::dst_reg++ used in a plain for-loop, which
// only advances the write counter and not the read counter.
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
