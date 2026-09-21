// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifdef TRISC_PACK
namespace ckernel::sfpu {
template <int vectors>
inline void sprint_v2_compensated_part() {
    static_assert(vectors > 0 && vectors % 2 == 0);
#pragma GCC unroll 8
    for (int i = 0; i < vectors; i += 2) {
        TTI_REPLAY(0, 15, 0, 0);
        TTI_REPLAY(1, 14, 0, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
inline void sprint_v2_compensated_overlap() {
    static_assert(DST_SYNC_MODE == DstSync::SyncHalf);
    static_assert(!DST_ACCUM_MODE);
    // Current pack half A remains held. Count2 therefore proves BOTH A and B
    // are committed, not merely that some half is available.
    while (semaphore_read(semaphore::MATH_PACK) != 2) {}
    const uint32_t sfpu_base = get_dest_buffer_base() ^ DEST_REGISTER_HALF_SIZE;
    // Independent review requires an explicit prior-SFPU drain before changing
    // its thread-local address. This does not wait for outstanding PACK work.
    TTI_STALLWAIT(p_stall::STALL_CFG | p_stall::STALL_SYNC | p_stall::STALL_SFPU,
                  p_stall::WAIT_SFPU);
    // Only SFPU's thread-local MATH offset changes. Outstanding PACK reads
    // retain their separate PACK_SEC0 offset pointing to A.
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, sfpu_base);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    sprint_v2_compensated_part<16>();
    // Include SYNC so the release's following STALLWAIT cannot replace this
    // wait gate before the pending SFPU stores have completed.
    TTI_STALLWAIT(p_stall::STALL_MATH | p_stall::STALL_SYNC | p_stall::STALL_SFPU,
                  p_stall::WAIT_SFPU);
    // Normal completion of A: pack drain, clear A, release, flip PACK to B.
    // ZEROACC(CLR_HALF) ignores AddrMod and doesn't alter B's SFPU RWC32.
    llk_pack_dest_section_done<false>();
    sprint_v2_compensated_part<16>();
    math::clear_dst_reg_addr();
}
}  // namespace ckernel::sfpu
#endif
