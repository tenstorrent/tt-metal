// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifdef TRISC_PACK
namespace ckernel {
// Only the first two numerator batches may use this. The next two batches
// overwrite slots0..5, consume retained correction6, and use canonical release,
// so BOTH halves are cleared before the denominator and following matmuls.
// Exact canonical pack completion, semaphore and bank selection sequence,
// minus ZEROACC. Source: Blackhole llk_pack_common.h:_llk_pack_dest_section_done_.
inline void sprint_v2_release_preserving_dst() {
    static_assert(DST_SYNC_MODE == DstSync::SyncHalf);
    static_assert(!DST_ACCUM_MODE);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);
    _llk_packer_set_math_semaphore_<p_stall::NONE>();
    flip_packer_dest_offset_id();
    select_packer_dest_registers<DstSync::SyncHalf>();
}
}
#endif
