
#pragma once

#include "ckernel_defs.h"
#include "ckernel_include.h"
#include "cmath_common.h"
#ifdef PERF_DUMP
#include "ckernel_perf_api.h"
#endif

using namespace ckernel::math;

template <DstSync Dst>
inline void llk_math_wait_for_dest_available() {
    // These liteweight functions for sync with packer imply
    // no mode change - entire epoch is either double buffer or single buffer
#ifdef PERF_DUMP
    if constexpr(MATH_PACK_DECOUPLE == 0) {
        math_dest_wait();
    }
#else
    math_dest_wait();
#endif
}

template <DstSync Dst = SyncFull, bool is_fp32_dest_acc_en = false>
inline void llk_math_dest_section_done() {
#ifdef PERF_DUMP
    if constexpr(MATH_PACK_DECOUPLE) {
        return;
    }
#endif

    constexpr uint32_t DEST_NUM_TILES_SHIFT = is_fp32_dest_acc_en ? (1) : (0);
    constexpr uint32_t DEST_NUM_TILES = DEST_NUM_TILES_FP16 >> DEST_NUM_TILES_SHIFT;

    set_math_semaphores();
    if constexpr ((Dst == DstSync::SyncHalf) || (Dst == DstSync::SyncTile2)) {
        math_sync_tile_dst_index = 0;
        dest_section_flip();
    } else if constexpr (Dst == DstSync::SyncTile16) {
        math_sync_tile_dst_index++;
        math_sync_tile_dst_index &= (DEST_NUM_TILES - 1);
    }
}

template <DstSync Dst, bool is_fp32_dest_acc_en = false>
inline void llk_math_pack_sync_init() {
#ifdef PERF_DUMP
    if constexpr(MATH_PACK_DECOUPLE) {
        return;
    }
#endif
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0) {
    };  // Wait for previous packs to finish before claiming all dest
    if constexpr (Dst == DstSync::SyncFull) {
        TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_1);
        reset_dest_offset_id();
        set_dest_section_base<StartZero>();
    } else if constexpr (Dst == DstSync::SyncHalf) {
        TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
        reset_dest_offset_id();
        set_dest_section_base<StartZero>();
    } else if constexpr (Dst == DstSync::SyncTile2) {
        TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
        reset_dest_offset_id();
        set_dest_section_base<StartZero>();
        math_sync_tile_dst_index = 0;
    } else {
        static_assert(Dst == DstSync::SyncTile16);

        constexpr uint32_t DEST_NUM_TILES_SHIFT = is_fp32_dest_acc_en ? (1) : (0);
        constexpr uint32_t DEST_NUM_TILES = DEST_NUM_TILES_FP16 >> DEST_NUM_TILES_SHIFT;
        static_assert(DEST_NUM_TILES > 15);

        TTI_SEMINIT(15, 0, p_stall::SEMAPHORE_1);
        reset_dest_offset_id();
        set_dest_section_base<StartZero>();
        math_sync_tile_dst_index = 0;
    }
}

template <bool mail2math=true, bool mail2pack=true>
inline void llk_math_get_tile(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t* p_tile) {
    if constexpr (mail2math) {
       *p_tile = mailbox_read(ThreadId::UnpackThreadId);
    } else {
       *p_tile = 0x0;
    }

}

template <bool mail2math=true, bool mail2pack=true>
inline void llk_math_release_tile(std::uint32_t operand) {
    if constexpr (mail2math) {
       semaphore_get(semaphore::UNPACK_OPERAND_SYNC);
    }
}
