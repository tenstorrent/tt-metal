// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The fused K = 512, 1024 or 2048 topk_xl chunk split across threads: copy and face transposes on MATH, SFPU work on PACK.
// Chunk c runs stage j at step 9c + 2j, so MATH transposes one chunk while PACK sorts the other.
// Any other copy of this schedule must change together with it.

#pragma once

#include <cstdint>

#include "ckernel.h"

namespace topk_xl_split
{

constexpr std::uint8_t F2S                   = ckernel::semaphore::FPU_SFPU;
constexpr std::uint8_t S2F                   = ckernel::semaphore::UNPACK_MATH_DONE;
constexpr std::uint32_t STEPS_PER_CHUNK      = 9;
constexpr std::uint32_t STAGES_FIRST_CHUNK   = 7;
constexpr std::uint32_t STAGES_MERGING_CHUNK = 9;

inline bool decode_step(const std::uint32_t c1, const std::uint32_t r, const std::uint32_t num_chunks, std::uint32_t& chunk, std::uint32_t& stage)
{
    if ((r & 1) == 0)
    {
        chunk = c1;
        stage = r >> 1;
    }
    else
    {
        if (c1 == 0)
        {
            return false;
        }
        chunk = c1 - 1;
        stage = (r + STEPS_PER_CHUNK) >> 1;
    }
    return chunk < num_chunks && stage < (chunk == 0 ? STAGES_FIRST_CHUNK : STAGES_MERGING_CHUNK);
}

// K 1024 and 2048 rows of two or more chunks run chunks 0 and 1 side by side up to stage 6; K 512 rows keep the straight-line order.
constexpr std::uint32_t HEAD_STEPS = 14;

template <std::uint32_t K>
constexpr bool paired_head(const std::uint32_t num_chunks)
{
    return K != 512 && num_chunks >= 2;
}

template <std::uint32_t K>
constexpr std::uint32_t tiles_per_sequence = K == 2048 ? 2 : 1;

// Chunks alternate DEST banks (bank 1 starts at tile 4): the two in-flight chunks sit in distinct tiles, and the next row's first
// chunk can be copied into bank 1 while this row finishes. Stage 6's merge still reads the previous survivor in the other chunk's bank.
template <std::uint32_t K>
constexpr std::uint32_t slot_tile(const std::uint32_t from_last)
{
    return ((from_last & 1) ? 4 : 0) + ((from_last >> 1) & 1) * tiles_per_sequence<K>;
}

template <std::uint32_t K>
constexpr std::uint32_t chunk_tile(const std::uint32_t chunk, const std::uint32_t num_chunks)
{
    // K 512 rows of two chunks keep the one-thread layout (survivor in tile 0, chunk 1 in tile 1): the index split
    // writes half of tile 1 and the rest of it then holds what the one-thread kernel leaves there.
    if (K == 512 && num_chunks == 2)
    {
        return chunk;
    }
    return slot_tile<K>(num_chunks - 1 - chunk);
}

// Where chunk `chunk`'s stages 7 and 8 find the survivor.
template <std::uint32_t K>
constexpr std::uint32_t survivor_tile(const std::uint32_t chunk, const std::uint32_t num_chunks)
{
    return K == 512 && num_chunks == 2 ? 0 : chunk_tile<K>(chunk, num_chunks);
}

// Rows from a chunk's tiles to the survivor of the chunk before it, mod the 1024-row Dst address; slot_parity is the
// chunk's from_last & 3.
template <std::uint32_t K>
constexpr int survivor_distance(const std::uint32_t slot_parity)
{
    return static_cast<int>(((slot_tile<K>(slot_parity + 1) - slot_tile<K>(slot_parity)) & 15) << 6);
}

// Whether the next row's first chunk can be copied while this row finishes: it lands in bank 1, which the survivor and
// index tiles never use.
template <std::uint32_t K>
constexpr bool can_prefetch(const std::uint32_t num_chunks)
{
    return (num_chunks & 1) == 0 && !(K == 512 && num_chunks == 2);
}

#ifdef LLK_TRISC_MATH

inline void take_pack_token()
{
    ckernel::t6_semaphore_wait_on_zero<ckernel::p_stall::STALL_SYNC>(S2F);
    ckernel::t6_semaphore_get(S2F);
}

// Runs before PACK issues any SFPU instruction; start_pack's MATH_PACK post is then PACK's start signal.
inline void init_tokens()
{
    ckernel::t6_semaphore_init(F2S, 0, 15);
    ckernel::t6_semaphore_init(S2F, 0, 15);
}

inline void start_pack()
{
    ckernel::t6_semaphore_post<ckernel::p_stall::MATH | ckernel::p_stall::WAIT_SFPU>(ckernel::semaphore::MATH_PACK);
}

inline void release_src()
{
    TTI_SETRWC(ckernel::p_setrwc::CLR_AB, 0, 0, 0, 0, ckernel::p_setrwc::SET_ABD);
}

// One Dst section. The SrcA/SrcB releases keep the single-thread order (after the copy, halfway through the chunk, before
// the next copy), and every release and copy runs with the transpose CFG block closed, as in the single-thread kernel.
template <std::uint32_t K, typename CopyChunk>
inline void math_row(const std::uint32_t num_chunks, CopyChunk&& copy_chunk, const bool first_prefetched = false, const bool prefetch_next = false)
{
    std::uint32_t posted = 0, taken = 0, half_release_step = 0;
    bool prev_step_busy = false, cfg_open = false, half_release_due = false;

    const bool paired = paired_head<K>(num_chunks);
    // Slot 2 of a paired row starts at step HEAD_STEPS: chunk 1's stages 5 and 6 are done, so its stages 7 and 8 come two steps early.
    const std::uint32_t slot_step0 = paired ? HEAD_STEPS - 2 * STEPS_PER_CHUNK : 0;

    for (std::uint32_t c1 = 0; c1 <= num_chunks; c1++)
    {
        if (paired && c1 == 1)
        {
            continue;
        }
        const bool head = paired && c1 == 0;
        for (std::uint32_t r = 0, steps = head ? HEAD_STEPS : STEPS_PER_CHUNK; r < steps; r++)
        {
            std::uint32_t chunk, stage;
            if (head)
            {
                chunk = r & 1;
                stage = r >> 1;
            }
            else
            {
                if (!decode_step(c1, r, num_chunks, chunk, stage))
                {
                    prev_step_busy = false;
                    continue;
                }
                if (paired && c1 == 2 && (r & 1))
                {
                    stage += 2;
                    if (stage >= STAGES_MERGING_CHUNK)
                    {
                        prev_step_busy = false;
                        continue;
                    }
                }
            }
            const std::uint32_t step = head ? r : c1 * STEPS_PER_CHUNK + r + slot_step0;
            // Wait for every SFPU part at least two steps back: that covers this chunk's previous stage, the survivor
            // rebuild before a merge and the merge that frees a tile before its next copy.
            const std::uint32_t need = (paired && step == 1) ? posted : posted - (prev_step_busy ? 1 : 0);
            for (; taken < need; taken++)
            {
                take_pack_token();
            }

            if (first_prefetched && step == 0)
            {
                posted++;
                prev_step_busy = true;
                continue;
            }
            if (stage == 0)
            {
                if (cfg_open)
                {
                    ckernel::sfpu::leave_transpose_cfg_block();
                    cfg_open = false;
                }
                if (chunk > 0)
                {
                    if (paired && step == 1)
                    {
                        // Copy 0's valid flags may not be set yet, and the release would clear them first.
                        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::SRCA_VLD | ckernel::p_stall::SRCB_VLD);
                    }
                    release_src();
                    half_release_due  = true;
                    half_release_step = step + 5;
                }
                copy_chunk(chunk, chunk_tile<K>(chunk, num_chunks));
            }
            else
            {
                if (half_release_due && step >= half_release_step)
                {
                    if (cfg_open)
                    {
                        ckernel::sfpu::leave_transpose_cfg_block();
                        cfg_open = false;
                    }
                    release_src();
                    half_release_due = false;
                }
                if (!cfg_open)
                {
                    ckernel::sfpu::enter_transpose_cfg_block();
                    cfg_open = true;
                }
                ckernel::sfpu::_topk_xl_split_transpose_<K>(
                    (stage >= STAGES_FIRST_CHUNK ? survivor_tile<K>(chunk, num_chunks) : chunk_tile<K>(chunk, num_chunks)) << 6);
            }

            ckernel::t6_semaphore_post<ckernel::p_stall::MATH>(F2S);
            posted++;
            prev_step_busy = true;
        }
    }

    if (cfg_open)
    {
        ckernel::sfpu::leave_transpose_cfg_block();
    }
    release_src();
    if (prefetch_next)
    {
        // Stage 6 of the last chunk has read the other chunk's tiles; stages 7 and 8 stay in the survivor's bank.
        for (; taken + 2 < posted; taken++)
        {
            take_pack_token();
        }
        copy_chunk(0, chunk_tile<K>(0, num_chunks));
        ckernel::t6_semaphore_post<ckernel::p_stall::MATH>(F2S);
    }
    for (; taken < posted; taken++)
    {
        take_pack_token();
    }
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

inline void take_math_token()
{
    ckernel::t6_semaphore_wait_on_zero<ckernel::p_stall::STALL_SYNC>(F2S);
    ckernel::t6_semaphore_get(F2S);
}

// Waits for MATH's start signal, then owns the SFPU state: LaneConfig, LRegs, PACK's ADDR_MODs, replay and MOP.
template <std::uint32_t K>
inline void wait_start()
{
    _llk_packer_wait_for_math_done_();
    _llk_packer_set_math_semaphore_<ckernel::p_stall::NONE>();
    ckernel::sfpu::_topk_xl_split_sfpu_init_<K, true>();
}

// mirror_last: the last chunk's rebuild (stages 7 and 8) writes ascending, as for a mirrored segment of an op.
template <std::uint32_t K, bool mirror_last, std::uint32_t stage>
inline __attribute__((always_inline)) void sfpu_stage(const std::uint32_t chunk, const std::uint32_t num_chunks)
{
    const std::uint32_t tile_offset = chunk_tile<K>(chunk, num_chunks) << 6;
    const bool ascending            = chunk > 0;
    const bool rebuild_dir          = mirror_last && chunk + 1 == num_chunks;
    switch (stage)
    {
        case 0:
            ckernel::sfpu::_topk_xl_split_stamp_<K>(tile_offset, chunk);
            if constexpr (K == 2048)
            {
                ckernel::sfpu::_topk_xl_split_sort_head_2048_(tile_offset, ascending);
            }
            else if constexpr (K == 1024)
            {
                ckernel::sfpu::_topk_xl_split_sort_head_1024_(tile_offset, ascending);
            }
            else
            {
                ckernel::sfpu::_topk_xl_split_sort_head_512_<true>(tile_offset, ascending);
            }
            break;
        case 1:
            ckernel::sfpu::_topk_xl_split_stride2_<K, 4>(tile_offset, ascending);
            break;
        case 2:
            ckernel::sfpu::_topk_xl_split_columns_<K, 0x5050>(tile_offset, ascending);
            break;
        case 3:
            ckernel::sfpu::_topk_xl_split_stride2_<K, 8>(tile_offset, ascending);
            break;
        case 4:
            ckernel::sfpu::_topk_xl_split_columns_<K, 0x5500>(tile_offset, ascending);
            break;
        case 5:
            ckernel::sfpu::_topk_xl_split_stride2_<K, 16>(tile_offset, ascending);
            break;
        case 6:
            ckernel::sfpu::_topk_xl_split_columns_<K, 0>(tile_offset, ascending);
            if constexpr (K == 512)
            {
                if (chunk > 0 && num_chunks == 2)
                {
                    ckernel::sfpu::_topk_xl_split_merge_<K, 64, true>(0);
                    break;
                }
            }
            if (chunk > 0)
            {
                switch ((num_chunks - 1 - chunk) & 3)
                {
                    case 1:
                        ckernel::sfpu::_topk_xl_split_merge_<K, survivor_distance<K>(1), true>(tile_offset);
                        break;
                    case 3:
                        ckernel::sfpu::_topk_xl_split_merge_<K, survivor_distance<K>(3), true>(tile_offset);
                        break;
                    default:
                        ckernel::sfpu::_topk_xl_split_merge_<K, survivor_distance<K>(0), true>(tile_offset);
                        break;
                }
            }
            break;
        case 7:
            ckernel::sfpu::_topk_xl_split_rebuild_build_<K>(survivor_tile<K>(chunk, num_chunks) << 6, rebuild_dir);
            break;
        default:
            ckernel::sfpu::_topk_xl_split_columns_<K, 0>(survivor_tile<K>(chunk, num_chunks) << 6, rebuild_dir);
            break;
    }
}

// One SFPU part. K 1024 and 2048 take it as a call so that each stage body exists once whatever the schedule (their pack code
// fills TRISC2); K 512 keeps every step inline in the straight-line slot.
template <std::uint32_t K, bool mirror_last, std::uint32_t stage>
inline __attribute__((always_inline)) void sfpu_step_inline(const std::uint32_t chunk, const std::uint32_t num_chunks)
{
    take_math_token();
    sfpu_stage<K, mirror_last, stage>(chunk, num_chunks);
    ckernel::t6_semaphore_post<ckernel::p_stall::WAIT_SFPU>(S2F);
}

template <std::uint32_t K, bool mirror_last, std::uint32_t stage>
inline __attribute__((noinline, noclone)) void sfpu_step_call(const std::uint32_t chunk, const std::uint32_t num_chunks)
{
    sfpu_step_inline<K, mirror_last, stage>(chunk, num_chunks);
}

template <std::uint32_t K, bool mirror_last, std::uint32_t stage>
inline __attribute__((always_inline)) void sfpu_step(const std::uint32_t chunk, const std::uint32_t num_chunks)
{
    if constexpr (K == 512)
    {
        sfpu_step_inline<K, mirror_last, stage>(chunk, num_chunks);
    }
    else
    {
        sfpu_step_call<K, mirror_last, stage>(chunk, num_chunks);
    }
}

// PACK's steps of chunk slot c1 in decode_step's order: chunk c1's stages 0 to 4 between chunk c1 - 1's stages 5 to 8
// (7 and 8 only for a merged chunk).
template <std::uint32_t K, bool mirror_last>
inline __attribute__((noinline)) void pack_slot(const std::uint32_t c1, const std::uint32_t num_chunks)
{
    const bool cur = c1 < num_chunks, prev = c1 > 0, prev_merged = c1 > 1;
    if (cur)
    {
        sfpu_step<K, mirror_last, 0>(c1, num_chunks);
    }
    if (prev)
    {
        sfpu_step<K, mirror_last, 5>(c1 - 1, num_chunks);
    }
    if (cur)
    {
        sfpu_step<K, mirror_last, 1>(c1, num_chunks);
    }
    if (prev)
    {
        sfpu_step<K, mirror_last, 6>(c1 - 1, num_chunks);
    }
    if (cur)
    {
        sfpu_step<K, mirror_last, 2>(c1, num_chunks);
    }
    if (prev_merged)
    {
        sfpu_step<K, mirror_last, 7>(c1 - 1, num_chunks);
    }
    if (cur)
    {
        sfpu_step<K, mirror_last, 3>(c1, num_chunks);
    }
    if (prev_merged)
    {
        sfpu_step<K, mirror_last, 8>(c1 - 1, num_chunks);
    }
    if (cur)
    {
        sfpu_step<K, mirror_last, 4>(c1, num_chunks);
    }
}

// Paired head: chunks 0 and 1 stages 0 to 6 side by side, then slot 2 with chunk 1's stages 7 and 8 two steps early.
template <std::uint32_t K, bool mirror_last>
inline __attribute__((noinline)) void pack_head(const std::uint32_t num_chunks)
{
    sfpu_step<K, mirror_last, 0>(0, num_chunks);
    sfpu_step<K, mirror_last, 0>(1, num_chunks);
    sfpu_step<K, mirror_last, 1>(0, num_chunks);
    sfpu_step<K, mirror_last, 1>(1, num_chunks);
    sfpu_step<K, mirror_last, 2>(0, num_chunks);
    sfpu_step<K, mirror_last, 2>(1, num_chunks);
    sfpu_step<K, mirror_last, 3>(0, num_chunks);
    sfpu_step<K, mirror_last, 3>(1, num_chunks);
    sfpu_step<K, mirror_last, 4>(0, num_chunks);
    sfpu_step<K, mirror_last, 4>(1, num_chunks);
    sfpu_step<K, mirror_last, 5>(0, num_chunks);
    sfpu_step<K, mirror_last, 5>(1, num_chunks);
    sfpu_step<K, mirror_last, 6>(0, num_chunks);
    sfpu_step<K, mirror_last, 6>(1, num_chunks);
    const bool cur = num_chunks > 2;
    if (cur)
    {
        sfpu_step<K, mirror_last, 0>(2, num_chunks);
    }
    sfpu_step<K, mirror_last, 7>(1, num_chunks);
    if (cur)
    {
        sfpu_step<K, mirror_last, 1>(2, num_chunks);
    }
    sfpu_step<K, mirror_last, 8>(1, num_chunks);
    if (cur)
    {
        sfpu_step<K, mirror_last, 2>(2, num_chunks);
        sfpu_step<K, mirror_last, 3>(2, num_chunks);
        sfpu_step<K, mirror_last, 4>(2, num_chunks);
    }
}

// Every SFPU part of one Dst section in MATH's step order, survivor left fused in sequence 0.
template <std::uint32_t K, bool mirror_last = false>
inline void pack_row(const std::uint32_t num_chunks)
{
    TTI_STALLWAIT(ckernel::p_stall::STALL_SFPU, ckernel::p_stall::MATH);
    std::uint32_t c1 = 0;
    if (paired_head<K>(num_chunks))
    {
        pack_head<K, mirror_last>(num_chunks);
        c1 = 3;
    }
    for (; c1 <= num_chunks; c1++)
    {
        pack_slot<K, mirror_last>(c1, num_chunks);
    }
}

// Releases Dst to MATH, clearing only bank 0: bank 1 may already hold the next row's first chunk.
template <bool fp32>
inline void release_dest_bank0()
{
    TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::PACK);
    TTI_ZEROACC(ckernel::p_zeroacc::CLR_HALF, fp32, 0, ckernel::ADDR_MOD_1, 0);
    _llk_packer_set_math_semaphore_<ckernel::p_stall::NONE>();
}

// The row-major global index split of sequence 0 into values (sequence 0) and indices (the tiles after it), on PACK.
template <std::uint32_t K>
inline void split_indices(const std::uint32_t seg_base)
{
    ckernel::sfpu::_topk_xl_separate_indices_row_major_global_init_();
    ckernel::sfpu::_topk_xl_split_begin_(0);
    if (seg_base != 0)
    {
        ckernel::sfpu::_topk_xl_separate_indices_row_major_global_base_<K>(seg_base);
    }
    else
    {
        ckernel::sfpu::_topk_xl_separate_indices_row_major_global_<K>();
    }
    TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
    TTI_STALLWAIT(ckernel::p_stall::STALL_PACK, ckernel::p_stall::WAIT_SFPU);
}

#endif // LLK_TRISC_PACK

} // namespace topk_xl_split
