// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// A row of two or more K = 512, 1024 or 2048 chunks on Blackhole: copies and face transposes on MATH, SFPU work on
// PACK. Chunk c runs stage j at step 9c + 2j, so MATH transposes one chunk while PACK sorts the other.

#include <cstdint>

namespace topk_large_indices_split {

constexpr std::uint8_t F2S = ckernel::semaphore::FPU_SFPU;
constexpr std::uint8_t S2F = ckernel::semaphore::UNPACK_MATH_DONE;
constexpr std::uint32_t STEPS_PER_CHUNK = 9;
constexpr std::uint32_t STAGES_FIRST_CHUNK = 7;
constexpr std::uint32_t STAGES_MERGING_CHUNK = 9;

inline bool decode_step(
    const std::uint32_t c1,
    const std::uint32_t r,
    const std::uint32_t num_chunks,
    std::uint32_t& chunk,
    std::uint32_t& stage) {
    if ((r & 1) == 0) {
        chunk = c1;
        stage = r >> 1;
    } else {
        if (c1 == 0) {
            return false;
        }
        chunk = c1 - 1;
        stage = (r + STEPS_PER_CHUNK) >> 1;
    }
    return chunk < num_chunks && stage < (chunk == 0 ? STAGES_FIRST_CHUNK : STAGES_MERGING_CHUNK);
}

// Tiles of one chunk: a fused sequence, or a Classic chunk's value and index tiles.
template <std::uint32_t K, bool classic>
constexpr std::uint32_t chunk_tiles = classic ? 2 : (K == 2048 ? 2 : 1);

template <std::uint32_t K, bool classic>
constexpr bool supported_split = (K == 512 || (K >= 1024 && !classic));

// DEST bank 1 starts at tile 4 in 32-bit mode. Fused chunks alternate banks, two slots per bank in turn, and each merge
// leaves the survivor in the merged chunk's tiles, so MATH's face moves on one chunk never share a bank with PACK's
// SFPU work on the other. Slots count back from the last chunk, so the row ends with the survivor in tile 0.
template <std::uint32_t K>
constexpr std::uint32_t slot_tile(const std::uint32_t from_last) {
    return ((from_last & 1) ? 4 : 0) + ((from_last >> 1) & 1) * (K == 2048 ? 2 : 1);
}

template <std::uint32_t K, bool classic>
constexpr std::uint32_t chunk_tile(const std::uint32_t chunk, const std::uint32_t num_chunks) {
    if constexpr (classic) {
        return chunk_tiles<K, classic> * (chunk == 0 ? 0 : 2 - (chunk & 1));
    }
    // K 512 rows of two chunks keep the one-thread layout (survivor in tile 0, chunk 1 in tile 1): the index split
    // writes half of tile 1 and the rest of it then holds what the one-thread kernel leaves there.
    if (K == 512 && num_chunks == 2) {
        return chunk;
    }
    return slot_tile<K>(num_chunks - 1 - chunk);
}

// Where chunk `chunk`'s stages 7 and 8 find the survivor.
template <std::uint32_t K, bool classic>
constexpr std::uint32_t survivor_tile(const std::uint32_t chunk, const std::uint32_t num_chunks) {
    return classic || (K == 512 && num_chunks == 2) ? 0 : chunk_tile<K, classic>(chunk, num_chunks);
}

// Rows from a fused chunk's tiles to the survivor of the chunk before it, modulo the 1024-row Dst address; slot_parity
// is the chunk's from_last & 3.
template <std::uint32_t K>
constexpr int survivor_distance(const std::uint32_t slot_parity) {
    return static_cast<int>(((slot_tile<K>(slot_parity + 1) - slot_tile<K>(slot_parity)) & 15) << 6);
}

// Whether the next row's first fused chunk can be copied while this row finishes: it lands in bank 1, which the
// survivor and index tiles never use.
template <std::uint32_t K>
constexpr bool can_prefetch(const std::uint32_t num_chunks) {
    return (num_chunks & 1) == 0 && !(K == 512 && num_chunks == 2);
}

#ifdef TRISC_MATH

inline void take_pack_token() {
    ckernel::t6_semaphore_wait_on_zero<ckernel::p_stall::STALL_SYNC>(S2F);
    ckernel::t6_semaphore_get(S2F);
}

// Once per kernel, before PACK issues any SFPU instruction; the MATH_PACK post is PACK's start signal.
inline void start() {
    ckernel::t6_semaphore_init(F2S, 0, 15);
    ckernel::t6_semaphore_init(S2F, 0, 15);
    ckernel::t6_semaphore_post<ckernel::p_stall::MATH | ckernel::p_stall::WAIT_SFPU>(ckernel::semaphore::MATH_PACK);
}

// A token that carries no step: MATH's row set-up is done.
inline void post_token() { ckernel::t6_semaphore_post<ckernel::p_stall::MATH | ckernel::p_stall::WAIT_SFPU>(F2S); }

inline void release_src() { TTI_SETRWC(ckernel::p_setrwc::CLR_AB, 0, 0, 0, 0, ckernel::p_setrwc::SET_ABD); }

// The SrcA/SrcB releases keep the single-thread order (after the copy, halfway through the chunk, before the next
// copy), and every release and copy runs with the transpose CFG block closed, as in the single-thread kernel.
template <std::uint32_t K, bool classic, typename CopyChunk>
inline void math_row(
    const std::uint32_t num_chunks,
    CopyChunk&& copy_chunk,
    const bool first_prefetched = false,
    const bool prefetch_next = false) {
    static_assert(
        supported_split<K, classic>, "the split takes fused K = 512, 1024 and 2048 rows and Classic K = 512 rows");
    std::uint32_t posted = 0, taken = 0, half_release_step = 0;
    bool prev_step_busy = false, cfg_open = false, half_release_due = false;

    for (std::uint32_t c1 = 0; c1 <= num_chunks; c1++) {
        for (std::uint32_t r = 0; r < STEPS_PER_CHUNK; r++) {
            std::uint32_t chunk, stage;
            if (!decode_step(c1, r, num_chunks, chunk, stage)) {
                prev_step_busy = false;
                continue;
            }
            const std::uint32_t step = c1 * STEPS_PER_CHUNK + r;
            // Wait for every SFPU part at least two steps back: that covers this chunk's previous stage, the survivor
            // rebuild before a merge and the merge that frees a tile before its next copy.
            const std::uint32_t need = posted - (prev_step_busy ? 1 : 0);
            for (; taken < need; taken++) {
                take_pack_token();
            }

            if (first_prefetched && step == 0) {
                posted++;
                prev_step_busy = true;
                continue;
            }
            if (stage == 0) {
                if (cfg_open) {
                    ckernel::sfpu::leave_transpose_cfg_block();
                    cfg_open = false;
                }
                if (chunk > 0) {
                    release_src();
                    half_release_due = true;
                    half_release_step = step + 5;
                }
                copy_chunk(chunk, chunk_tile<K, classic>(chunk, num_chunks));
            } else {
                if (half_release_due && step >= half_release_step) {
                    if (cfg_open) {
                        ckernel::sfpu::leave_transpose_cfg_block();
                        cfg_open = false;
                    }
                    release_src();
                    half_release_due = false;
                }
                if (!cfg_open) {
                    ckernel::sfpu::enter_transpose_cfg_block();
                    cfg_open = true;
                }
                const std::uint32_t tile = stage >= STAGES_FIRST_CHUNK ? survivor_tile<K, classic>(chunk, num_chunks)
                                                                       : chunk_tile<K, classic>(chunk, num_chunks);
                if constexpr (classic) {
                    if (stage >= STAGES_FIRST_CHUNK) {
                        ckernel::sfpu::_topk_xl_split_transpose_unfused_512_(0);
                    } else {
                        ckernel::sfpu::_topk_xl_split_transpose_<K>(tile << 6);
                    }
                } else {
                    ckernel::sfpu::_topk_xl_split_transpose_<K>(tile << 6);
                }
            }

            ckernel::t6_semaphore_post<ckernel::p_stall::MATH>(F2S);
            posted++;
            prev_step_busy = true;
        }
    }

    if (cfg_open) {
        ckernel::sfpu::leave_transpose_cfg_block();
    }
    release_src();
    if (prefetch_next) {
        // Stage 6 of the last chunk has read the other chunk's tiles; stages 7 and 8 stay in the survivor's bank.
        for (; taken + 2 < posted; taken++) {
            take_pack_token();
        }
        copy_chunk(0, chunk_tile<K, classic>(0, num_chunks));
        ckernel::t6_semaphore_post<ckernel::p_stall::MATH>(F2S);
    }
    for (; taken < posted; taken++) {
        take_pack_token();
    }
    take_pack_token();
}

#endif  // TRISC_MATH

#ifdef TRISC_PACK

inline void take_math_token() {
    ckernel::t6_semaphore_wait_on_zero<ckernel::p_stall::STALL_SYNC>(F2S);
    ckernel::t6_semaphore_get(F2S);
}

// Once per kernel: waits for MATH's start signal.
inline void wait_start() {
    _llk_packer_wait_for_math_done_();
    _llk_packer_set_math_semaphore_<ckernel::p_stall::NONE>();
}

// Releases Dst to MATH, clearing only bank 0: bank 1 may already hold the next row's first chunk.
template <bool fp32>
inline void release_dest_bank0() {
    TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::PACK);
    TTI_ZEROACC(ckernel::p_zeroacc::CLR_HALF, fp32, 0, ckernel::ADDR_MOD_1, 0);
    _llk_packer_set_math_semaphore_<ckernel::p_stall::NONE>();
}

// rebuild_dir: the direction of stages 7 and 8 (ascending for the last chunk of a mirrored segment).
template <std::uint32_t K, bool classic>
inline __attribute__((always_inline)) void sfpu_stage_body(
    const std::uint32_t chunk, const std::uint32_t stage, const std::uint32_t num_chunks, const bool rebuild_dir) {
    static_assert(
        supported_split<K, classic>, "the split takes fused K = 512, 1024 and 2048 rows and Classic K = 512 rows");
    using namespace ckernel::sfpu;
    constexpr int chunk_rows = 64 * chunk_tiles<K, classic>;
    const std::uint32_t tile_offset = chunk_tile<K, classic>(chunk, num_chunks) << 6;
    const bool ascending = chunk > 0;
    if constexpr (classic) {
        // The other chunk's unfused segments leave index tracking on.
        if (stage < STAGES_FIRST_CHUNK) {
            _init_sfpu_config_reg();
        }
    }
    switch (stage) {
        case 0:
            if constexpr (classic) {
                addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_3);
            }
            _topk_xl_split_stamp_<K>(tile_offset, classic ? 0 : chunk);
            if constexpr (K == 2048) {
                _topk_xl_split_sort_head_2048_(tile_offset, ascending);
            } else if constexpr (K == 1024) {
                _topk_xl_split_sort_head_1024_(tile_offset, ascending);
            } else {
                _topk_xl_split_sort_head_512_<!classic>(tile_offset, ascending);
            }
            break;
        case 1: _topk_xl_split_stride2_<K, 4>(tile_offset, ascending); break;
        case 2: _topk_xl_split_columns_<K, 0x5050>(tile_offset, ascending); break;
        case 3: _topk_xl_split_stride2_<K, 8>(tile_offset, ascending); break;
        case 4: _topk_xl_split_columns_<K, 0x5500>(tile_offset, ascending); break;
        case 5: _topk_xl_split_stride2_<K, 16>(tile_offset, ascending); break;
        case 6:
            _topk_xl_split_columns_<K, 0>(tile_offset, ascending);
            if constexpr (classic) {
                _topk_xl_split_separate_512_(tile_offset);
                if (chunk > 0) {
                    if (chunk & 1) {
                        _topk_xl_split_merge_unfused_512_<chunk_rows>(0);
                    } else {
                        _topk_xl_split_merge_unfused_512_<2 * chunk_rows>(0);
                    }
                }
            } else if (chunk > 0) {
                if constexpr (K == 512) {
                    if (num_chunks == 2) {
                        _topk_xl_split_merge_<K, 64, true>(0);
                        break;
                    }
                }
                switch ((num_chunks - 1 - chunk) & 3) {
                    case 1: _topk_xl_split_merge_<K, survivor_distance<K>(1), true>(tile_offset); break;
                    case 3: _topk_xl_split_merge_<K, survivor_distance<K>(3), true>(tile_offset); break;
                    default: _topk_xl_split_merge_<K, survivor_distance<K>(0), true>(tile_offset); break;
                }
            }
            break;
        case 7:
            if constexpr (classic) {
                _topk_xl_split_rebuild_build_unfused_512_(0, false);
            } else {
                _topk_xl_split_rebuild_build_<K>(survivor_tile<K, classic>(chunk, num_chunks) << 6, rebuild_dir);
            }
            break;
        default:
            if constexpr (classic) {
                _topk_xl_split_rebuild_columns_unfused_512_(0, false);
            } else {
                _topk_xl_split_columns_<K, 0>(survivor_tile<K, classic>(chunk, num_chunks) << 6, rebuild_dir);
            }
            break;
    }
}

template <std::uint32_t K, bool classic, bool mirror_last, std::uint32_t stage>
inline __attribute__((always_inline)) void sfpu_step(const std::uint32_t chunk, const std::uint32_t num_chunks) {
    take_math_token();
    sfpu_stage_body<K, classic>(chunk, stage, num_chunks, mirror_last && chunk + 1 == num_chunks);
    ckernel::t6_semaphore_post<ckernel::p_stall::WAIT_SFPU>(S2F);
}

// PACK's steps of chunk slot c1 in decode_step's order, straight-line: chunk c1's stages 0 to 4 between chunk c1 - 1's
// stages 5 to 8 (7 and 8 only for a merged chunk). pack_row takes slot 0's first token.
template <std::uint32_t K, bool classic, bool mirror_last>
inline __attribute__((noinline)) void pack_slot(const std::uint32_t c1, const std::uint32_t num_chunks) {
    const bool cur = c1 < num_chunks, prev = c1 > 0, prev_merged = c1 > 1;
    if (cur) {
        if (c1 > 0) {
            take_math_token();
        }
        sfpu_stage_body<K, classic>(c1, 0, num_chunks, mirror_last && c1 + 1 == num_chunks);
        ckernel::t6_semaphore_post<ckernel::p_stall::WAIT_SFPU>(S2F);
    }
    if (prev) {
        sfpu_step<K, classic, mirror_last, 5>(c1 - 1, num_chunks);
    }
    if (cur) {
        sfpu_step<K, classic, mirror_last, 1>(c1, num_chunks);
    }
    if (prev) {
        sfpu_step<K, classic, mirror_last, 6>(c1 - 1, num_chunks);
    }
    if (cur) {
        sfpu_step<K, classic, mirror_last, 2>(c1, num_chunks);
    }
    if (prev_merged) {
        sfpu_step<K, classic, mirror_last, 7>(c1 - 1, num_chunks);
    }
    if (cur) {
        sfpu_step<K, classic, mirror_last, 3>(c1, num_chunks);
    }
    if (prev_merged) {
        sfpu_step<K, classic, mirror_last, 8>(c1 - 1, num_chunks);
    }
    if (cur) {
        sfpu_step<K, classic, mirror_last, 4>(c1, num_chunks);
    }
}

// Every SFPU part of one row, survivor left in tile 0 (ascending if mirror_last), then epilogue(). PACK takes the SFPU
// over (its state and row_init()) after its first token, once MATH's last SFPU instruction has run.
template <std::uint32_t K, bool classic, bool mirror_last = false, typename RowInit, typename Epilogue>
inline void pack_row(const std::uint32_t num_chunks, RowInit&& row_init, Epilogue&& epilogue) {
    static_assert(
        supported_split<K, classic>, "the split takes fused K = 512, 1024 and 2048 rows and Classic K = 512 rows");
    take_math_token();
    ckernel::sfpu::_topk_xl_split_sfpu_init_<K, !classic>();
    row_init();
    TTI_STALLWAIT(ckernel::p_stall::STALL_SFPU, ckernel::p_stall::MATH);
    for (std::uint32_t c1 = 0; c1 <= num_chunks; c1++) {
        pack_slot<K, classic, mirror_last>(c1, num_chunks);
    }
    epilogue();
    ckernel::t6_semaphore_post<ckernel::p_stall::WAIT_SFPU>(S2F);
}

// A segmented row's later segment: after MATH's token (its unfused merge and rebuild done, or the previous segment's
// epilogue), move_acc() moves the accumulator out of Dst; MATH copies into tile 0 after the token that follows.
template <typename MoveAcc>
inline void hold_accumulator(MoveAcc&& move_acc) {
    take_math_token();
    move_acc();
    ckernel::t6_semaphore_post<ckernel::p_stall::PACK>(S2F);
}

#endif  // TRISC_PACK

#ifdef TRISC_MATH

// MATH's side of hold_accumulator.
inline void release_accumulator() {
    ckernel::t6_semaphore_post<ckernel::p_stall::MATH | ckernel::p_stall::WAIT_SFPU>(F2S);
    take_pack_token();
}

#endif  // TRISC_MATH

}
