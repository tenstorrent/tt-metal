// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/circular_buffer.h"
#include "topk_large_indices_compute_body_mode.hpp"
#include "topk_large_indices_metadata.hpp"
#include "topk_large_indices_runtime_args.hpp"
#ifdef ARCH_BLACKHOLE
#include "topk_large_indices_split.hpp"
#endif

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// The first fused chunk in each row runs the full TopK init. Later chunks can
// use this hot-path reinit because copy init only clobbers ADDR_MOD_0/3 and the
// MOP, while index stamping additionally changes ADDR_MOD_6 from +32 to +4.
// Fused sort/merge never consumes ADDR_MOD_0/3/4; ADDR_MOD_1/5 and the other
// TopK state remain live. Restoring +32 and the fused MOP therefore avoids the
// two redundant ADDR_MOD writes in every subsequent chunk's full init.
inline void _topk_large_indices_reinit_fused_after_stamp_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 32},
    }
        .set(ADDR_MOD_6);
    topk_mop_config<true>();
}

}  // namespace ckernel::sfpu
#endif

#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {

// The final TopK XL survivor stores FP32 value words followed by UINT32 index
// words. Mark indices paired with exact -inf values as invalid without
// disturbing the engine-specific survivor layout.
inline void _topk_large_indices_mark_neginf_indices_init_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_0);
}

template <uint32_t K>
inline void _topk_large_indices_mark_neginf_indices_() {
    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = K == 2048 ? 2 : 1;
    constexpr uint32_t indices_offset = tiles_per_sequence * 64;
    constexpr uint32_t iterations = (K == 512 ? 1 : K == 1024 ? 2 : 4) * 16;

    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, 0xFF80);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, 0xFFFF);

    for (uint32_t i = 0; i < iterations; ++i) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::INT32, ADDR_MOD_0, indices_offset);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

// A K = 2048 segment's unfused result from tiles 0 to 3 to tiles 4 to 7, raw words; ADDR_MOD_0 steps two rows.
inline void _topk_large_indices_move_segment_result_() {
    for (uint32_t i = 0; i < 128; ++i) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, 256);
    }
}

}  // namespace ckernel::sfpu
#endif

namespace {

constexpr uint32_t elements_per_tile = TILE_R_DIM * TILE_C_DIM;
// The input's rows span two or more chunks; a program of one-chunk rows compiles neither the split nor the short init.
constexpr bool multi_chunk_rows = get_compile_time_arg_val(8) != 0;

using ttnn::operations::experimental::topk_large_indices::program::ComputeBodyMode;
using ttnn::operations::experimental::topk_large_indices::program::max_fused_chunks;

constexpr uint32_t fused_chunk_id_mask = max_fused_chunks - 1;

// FirstOfRow: a row's first chunk configures the unpacker and math unit; later chunks find that intact and set up only the copy.
template <uint32_t K, bool FirstOfRow = true>
FORCE_INLINE void copy_chunk(CircularBuffer& input, uint32_t dst, uint32_t active_elements) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    const uint32_t input_cb = input.get_cb_id();

    input.wait_front(tiles_per_sequence);
    if constexpr (FirstOfRow || !multi_chunk_rows) {
        topk_xl_copy_tile_init(input_cb);
    } else {
        topk_xl_copy_tile_init_short(input_cb);
    }
    topk_xl_copy_tile<K>(input_cb, dst, 0, active_elements);
    input.pop_front(tiles_per_sequence);
}

// FullInit is compile-time and this helper is force-inlined so the hot loop has
// neither a mode branch nor a repeated full TopK configuration sequence.
template <uint32_t K, bool FullInit, bool Columns = false>
FORCE_INLINE void sort_fused_chunk(
    CircularBuffer& input, uint32_t dst, uint32_t active_elements, bool ascending, uint32_t local_chunk_id) {
    copy_chunk<K, FullInit>(input, dst, active_elements);

    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices_rt<K>(dst, local_chunk_id);
    if constexpr (FullInit) {
        topk_xl_init<K, true>();
    } else {
        MATH((ckernel::sfpu::_topk_large_indices_reinit_fused_after_stamp_()));
    }
    if constexpr (Columns) {
        topk_xl_local_sort_generic<K, true /*early_exit_K64*/>(dst, ascending);
    } else {
        topk_xl_local_sort<K>(dst, ascending);
    }
}

// Each body reduces its chunks into an unfused survivor at DST slot 0 with row-global indices.
template <uint32_t K, bool final_ascending>
FORCE_INLINE void reduce_fused_row(
    CircularBuffer& input, uint32_t first_chunk, uint32_t num_chunks, uint32_t tail_elements) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t survivor_slot = 0;
    // A fused survivor has no separate index tiles. Its merge operand is
    // therefore adjacent at one sequence, rather than two sequences, away.
    constexpr uint32_t incoming_slot = tiles_per_sequence;

    // The row has at most 32 chunks, so the row-global chunk id fits the stamp directly.
    sort_fused_chunk<K, true>(input, survivor_slot, num_chunks == 1 ? tail_elements : K, false, first_chunk);

    if (num_chunks == 1) {
        topk_xl_rebuild<K, true>(survivor_slot, final_ascending);
    } else {
        for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
            const uint32_t active_elements = chunk + 1 == num_chunks ? tail_elements : K;
            sort_fused_chunk<K, false>(input, incoming_slot, active_elements, true, first_chunk + chunk);
            topk_xl_merge<K, true>(survivor_slot);
            topk_xl_rebuild<K, true>(survivor_slot, final_ascending && chunk + 1 == num_chunks);
        }
    }

    // Decode the five-bit chunk stamp once, after the final fused survivor is
    // known, and leave the standard [values, indices] epilogue layout.
    topk_xl_separate_indices_row_major_global_init();
    topk_xl_separate_indices_row_major_global<K>(survivor_slot);
}

// Columns (k <= 64): the per-column top 64 survivors hold the segment's top 64; one full sort ranks them at the end.
template <uint32_t K, bool Columns>
FORCE_INLINE void reduce_segmented_row(
    CircularBuffer& input, uint32_t first_chunk, uint32_t num_chunks, uint32_t tail_elements, bool final_ascending) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t segment_capacity = max_fused_chunks;
    constexpr uint32_t accumulator_slot = 0;
    constexpr uint32_t segment_slot = 2 * tiles_per_sequence;

    const uint32_t end_chunk_total = first_chunk + num_chunks;
    // Segments follow the row's 32-chunk blocks, so the stamp is the chunk id within its block.
    uint32_t segment_first = first_chunk;
    for (uint32_t segment = 0; segment_first < end_chunk_total; ++segment) {
        const uint32_t block_end = (segment_first / segment_capacity + 1) * segment_capacity;
        const uint32_t end_chunk = block_end < end_chunk_total ? block_end : end_chunk_total;
        const uint32_t last_chunk = end_chunk - 1;
        const bool last_segment = end_chunk == end_chunk_total;

        // Segment zero is already in the final accumulator slot. Later
        // segments use the second unfused sequence. Their fused input chunk
        // is adjacent to the fused survivor at base + tiles_per_sequence.
        const uint32_t base = segment == 0 ? accumulator_slot : segment_slot;
        const uint32_t chunk_slot = base + tiles_per_sequence;
        const bool mirror_final_survivor = segment != 0 || (last_segment && final_ascending);

        const uint32_t first_elements = segment_first + 1 == end_chunk_total ? tail_elements : K;
        if (segment == 0) {
            sort_fused_chunk<K, true, Columns>(input, base, first_elements, false, segment_first & fused_chunk_id_mask);
        } else {
            sort_fused_chunk<K, false, Columns>(
                input, base, first_elements, false, segment_first & fused_chunk_id_mask);
        }

        if constexpr (Columns) {
            for (uint32_t chunk = segment_first + 1; chunk <= last_chunk; ++chunk) {
                const uint32_t active_elements = chunk + 1 == end_chunk_total ? tail_elements : K;
                sort_fused_chunk<K, false /*FullInit*/, true /*Columns*/>(
                    input, chunk_slot, active_elements, true /*ascending*/, chunk & fused_chunk_id_mask);
                topk_xl_merge<K, true>(base);
                if (chunk != last_chunk) {
                    topk_xl_rebuild_columns<K>(base, false /*ascending*/);
                }
            }
            topk_xl_local_sort<K>(base, mirror_final_survivor);
        } else if (segment_first == last_chunk) {
            topk_xl_rebuild<K, true>(base, mirror_final_survivor);
        } else {
            for (uint32_t chunk = segment_first + 1; chunk <= last_chunk; ++chunk) {
                const uint32_t active_elements = chunk + 1 == end_chunk_total ? tail_elements : K;
                sort_fused_chunk<K, false>(input, chunk_slot, active_elements, true, chunk & fused_chunk_id_mask);
                topk_xl_merge<K, true>(base);
                topk_xl_rebuild<K, true>(base, mirror_final_survivor && chunk == last_chunk);
            }
        }

        // Split once per segment. The base is a multiple of 32*K, so it does
        // not overlap the decoded segment-local index bits.
        topk_xl_separate_indices_row_major_global_init();
        topk_xl_separate_indices_row_major_global_base<K>(
            base, (segment_first / segment_capacity) * (segment_capacity * K));

        if (segment != 0) {
            // Operand zero is descending and each later segment was rebuilt
            // ascending, satisfying the unfused bitonic merge precondition.
            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(accumulator_slot);
            topk_xl_rebuild<K, false>(accumulator_slot, last_segment && final_ascending);
        }
        segment_first = end_chunk;
    }
}

// Tree merge round: the child's unfused survivor lands as raw FP32 tiles and merges into ours.
template <uint32_t K>
FORCE_INLINE void merge_landed_survivor(
    CircularBuffer& landing, uint32_t input_cb, uint32_t survivor_slot, bool final_ascending) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t survivor_tiles = 2 * tiles_per_sequence;
    const uint32_t landing_cb = landing.get_cb_id();
    const uint32_t incoming_slot = survivor_slot + survivor_tiles;

    landing.wait_front(survivor_tiles);
    reconfig_data_format_srca(landing_cb);
    copy_init(landing_cb);
    for (uint32_t tile = 0; tile < survivor_tiles; ++tile) {
        copy_tile(landing_cb, tile, incoming_slot + tile);
    }
    landing.pop_front(survivor_tiles);
    // The integer landing format leaves the FPU in int math; the rebuild transposes need the float path.
    reconfig_data_format_srca(input_cb);

    topk_xl_init<K, false>();
    topk_xl_merge<K, false>(survivor_slot);
    topk_xl_rebuild<K, false>(survivor_slot, final_ascending);
}

template <uint32_t K>
FORCE_INLINE void mark_neginf_indices(uint32_t dst) {
    MATH((ckernel::sfpu::_topk_large_indices_mark_neginf_indices_init_()));
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::_topk_large_indices_mark_neginf_indices_<K>, dst, VectorMode::RC_custom)));
}

template <uint32_t K>
FORCE_INLINE void materialize_index_rank_order(uint32_t dst, uint32_t indices_cb) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    transpose_dest_init<true, false>(indices_cb);
    for (uint32_t tile = 0; tile < tiles_per_sequence; ++tile) {
        transpose_dest<true, false>(dst + tiles_per_sequence + tile);
    }
}

#ifdef ARCH_BLACKHOLE
// A row of two or more chunks split across MATH and PACK (topk_large_indices_split.hpp); it ends with the -inf indices
// marked and the index tiles transposed, as mark_neginf_indices and materialize_index_rank_order leave a row.
template <uint32_t K>
FORCE_INLINE void reduce_split_row(
    CircularBuffer& input, uint32_t indices_cb, uint32_t num_chunks, uint32_t tail_elements) {
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    const uint32_t input_cb = input.get_cb_id();
#ifdef TRISC_UNPACK
    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        input.wait_front(tiles_per_sequence);
        if (chunk == 0) {
            topk_xl_copy_tile_init(input_cb);
        }
        topk_xl_copy_tile<K>(input_cb, 0, 0, chunk + 1 == num_chunks ? tail_elements : K);
        input.pop_front(tiles_per_sequence);
        llk_unpack_set_srcb_dummy_valid();
        if (chunk > 0) {
            llk_unpack_set_srcb_dummy_valid();
        }
    }
#endif
#ifdef TRISC_MATH
    _llk_math_eltwise_unary_sfpu_init_once_();
    topk_large_indices_split::math_row<K, false>(num_chunks, [&](uint32_t chunk, uint32_t tile) {
        if (chunk == 0) {
            topk_xl_copy_tile_init(input_cb);
        }
        topk_xl_copy_tile<K>(input_cb, tile, 0, chunk + 1 == num_chunks ? tail_elements : K);
    });
#endif
#ifdef TRISC_PACK
    topk_large_indices_split::pack_row<K, false>(
        num_chunks,
        [] {},
        [] {
            ckernel::sfpu::_topk_xl_separate_indices_row_major_global_init_();
            ckernel::sfpu::_topk_xl_split_begin_(0);
            ckernel::sfpu::_topk_xl_separate_indices_row_major_global_<K>();
            TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
            ckernel::sfpu::_topk_large_indices_mark_neginf_indices_init_();
            ckernel::sfpu::_topk_xl_split_begin_(0);
            ckernel::sfpu::_topk_large_indices_mark_neginf_indices_<K>();
            TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
        });
#endif
    transpose_dest_init<true, false>(indices_cb);
    for (uint32_t tile = 0; tile < tiles_per_sequence; ++tile) {
        transpose_dest<true, false>(tiles_per_sequence + tile);
    }
}

// A K = 2048 row of more than 32 chunks, its segments split as above. A later segment needs tiles 0 to 5, so the
// unfused accumulator (tiles 0 to 3) waits in acc_cb's L1 until its merge with the segment.
template <uint32_t K>
FORCE_INLINE void reduce_split_segmented_row(
    CircularBuffer& input, uint32_t acc_cb, uint32_t indices_cb, uint32_t num_chunks, uint32_t tail_elements) {
    static_assert(K == 2048, "the segmented body is K = 2048's");
    constexpr uint32_t tiles_per_sequence = 2;
    constexpr uint32_t segment_capacity = 32;
    constexpr uint32_t accumulator_slot = 0;
    constexpr uint32_t segment_slot = 2 * tiles_per_sequence;
    constexpr uint32_t chunk_slot = segment_slot + tiles_per_sequence;
    // A shorter last segment runs on one thread: the accumulator's two copies cost more than its split saves.
    constexpr uint32_t min_split_later_chunks = 3;
    const uint32_t input_cb = input.get_cb_id();
    const uint32_t num_segments = (num_chunks + segment_capacity - 1) / segment_capacity;
    CircularBuffer acc(acc_cb);
    for (uint32_t segment = 0; segment < num_segments; ++segment) {
        const uint32_t first_chunk = segment * segment_capacity;
        const uint32_t seg_chunks =
            num_chunks - first_chunk < segment_capacity ? num_chunks - first_chunk : segment_capacity;
        const uint32_t seg_tail = first_chunk + seg_chunks == num_chunks ? tail_elements : K;
        const bool later = segment != 0;
        const bool split = !later || seg_chunks >= min_split_later_chunks;

        if (!split) {
            sort_fused_chunk<K, true>(input, segment_slot, seg_chunks == 1 ? seg_tail : K, false, 0);
            if (seg_chunks == 1) {
                topk_xl_rebuild<K, true>(segment_slot, true);
            }
            for (uint32_t chunk = 1; chunk < seg_chunks; ++chunk) {
                sort_fused_chunk<K, false>(input, chunk_slot, chunk + 1 == seg_chunks ? seg_tail : K, true, chunk);
                topk_xl_merge<K, true>(segment_slot);
                topk_xl_rebuild<K, true>(segment_slot, chunk + 1 == seg_chunks);
            }
        } else {
            if (later) {
#ifdef TRISC_MATH
                topk_large_indices_split::release_accumulator();
#endif
#ifdef TRISC_PACK
                topk_large_indices_split::hold_accumulator([&] {
                    // Segment 1 finds the packer in its untilize mode, a later segment in pack_tile mode with the MOP
                    // the pack thread's SFPU work reprogrammed.
                    if (segment == 1) {
                        pack_untilize_uninit(acc_cb);
                    } else {
                        pack_init(acc_cb);
                    }
                    acc.reserve_back(2 * tiles_per_sequence);
                    for (uint32_t tile = 0; tile < 2 * tiles_per_sequence; ++tile) {
                        pack_tile<true>(accumulator_slot + tile, acc_cb, tile);
                    }
                    acc.push_back(2 * tiles_per_sequence);
                });
#endif
            }
#ifdef TRISC_UNPACK
            for (uint32_t chunk = 0; chunk < seg_chunks; ++chunk) {
                input.wait_front(tiles_per_sequence);
                if (chunk == 0) {
                    topk_xl_copy_tile_init(input_cb);
                }
                topk_xl_copy_tile<K>(input_cb, 0, 0, chunk + 1 == seg_chunks ? seg_tail : K);
                input.pop_front(tiles_per_sequence);
                llk_unpack_set_srcb_dummy_valid();
                if (chunk > 0) {
                    llk_unpack_set_srcb_dummy_valid();
                }
            }
#endif
#ifdef TRISC_MATH
            _llk_math_eltwise_unary_sfpu_init_once_();
            topk_large_indices_split::math_row<K, false>(seg_chunks, [&](uint32_t chunk, uint32_t tile) {
                if (chunk == 0) {
                    topk_xl_copy_tile_init(input_cb);
                }
                topk_xl_copy_tile<K>(input_cb, tile, 0, chunk + 1 == seg_chunks ? seg_tail : K);
            });
#endif
#ifdef TRISC_PACK
            auto epilogue = [segment, later] {
                ckernel::sfpu::_topk_xl_separate_indices_row_major_global_init_();
                ckernel::sfpu::_topk_xl_split_begin_(accumulator_slot);
                ckernel::sfpu::_topk_xl_separate_indices_row_major_global_base_<K>(segment * (segment_capacity * K));
                TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
                if (later) {
                    ckernel::sfpu::_topk_xl_split_begin_(accumulator_slot);
                    ckernel::sfpu::_topk_large_indices_move_segment_result_();
                    TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
                }
            };
            if (later) {
                topk_large_indices_split::pack_row<K, false, true>(seg_chunks, [] {}, epilogue);
            } else {
                topk_large_indices_split::pack_row<K, false>(seg_chunks, [] {}, epilogue);
            }
#endif
        }

        if (later) {
            if (split) {
                // copy_init sets no formats: the unpack to Dst needs acc_cb's, the merge and transposes the input's.
                reconfig_data_format_srca(input_cb, acc_cb);
                copy_init(acc_cb);
                acc.wait_front(2 * tiles_per_sequence);
                for (uint32_t tile = 0; tile < 2 * tiles_per_sequence; ++tile) {
                    copy_tile(acc_cb, tile, accumulator_slot + tile);
                }
                acc.pop_front(2 * tiles_per_sequence);
                reconfig_data_format_srca(acc_cb, input_cb);
                // The next copy init configures the unpacker from the RISC, after the last unpack to Dst has finished.
                UNPACK((ckernel::tensix_sync()));
            }
            if (!split) {
                topk_xl_separate_indices_row_major_global_init();
                topk_xl_separate_indices_row_major_global_base<K>(segment_slot, segment * (segment_capacity * K));
            }
            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(accumulator_slot);
            topk_xl_rebuild<K, false>(accumulator_slot, false);
        }
    }
    mark_neginf_indices<K>(accumulator_slot);
    materialize_index_rank_order<K>(accumulator_slot, indices_cb);
}
#endif

}  // namespace

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(topk_core_args::compute_num_rows);
    constexpr bool tree = get_named_compile_time_arg_val("tree") != 0;
    constexpr bool split_whole_rows = get_named_compile_time_arg_val("split_whole_rows") != 0;
    const uint32_t seg_first_chunk = tree ? get_arg_val<uint32_t>(topk_core_args::compute_seg_first_chunk) : 0;
    const uint32_t seg_end_chunk = tree ? get_arg_val<uint32_t>(topk_core_args::compute_seg_end_chunk) : 0;
    const uint32_t num_recv_rounds = tree ? get_arg_val<uint32_t>(topk_core_args::compute_num_recv_rounds) : 0;
    const bool sends_survivor = tree && get_arg_val<uint32_t>(topk_core_args::compute_sends_survivor) != 0;
    const bool has_split = tree && get_arg_val<uint32_t>(topk_core_args::compute_has_split) != 0;
    // Without a split row the segment args already describe the whole rows.
    const uint32_t whole_first_chunk = has_split ? 0 : seg_first_chunk;
    const uint32_t whole_end_chunk = has_split ? UINT32_MAX : seg_end_chunk;
    const uint32_t search_len = get_common_arg_val<uint32_t>(topk_common_args::compute_search_length);

    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t indices_cb = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr auto body_mode = static_cast<ComputeBodyMode>(get_compile_time_arg_val(3));
    constexpr bool valid_length_from_metadata = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t meta_cb = get_compile_time_arg_val(5);
    constexpr uint32_t landing_cb = get_compile_time_arg_val(6);
    constexpr uint32_t send_cb = get_compile_time_arg_val(7);
    constexpr uint32_t acc_cb = get_compile_time_arg_val(9);

    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    static_assert(
        body_mode == ComputeBodyMode::FusedEndToEnd || body_mode == ComputeBodyMode::FusedSegmented ||
            body_mode == ComputeBodyMode::ColumnSegmented,
        "invalid TopK compute body mode");
    constexpr bool columns = body_mode == ComputeBodyMode::ColumnSegmented;
    static_assert(!columns || K == 1024, "the column body runs on K 1024 chunks");

    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t survivor_tiles = 2 * tiles_per_sequence;
    constexpr uint32_t final_survivor = 0;

    compute_kernel_hw_startup(input_cb, indices_cb);
    pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(indices_cb);

    uint32_t num_chunks;
    uint32_t tail_elements;
    uint32_t split_num_chunks;
    uint32_t split_tail_elements;
    if constexpr (valid_length_from_metadata) {
        // The reader already clipped the rows and the segment against the metadata length.
        CircularBuffer meta_cb_obj(meta_cb);
        meta_cb_obj.wait_front(1);
        num_chunks = ckernel::read_tile_value(meta_cb, 0, topk_metadata_num_chunks_word);
        tail_elements = ckernel::read_tile_value(meta_cb, 0, topk_metadata_tail_elements_word);
        if constexpr (tree) {
            split_num_chunks =
                ckernel::read_tile_value(meta_cb, 0, topk_metadata_split_word_offset + topk_metadata_num_chunks_word);
            split_tail_elements = ckernel::read_tile_value(
                meta_cb, 0, topk_metadata_split_word_offset + topk_metadata_tail_elements_word);
        } else {
            split_num_chunks = num_chunks;
            split_tail_elements = tail_elements;
        }
        meta_cb_obj.pop_front(1);
    } else if constexpr (!tree) {
        const auto bounds = calculate_topk_bounds(search_len, K);
        num_chunks = bounds.num_chunks;
        tail_elements = bounds.tail_elements;
        split_num_chunks = num_chunks;
        split_tail_elements = tail_elements;
    } else {
        const auto row = calculate_topk_bounds(search_len, K);
        const auto whole = calculate_topk_segment_bounds(row, K, whole_first_chunk, whole_end_chunk);
        const auto split = calculate_topk_segment_bounds(row, K, seg_first_chunk, seg_end_chunk);
        num_chunks = whole.num_chunks;
        tail_elements = whole.tail_elements;
        split_num_chunks = split.num_chunks;
        split_tail_elements = split.tail_elements;
    }

    CircularBuffer input(input_cb);
    CircularBuffer indices(indices_cb);
    CircularBuffer landing(landing_cb);
    CircularBuffer send(send_cb);

    const uint32_t num_whole_rows = has_split ? num_rows - 1 : num_rows;
    uint32_t first_row = 0;
#ifdef ARCH_BLACKHOLE
    // Whole rows of 2 to 32 chunks (3 at K 1024, and at K 512 without a split row in the program), and K = 2048 rows
    // longer than 32 with the accumulator buffer, split across MATH and PACK; a core's split row then runs on one
    // thread.
    if constexpr (false && split_whole_rows && !columns) {
        constexpr bool split_segments = K == 2048 && body_mode == ComputeBodyMode::FusedSegmented && acc_cb != 0;
        constexpr uint32_t min_split_chunks = (K == 1024 || (K == 512 && !tree)) ? 3 : 2;
        if (num_whole_rows > 0 && num_chunks >= min_split_chunks &&
            (num_chunks <= max_fused_chunks || split_segments)) {
            MATH((topk_large_indices_split::start()));
            PACK((topk_large_indices_split::wait_start()));
            for (uint32_t row = 0; row < num_whole_rows; ++row) {
                tile_regs_acquire();
                if constexpr (split_segments) {
                    if (num_chunks > max_fused_chunks) {
                        reduce_split_segmented_row<K>(input, acc_cb, indices_cb, num_chunks, tail_elements);
                    } else {
                        reduce_split_row<K>(input, indices_cb, num_chunks, tail_elements);
                    }
                } else {
                    reduce_split_row<K>(input, indices_cb, num_chunks, tail_elements);
                }
                tile_regs_commit();
                tile_regs_wait();

                // The pack thread's SFPU work reprograms its ADDR_MODs and MOP.
                pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(indices_cb);
                indices.reserve_back(1);
                pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(
                    indices_cb, 1, 0, final_survivor + tiles_per_sequence);
                indices.push_back(1);

                tile_regs_release();
            }
            if (!has_split) {
                return;
            }
            first_row = num_whole_rows;
        }
    }
#endif

    for (uint32_t row = first_row; row < num_rows; ++row) {
        const bool split_row = has_split && row == num_whole_rows;
        const uint32_t first_chunk = split_row ? seg_first_chunk : whole_first_chunk;
        const uint32_t row_chunks = split_row ? split_num_chunks : num_chunks;
        const uint32_t row_tail = split_row ? split_tail_elements : tail_elements;
        const uint32_t row_recv_rounds = split_row ? num_recv_rounds : 0;
        const bool row_sends = split_row && sends_survivor;
        // A survivor that leaves this core is merge operand one of its parent and must end ascending.
        const bool body_final_ascending = row_sends && row_recv_rounds == 0;

        tile_regs_acquire();

        if constexpr (body_mode == ComputeBodyMode::FusedEndToEnd) {
            // The rebuild assembles its instructions from the direction, so each direction gets its own body.
            if (body_final_ascending) {
                reduce_fused_row<K, true>(input, first_chunk, row_chunks, row_tail);
            } else {
                reduce_fused_row<K, false>(input, first_chunk, row_chunks, row_tail);
            }
        } else {
            reduce_segmented_row<K, columns>(input, first_chunk, row_chunks, row_tail, body_final_ascending);
        }

        for (uint32_t round = 0; round < row_recv_rounds; ++round) {
            merge_landed_survivor<K>(landing, input_cb, final_survivor, row_sends && round + 1 == row_recv_rounds);
        }

        if (row_sends) {
            tile_regs_commit();
            tile_regs_wait();

            // Raw tile pack so the parent can unpack the survivor straight back into DST.
            send.reserve_back(survivor_tiles);
            pack_untilize_uninit(send_cb);
            pack_reconfig_data_format(send_cb);
            for (uint32_t tile = 0; tile < survivor_tiles; ++tile) {
                pack_tile(final_survivor + tile, send_cb);
            }
            send.push_back(survivor_tiles);
        } else {
            mark_neginf_indices<K>(final_survivor);
            materialize_index_rank_order<K>(final_survivor, indices_cb);

            tile_regs_commit();
            tile_regs_wait();

            indices.reserve_back(1);
            pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(
                indices_cb, 1, 0, final_survivor + tiles_per_sequence);
            indices.push_back(1);
        }

        tile_regs_release();

        if (row_sends) {
            // Re-arm the untilize pack the row output path expects.
            pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(indices_cb);
        }
    }
    if (num_recv_rounds > 0 || sends_survivor) {
        // Workaround: one discarded datacopy here keeps the next program's first tile on this core intact.
        reconfig_data_format_srca(input_cb);
        copy_init(input_cb);
        tile_regs_acquire();
        copy_tile(input_cb, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
    }
}
