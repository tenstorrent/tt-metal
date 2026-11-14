// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sdpa_decode_lse writer: a copy of the stock
// ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp
// (tt-metal 298c06426a8) with these edits, each marked [sdpa_decode_lse]:
//   1. the dataflow_common.hpp include names the stock header by its path;
//   2. runtime arg 1 is the stats tensor's address (the stock args follow, shifted by one) and the stats
//      tensor's TensorAccessorArgs follow the output's in the compile-time args;
//   3. the tree root, after writing O, drains the (m, l) tiles the compute kernel copied into cb_out_m / cb_out_l
//      (unused on the root by the stock kernels) into the stats tensor;
//   4. under SDPA_LSE_PAIR_SPLIT (the op's pair_split mode only; the half index is the compile-time arg after the
//      stats TensorAccessorArgs) the K-chunk range is this device's half of the row (pair_split.hpp), a child "has
//      data" when its member index is below the half's chunk count, and the root of an empty half (no chunk of the
//      row on this device) writes the sentinel O = 0, m = -30000, l = 1 instead of returning.
// The op is MQA only (one KV head per chip, interleaved output), asserted below.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/debug/assert.h"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
// [sdpa_decode_lse] the stock header by path (was "dataflow_common.hpp")
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#if defined(SDPA_LSE_PAIR_SPLIT) || defined(SDPA_LSE_OWNED)
#include "pair_split.hpp"  // [sdpa_decode_lse] pair split / owned mode: the empty-half sentinel values
#endif

#define MAX_TREE_REDUCTION_ROUNDS 6

void kernel_main() {
    Noc noc;

    constexpr uint32_t PNHt = get_compile_time_arg_val(0);        // padded number of heads in tiles
    constexpr uint32_t St = get_compile_time_arg_val(1);          // full sequence length of kv cache in tiles
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);        // head dim for V
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(3);  // number of tiles in seqlen of a k/v/mask chunk
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(4);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(5);           // num cores per batch
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);                     // num running cores in total
    constexpr uint32_t reducer_semaphore_id = get_compile_time_arg_val(7);          // semaphore ID for reducer
    uint32_t reducer_semaphore_addr = get_semaphore(reducer_semaphore_id);          // semaphore for reducer
    constexpr uint32_t output_semaphore_id = get_compile_time_arg_val(8);           // semaphore ID for sender
    constexpr bool is_out_sharded = get_compile_time_arg_val(9);
    constexpr uint32_t num_q_heads = get_compile_time_arg_val(10);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(11);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(12);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(13);
    constexpr uint32_t num_reducer_cores = get_compile_time_arg_val(14);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(15);
    constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(16);
    constexpr bool is_causal = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(18);
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(19);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(20);
    constexpr uint32_t original_block_size = get_compile_time_arg_val(21);
    constexpr bool has_block_padding = original_block_size > 0 && original_block_size < 32;
    // Speculative multi-position mode: Tg candidates on each batch row, PNHt == Tg, one causal
    // bound per row-tile taken from this row's group of Tg entries in the [B*Tg] cur_pos
    // vector (group base = cur_batch*Tg). 0 = off.
    constexpr uint32_t spec_multi_pos_T = get_compile_time_arg_val(22);
    constexpr bool spec_multi_pos = spec_multi_pos_T > 0;

    constexpr auto out_args = TensorAccessorArgs<23>();
    // [sdpa_decode_lse] the stats tensor [1, B, PNHt * 32, 64]: page (b * PNHt + t) * 2 holds row tile t's m column
    // tile, page (b * PNHt + t) * 2 + 1 its l column tile.
    constexpr auto stats_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    static_assert(num_kv_heads == 1, "sdpa_decode_lse: one KV head per chip (the MQA output path)");
    static_assert(!is_out_sharded, "sdpa_decode_lse: interleaved output only");
    static_assert(!spec_multi_pos, "sdpa_decode_lse: no speculative multi-position mode");
#if defined(SDPA_LSE_PAIR_SPLIT) || defined(SDPA_LSE_HALF_BLOCK)
    // [sdpa_decode_lse] pair split / half-block mode: this device's pair half (its mesh column's parity)
    constexpr uint32_t pair_half = get_compile_time_arg_val(stats_args.next_compile_time_args_offset());
    static_assert(sliding_window_size == 0, "sdpa_decode_lse pair split: no sliding window");
#endif

    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;
    // #44366: c_8 is the writer-only cur_pos CB after the split.
    // Compute reads from c_15 (see sdpa_flash_decode.cpp), so the writer
    // owns c_8 alone and can always pop it.
    constexpr uint32_t cb_cur_pos = tt::CBIndex::c_8;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_11;
    constexpr uint32_t cb_zero_in = tt::CBIndex::c_12;
    constexpr uint32_t cb_sliding_window_mask_in = tt::CBIndex::c_13;  // Separate buffer for sliding window mask
    constexpr uint32_t cb_block_pad_mask = tt::CBIndex::c_14;          // Block padding mask (block_size < TILE_HEIGHT)
    constexpr uint32_t cb_out_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_worker = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;
    constexpr uint32_t cb_intermed_out =
        tt::CBIndex::c_19;  // this cb holds the output intermediates from other worker cores
    constexpr uint32_t cb_out = tt::CBIndex::c_20;

    uint32_t arg_idx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stats_addr = get_arg_val<uint32_t>(arg_idx++);  // [sdpa_decode_lse]
    const bool do_output = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);

    // Tree reduction parameters
    const bool is_tree_root = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t parent_core_in_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t send_at_round = get_arg_val<uint32_t>(arg_idx++);

    // Semaphore encoding: each round uses a 4-bit field (nibble) in the semaphore value
    // Round 0: bits 0-3, Round 1: bits 4-7, Round 2: bits 8-11, etc.
    // step_semaphore_inc[r] = 1 << (r * 4) is the value to add to increment round r's counter
    constexpr uint32_t step_semaphore_inc[6] = {1, 16, 256, 4096, 65536, 1048576};
    // step_semaphore_shift[r] = r * 4 is the bit position to read round r's counter
    constexpr uint32_t step_semaphore_shift[6] = {0, 4, 8, 12, 16, 20};

    // Read children_per_round array

    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        children_per_round[r] = get_arg_val<uint32_t>(arg_idx++);
    }

    // Read reduction group physical core coordinates
    tt_l1_ptr uint32_t* reduction_group_core_xs = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores_per_head;
    tt_l1_ptr uint32_t* reduction_group_core_ys = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores_per_head;

    // idle core
    if (out_addr == 0) {
        return;
    }
    // Get cur_pos
    constexpr uint32_t cur_pos_base = St * 32 - 1;
    uint32_t cur_pos = cur_pos_base;  // default to non-causal, which we do attention on the entire kv cache. In this
                                      // case we set cur_pos to the last position
    // Spec mode: a private copy of THIS batch row's group of Tg bounds, taken before the CB is
    // popped, so the per-row-tile mask bounds stay readable for the rest of the kernel.
    constexpr uint32_t spec_pos_n = spec_multi_pos ? spec_multi_pos_T : 1;
    uint32_t spec_pos[spec_pos_n];
    // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list
    if constexpr (is_causal) {
        if (cur_pos_arg != UINT32_MAX) {
            cur_pos = cur_pos_arg;
        } else {
            CircularBuffer cb_index(cb_cur_pos);
            cb_index.wait_front(1);
            uint32_t index_cb_ptr = cb_index.get_read_ptr();
            volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_ptr);
            if constexpr (spec_multi_pos) {
                // cur_pos holds B*Tg bounds; this batch row owns the group starting at
                // cur_batch*Tg. Row-tile j of the group takes bound spec_pos[j].
                const uint32_t spec_pos_base = cur_batch * spec_multi_pos_T;
                for (uint32_t j = 0; j < spec_multi_pos_T; ++j) {
                    spec_pos[j] = index_ptr[spec_pos_base + j];
                }
#if ASSERT_ENABLED
                // The reader derives the group's KV scan range from the group's LAST entry and skips
                // k-chunks that end before its FIRST, so a non-ascending group silently drops KV; this
                // is the on-device half of a contract validate cannot see, because in spec mode the
                // positions live in cur_pos_tensor and are only ever read here, on device.
                for (uint32_t j = 0; j + 1 < spec_multi_pos_T; ++j) {
                    ASSERT(spec_pos[j] <= spec_pos[j + 1]);
                }
#endif
                // Positions are ascending within a group, so its last entry sets the KV scan range.
                cur_pos = spec_pos[spec_multi_pos_T - 1];
            } else {
                cur_pos = index_ptr[(uint32_t)(cur_batch / q_heads_parallel_factor)];
            }
            cb_index.pop_front(1);
        }

#ifdef SDPA_LSE_HALF_BLOCK
        // [sdpa_decode_lse] half-block mode (pair-sharded global KV): this device holds chunk half pair_half of
        // every logical block (chunks of SDPA_LSE_HALF_BLOCK tokens), entry i of the page table being its half of
        // the row's block i; its keys are the chunks c with c % 2 == pair_half, attended as one compacted sequence
        // whose last index is the effective cur_pos (-1 when this device owns none of the row's keys).
        if (cur_pos != UINT32_MAX) {
            constexpr uint32_t hb = SDPA_LSE_HALF_BLOCK;
            const uint32_t n = cur_pos / hb;
            const uint32_t len = ((n + 1 - pair_half) / 2) * hb + ((n % 2) == pair_half ? cur_pos % hb + 1 : 0);
            cur_pos = len == 0 ? UINT32_MAX : len - 1;
        }
#endif
        if (cur_pos == UINT32_MAX) {
            // cur_pos of -1 indicates that the user should be skipped
#ifdef SDPA_LSE_OWNED
            // [sdpa_decode_lse] owned mode (pair-sharded global KV): -1 also means this chip owns none of the
            // row's blocks, so the row's root writes the empty-half sentinel the pair merge weighs to 0 (an
            // inactive row's outputs are unused, so writing it there too is harmless). The reader and the
            // compute returned before touching any CB; c_20 is free scratch on this core.
            if (is_tree_root) {
                constexpr uint32_t empty_out_tiles = PNHt * vDHt;
                constexpr uint32_t fill_bytes = get_tile_size(cb_out);
                static_assert(empty_out_tiles >= 3, "sdpa_decode_lse owned mode: c_20 holds the 3 sentinel tiles");
                CircularBuffer cb_fill(cb_out);
                const uint32_t zero_addr = cb_fill.get_write_ptr();
                const uint32_t m_addr = zero_addr + fill_bytes;
                const uint32_t l_addr = m_addr + fill_bytes;
                volatile tt_l1_ptr uint32_t* zero_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_addr);
                volatile tt_l1_ptr uint32_t* m_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m_addr);
                volatile tt_l1_ptr uint32_t* l_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_addr);
                for (uint32_t i = 0; i < fill_bytes / sizeof(uint32_t); ++i) {
                    zero_ptr[i] = 0;
                    m_ptr[i] = PAIR_EMPTY_M_PACKED;
                    l_ptr[i] = PAIR_EMPTY_L_PACKED;
                }
                const auto empty_out_writer = TensorAccessor(out_args, out_addr);
                const auto empty_stats_writer = TensorAccessor(stats_args, stats_addr);
                noc.async_write_barrier();
                for (uint32_t t = 0; t < empty_out_tiles; ++t) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(zero_addr),
                        empty_out_writer,
                        fill_bytes,
                        {},
                        {.page_id = cur_batch * empty_out_tiles + t});
                }
                for (uint32_t t = 0; t < PNHt; ++t) {
                    const uint32_t page = (cur_batch * PNHt + t) * 2;
                    noc.async_write(
                        CoreLocalMem<uint32_t>(m_addr), empty_stats_writer, fill_bytes, {}, {.page_id = page});
                    noc.async_write(
                        CoreLocalMem<uint32_t>(l_addr), empty_stats_writer, fill_bytes, {}, {.page_id = page + 1});
                }
                noc.async_write_barrier();
            }
#endif
            return;
        }
    }

    // When block_size < TILE_HEIGHT, convert cur_pos to padded tile space.
    // Only for causal mode; non-causal cur_pos is already in padded space.
    if constexpr (has_block_padding && is_causal) {
        cur_pos = (cur_pos / original_block_size) * 32 + (cur_pos % original_block_size);
    }

    auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos);
    auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;

    // Sequence length assignment
#ifdef SDPA_LSE_PAIR_SPLIT
    // [sdpa_decode_lse] pair split: this core's chunks within this device's half of the row
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk, half_num_chunks] =
        get_pair_workload_for_core<pair_half>(cur_pos, core_num_in_reduce, num_cores_per_head, k_chunk_size_dynamic);
#else
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        get_workload_for_core(
            cur_pos,
            core_num_in_reduce,
            num_cores_per_head,
            k_chunk_size_dynamic,
            sliding_window_size > 0 ? std::optional<uint32_t>(sliding_window_size) : std::nullopt);
#endif

    // Check if this core has local data to process
    bool has_local_data = (k_chunk_start != k_chunk_end);

    // Cores without data don't participate in tree reduction at all
    if (!has_local_data) {
#ifdef SDPA_LSE_PAIR_SPLIT
        // [sdpa_decode_lse] pair split, an empty half: member 0 takes chunks whenever the half has any, so a root
        // without data means no core of this row's group has data on this device -- no partial will arrive and the
        // compute and reader kernels of this core already returned (after the reader pushed, and this kernel
        // popped, the cur_pos stick). The root writes the finite sentinel the pair merge weighs to 0: O = 0 over the
        // row's output tiles, and m = -30000 (bf16 -29952) / l = 1 over its m / l stats tiles, every element of each
        // full tile (page). Scratch: this core's c_20 (the root's output CB, PNHt * vDHt >= 8 full tiles), unused
        // by anything else here.
        if (is_tree_root) {
            constexpr uint32_t empty_out_tiles = PNHt * vDHt;
            constexpr uint32_t fill_bytes = get_tile_size(cb_out);  // one full bf16 tile = one page of out / stats
            static_assert(empty_out_tiles >= 3, "sdpa_decode_lse pair split: c_20 holds the 3 sentinel tiles");
            CircularBuffer cb_fill(cb_out);
            const uint32_t zero_addr = cb_fill.get_write_ptr();
            const uint32_t m_addr = zero_addr + fill_bytes;
            const uint32_t l_addr = m_addr + fill_bytes;
            volatile tt_l1_ptr uint32_t* zero_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_addr);
            volatile tt_l1_ptr uint32_t* m_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m_addr);
            volatile tt_l1_ptr uint32_t* l_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_addr);
            for (uint32_t i = 0; i < fill_bytes / sizeof(uint32_t); ++i) {
                zero_ptr[i] = 0;
                m_ptr[i] = PAIR_EMPTY_M_PACKED;
                l_ptr[i] = PAIR_EMPTY_L_PACKED;
            }
            const auto empty_out_writer = TensorAccessor(out_args, out_addr);
            const auto empty_stats_writer = TensorAccessor(stats_args, stats_addr);
            noc.async_write_barrier();  // #19201 BH hang workaround, as the stock path before its first writes
            for (uint32_t t = 0; t < empty_out_tiles; ++t) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(zero_addr),
                    empty_out_writer,
                    fill_bytes,
                    {},
                    {.page_id = cur_batch * empty_out_tiles + t});
            }
            for (uint32_t t = 0; t < PNHt; ++t) {
                const uint32_t page = (cur_batch * PNHt + t) * 2;
                noc.async_write(CoreLocalMem<uint32_t>(m_addr), empty_stats_writer, fill_bytes, {}, {.page_id = page});
                noc.async_write(
                    CoreLocalMem<uint32_t>(l_addr), empty_stats_writer, fill_bytes, {}, {.page_id = page + 1});
            }
            noc.async_write_barrier();
        }
#endif
        return;
    }

    // Determine which children actually participate in reduction (based on chunk allocation)
    // A child at core_num has data if core_num < k_num_chunks
    uint32_t num_active_children = 0;
    uint32_t active_children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    uint32_t num_active_rounds = 0;

    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        uint32_t child_id = children_per_round[r];
#ifdef SDPA_LSE_PAIR_SPLIT
        // [sdpa_decode_lse] pair split: member c of this device has data iff c < the half's chunk count
        if (child_id != UINT32_MAX && child_id < half_num_chunks) {
#else
        if (child_id != UINT32_MAX && child_id < k_num_chunks) {
#endif
            // This child has data
            active_children_per_round[r] = child_id;
            num_active_children++;
            num_active_rounds = r + 1;
        } else {
            active_children_per_round[r] = UINT32_MAX;
        }
    }

    // We send if we have a parent (and we have data, which we do if we reach here)
    const bool should_send_to_parent = (parent_core_in_group != UINT32_MAX);

    tt_l1_ptr uint32_t* all_reducer_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_reducer_cores;
    tt_l1_ptr uint32_t* all_reducer_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_reducer_cores;
    tt_l1_ptr uint32_t* all_output_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_output_cores;
    tt_l1_ptr uint32_t* all_output_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx++));

    constexpr uint32_t out_chunk_tiles = PNHt * vDHt;

    // generate and send scaler to compute
    // These helper functions respect tile size of CBs (ie. no need for special handling of tiny tiles)
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    dataflow_kernel_lib::prepare_zero_tile<cb_zero_in>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);

    // Generate sliding window mask only if we have local data and need it
    if (k_chunk_start == window_start_chunk && window_start_unaligned > 0) {
        // If this core processes the first chunk and we need to apply sliding window mask, generate it here
        generate_sliding_window_mask<cb_sliding_window_mask_in, PNHt>(Sk_chunk_t_dynamic, window_start_unaligned);
    }

    // Generate block padding mask: rows [0, block_size) = 0, rows [block_size, 32) = -inf.
    // Pushed once and reused (not popped) by compute for every K chunk.
    if constexpr (has_block_padding) {
        generate_block_padding_mask<cb_block_pad_mask, PNHt>(Sk_chunk_t_dynamic, original_block_size);
    }

    // generate and send mask to compute if causal (only if we have local data to process)
    if constexpr (spec_multi_pos) {
        // Spec mode: this row's Tg candidate bounds span <= Tg-1 consecutive positions while a
        // chunk is >= 32 positions wide, so at most the top two chunks of the group's scan
        // carry any masking. A chunk needs a mask iff its LAST covered position is past the
        // group's smallest bound; every other chunk is fully visible to all of the group's
        // candidates. The predicate is per-chunk (not "the last chunk of the reducer core"),
        // because with the reversed chunk distribution the two masked chunks can land on two
        // different cores. The compute kernel evaluates the identical predicate and pops one
        // block per masked chunk, so pushes and pops stay in lockstep through the
        // single-buffered mask CB.
        const uint32_t spec_pos_min = spec_pos[0];
        for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
            const uint32_t chunk_start_pos = k_chunk * k_chunk_size_dynamic;
            const uint32_t chunk_last_pos = chunk_start_pos + k_chunk_size_dynamic - 1;
            if (chunk_last_pos > spec_pos_min) {
                generate_mask_spec_multi_pos<cb_mask_in, PNHt>(Sk_chunk_t_dynamic, chunk_start_pos, spec_pos);
            }
        }
    } else if constexpr (is_causal) {
        // These helper functions respect tile size of CBs (ie. no need for special handling of tiny tiles)
        generate_mask<cb_mask_in, PNHt>(k_num_chunks, Sk_chunk_t_dynamic, cur_pos);
    }

    // *** Tree Reduction Logic ***
    // Each core participates in tree reduction by:
    // 1. Computing its local attention (done in compute kernel)
    // 2. For each round it receives: wait for child, receive data, combine with local (done in compute kernel)
    // 3. If not root: send combined result to parent
    // 4. If root: write final output

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);

    const auto out_writer = TensorAccessor(out_args, out_addr);

    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_semaphore_addr);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    noc.async_write_barrier();  // #19201 BH hang workaround

    for (uint32_t cur_head = cur_head_group * num_heads_per_core;
         cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
         ++cur_head) {
        // Tree reduction: receive from children at each round
        // Each round, we wait for one child (if any), read remote_sum, remote_max, remote_output, and push to CBs
        // The compute kernel processes each child's data before we move to the next round
        // Only receive from children that actually have data
        if (num_active_children > 0) {
            // If there are workers, then head must be split across workers
            ASSERT(num_heads_per_core == 1);

            for (uint32_t round = 0; round < num_active_rounds; ++round) {
                uint32_t child_id = active_children_per_round[round];
                if (child_id != UINT32_MAX) {
                    // Wait for this specific child to send its results
                    // Poll until round-specific nibble is >= 1
                    // Each round uses a 4-bit field: round 0 = bits 0-3, round 1 = bits 4-7, etc.
                    while (true) {
                        invalidate_l1_cache();
                        uint32_t sem_val = *in0_receiver_semaphore_addr_ptr;
                        uint8_t step_sem = (sem_val >> step_semaphore_shift[round]) & 0x0F;
                        if (step_sem >= 1) {
                            break;
                        }
                    }

                    // Now read the data from the intermediate buffer
                    constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);
                    constexpr uint32_t o_read_size = out_chunk_tiles * tile_bytes_intermed;
                    constexpr uint32_t ml_read_size = PNHt * tile_bytes_intermed;

                    // Calculate offset based on round (child writes at round offset)
                    uint32_t block_offset = round * (out_chunk_tiles + 2 * PNHt) * tile_bytes_intermed;
                    CircularBuffer cb_intermed(cb_intermed_out);
                    const uint8_t noc_id = noc.get_noc_id();
                    const uint32_t my_noc_x = my_x[noc_id];
                    const uint32_t my_noc_y = my_y[noc_id];
                    uint32_t intermed_l1_read_addr = cb_intermed.get_read_ptr() + block_offset;
                    UnicastEndpoint intermed_src;

                    // Reserve space in CBs and read data
                    // Order: l, m, o (same as sender writes)
                    CircularBuffer cb_l(cb_l_in);
                    cb_l.reserve_back(PNHt);
                    uint32_t l_write_ptr = cb_l.get_write_ptr();
                    noc.async_read(
                        intermed_src,
                        CoreLocalMem<uint32_t>(l_write_ptr),
                        ml_read_size,
                        {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr},
                        {});
                    intermed_l1_read_addr += ml_read_size;
                    noc.async_read_barrier();
                    cb_l.push_back(PNHt);

                    CircularBuffer cb_m(cb_m_in);
                    cb_m.reserve_back(PNHt);
                    uint32_t m_write_ptr = cb_m.get_write_ptr();
                    noc.async_read(
                        intermed_src,
                        CoreLocalMem<uint32_t>(m_write_ptr),
                        ml_read_size,
                        {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr},
                        {});
                    intermed_l1_read_addr += ml_read_size;
                    noc.async_read_barrier();
                    cb_m.push_back(PNHt);

                    CircularBuffer cb_o(cb_out_o);
                    cb_o.reserve_back(out_chunk_tiles);
                    uint32_t o_write_ptr = cb_o.get_write_ptr();
                    noc.async_read(
                        intermed_src,
                        CoreLocalMem<uint32_t>(o_write_ptr),
                        o_read_size,
                        {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr},
                        {});
                    noc.async_read_barrier();
                    cb_o.push_back(out_chunk_tiles);
                }
            }
        }

        // SENDER: send intermediates to parent (only need to do this ONCE, once you send you are done)
        // We have data (checked at function start), so send it
        if (!is_tree_root && should_send_to_parent) {
            // Wait for compute to finish writing to cb_out_worker, cb_out_m, cb_out_l
            CircularBuffer cb_out_w(cb_out_worker);
            CircularBuffer cb_out_m_buf(cb_out_m);
            CircularBuffer cb_out_l_buf(cb_out_l);
            CircularBuffer cb_intermed(cb_intermed_out);
            cb_out_w.wait_front(out_chunk_tiles);
            cb_out_m_buf.wait_front(PNHt);
            cb_out_l_buf.wait_front(PNHt);

            constexpr uint32_t tile_bytes = get_tile_size(cb_out_worker);
            uint32_t block_offset = send_at_round * (out_chunk_tiles + 2 * PNHt) * tile_bytes;
            constexpr uint32_t o_write_size = out_chunk_tiles * tile_bytes;
            constexpr uint32_t ml_write_size = PNHt * tile_bytes;

            // Get parent's NOC address
            uint32_t parent_noc_x = reduction_group_core_xs[parent_core_in_group];
            uint32_t parent_noc_y = reduction_group_core_ys[parent_core_in_group];
            uint32_t output_write_addr = cb_intermed.get_write_ptr() + block_offset;
            UnicastEndpoint output_dst;

            // Send l, m, o to parent (same order as original worker_compute)
            noc.async_write(
                CoreLocalMem<uint32_t>(cb_out_l_buf.get_read_ptr()),
                output_dst,
                ml_write_size,
                {},
                {.noc_x = parent_noc_x, .noc_y = parent_noc_y, .addr = output_write_addr});
            output_write_addr += ml_write_size;
            noc.async_write(
                CoreLocalMem<uint32_t>(cb_out_m_buf.get_read_ptr()),
                output_dst,
                ml_write_size,
                {},
                {.noc_x = parent_noc_x, .noc_y = parent_noc_y, .addr = output_write_addr});
            output_write_addr += ml_write_size;
            noc.async_write(
                CoreLocalMem<uint32_t>(cb_out_w.get_read_ptr()),
                output_dst,
                o_write_size,
                {},
                {.noc_x = parent_noc_x, .noc_y = parent_noc_y, .addr = output_write_addr});
            noc.async_write_barrier();
            // Reducer semaphore lives at the same L1 offset on every core, so Semaphore<>::up()
            // (which encodes the dst NoC addr from local_l1_addr_) is the correct fit here.
            Semaphore<>(reducer_semaphore_id).up(noc, parent_noc_x, parent_noc_y, step_semaphore_inc[send_at_round]);

            // pop front
            cb_out_w.pop_front(out_chunk_tiles);
            cb_out_m_buf.pop_front(PNHt);
            cb_out_l_buf.pop_front(PNHt);
            noc.async_atomic_barrier();
            // Senders can return, dont need to participate
            return;
        }

        if (!is_tree_root) {
            return;
        }

        // ROOT CORE REMAINING WRITER WORK
        // Offset for current batch
        uint32_t out_tile_id = cur_batch * out_chunk_tiles;
        CircularBuffer cb_out_buf(cb_out);
        if constexpr (num_kv_heads > 1 || !is_out_sharded) {
            cb_out_buf.wait_front(out_chunk_tiles);
        }
        noc.async_writes_flushed();

        if constexpr (num_kv_heads > 1) {
            // if gqa, we will need to write partial outputs for each head
            // we are assuming here that num_heads_to_write = nh/nkv is a power of 2 here, so that we don't write
            // partial across phase
            constexpr uint32_t num_heads_to_write = num_q_heads / num_kv_heads;  // each head is one row in a tile
            if (!is_out_sharded) {
                barrier_count = write_partial_tiles_to_memory<cb_out, ELEMENT_SIZE, barrier_threshold, PNHt>(
                    out_tile_id, out_writer, barrier_count, cur_head, num_heads_to_write, out_chunk_tiles);
            }
            // sharded out case
            else if (do_output) {
                constexpr uint32_t SUBTILE_LINE_BYTES = 16 * ELEMENT_SIZE;
                // read from reducer cores
                constexpr uint32_t num_reducers_per_output = num_reducer_cores / num_output_cores;
                constexpr uint32_t num_reducers_to_wait = num_reducers_per_output - 1;
                Semaphore<>(output_semaphore_id).wait(num_reducers_to_wait);

                uint32_t reduce_core_read_index_start = (cur_batch * num_cores_per_batch) / num_cores_per_head;

                UnicastEndpoint out_src;
                const uint32_t peer_l1_base = cb_out_buf.get_read_ptr();
                for (uint32_t reduce_core_read_index = reduce_core_read_index_start + 1;
                     reduce_core_read_index < reduce_core_read_index_start + num_reducers_per_output;
                     reduce_core_read_index++) {
                    uint32_t reduce_core_read_noc_x = all_reducer_noc_x[reduce_core_read_index];
                    uint32_t reduce_core_read_noc_y = all_reducer_noc_y[reduce_core_read_index];

                    for (uint32_t tile = 0; tile < out_chunk_tiles; ++tile) {
                        uint32_t l1_write_addr = cb_out_buf.get_write_ptr() + tile * tile_bytes;
                        uint32_t peer_tile_base = peer_l1_base + tile * tile_bytes;
                        // write partial output for each head
                        for (uint32_t head = 0; head < num_heads_to_write; ++head) {
                            uint32_t starting_row = cur_head * num_heads_to_write + head;
                            uint32_t in_tile_offset_by_starting_head =
                                starting_row < 16 ? starting_row * SUBTILE_LINE_BYTES
                                                  : (starting_row - 16) * SUBTILE_LINE_BYTES + 512 * ELEMENT_SIZE;
                            uint32_t peer_addr_head = peer_tile_base + in_tile_offset_by_starting_head;
                            uint32_t l1_write_addr_head = l1_write_addr + in_tile_offset_by_starting_head;

                            // First phase
                            noc.async_read(
                                out_src,
                                CoreLocalMem<uint32_t>(l1_write_addr_head),
                                SUBTILE_LINE_BYTES,
                                {.noc_x = reduce_core_read_noc_x,
                                 .noc_y = reduce_core_read_noc_y,
                                 .addr = peer_addr_head},
                                {});

                            // Second phase
                            noc.async_read(
                                out_src,
                                CoreLocalMem<uint32_t>(l1_write_addr_head + 256 * ELEMENT_SIZE),
                                SUBTILE_LINE_BYTES,
                                {.noc_x = reduce_core_read_noc_x,
                                 .noc_y = reduce_core_read_noc_y,
                                 .addr = peer_addr_head + 256 * ELEMENT_SIZE},
                                {});

                            if (++barrier_count == barrier_threshold) {
                                noc.async_read_barrier();
                                barrier_count = 0;
                            }
                        }
                    }
                }
                noc.async_read_barrier();
            } else {
                // tell output core that its output is ready
                uint32_t output_core_noc_x = all_output_noc_x[cur_batch];
                uint32_t output_core_noc_y = all_output_noc_y[cur_batch];
                Semaphore<>(output_semaphore_id).up(noc, output_core_noc_x, output_core_noc_y, 1);
            }
        } else {
            // MQA (Multi Query Attention):  we don't need to gather outputs for other heads so we can just write entire
            // tiles to memory
            if (!is_out_sharded) {
                barrier_count = write_tiles_to_memory<cb_out, out_chunk_tiles, barrier_threshold>(
                    out_tile_id, out_writer, barrier_count);
            }
        }
        if constexpr (num_kv_heads > 1 || !is_out_sharded) {
            noc.async_write_barrier();
            cb_out_buf.pop_front(out_chunk_tiles);
        }

        // [sdpa_decode_lse] The root's final statistics: the compute kernel copied its m (the row max of the raw
        // scores) and l (the sum of exp((s - m) * scale)) column tiles into cb_out_m / cb_out_l before the 1 / l
        // normalization. Each tile is written whole into its own page of the stats tensor; with 16x32 half tiles
        // (num_q_heads <= 16) that is the page's first two faces, i.e. rows 0..15, and rows 16..31 stay untouched.
        {
            CircularBuffer cb_stats_m(cb_out_m);
            CircularBuffer cb_stats_l(cb_out_l);
            constexpr uint32_t stats_tile_bytes = get_tile_size(cb_out_m);
            const auto stats_writer = TensorAccessor(stats_args, stats_addr);
            cb_stats_m.wait_front(PNHt);
            cb_stats_l.wait_front(PNHt);
            uint32_t m_read_addr = cb_stats_m.get_read_ptr();
            uint32_t l_read_addr = cb_stats_l.get_read_ptr();
            for (uint32_t t = 0; t < PNHt; ++t) {
                const uint32_t page = (cur_batch * PNHt + t) * 2;
                noc.async_write(
                    CoreLocalMem<uint32_t>(m_read_addr), stats_writer, stats_tile_bytes, {}, {.page_id = page});
                noc.async_write(
                    CoreLocalMem<uint32_t>(l_read_addr), stats_writer, stats_tile_bytes, {}, {.page_id = page + 1});
                m_read_addr += stats_tile_bytes;
                l_read_addr += stats_tile_bytes;
            }
            noc.async_write_barrier();
            cb_stats_m.pop_front(PNHt);
            cb_stats_l.pop_front(PNHt);
        }
    }
}
