// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Weight reader of the gemma4-local moe_compute port (custom_ops/moe_compute/README.md section 2): the shared op's
// dm0.cpp with the stream reordered so each expert's W0/W1 and W2 slices of this ring core cross DRAM ONCE per call
// into two resident CB regions (c_3: W0/W1, c_10: W2), whatever the number of 32-token chunks routed to the expert,
// with `prefetch_depth` whole-block reads in flight. The compute kernel addresses the resident regions by tile index
// and pops each region once per expert, so the next expert's W0/W1 streams while the current expert's last down
// phase runs. The DRAM layout, bank walk and block geometry are the shared op's (moe_ring_common.h).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/moe_ring_common.h"

void kernel_main() {
    constexpr bool has_bias = get_named_compile_time_arg_val("has_bias") == 1;
    constexpr uint32_t Ht = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t Nt = get_named_compile_time_arg_val("intermediate_tiles");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t num_shared_experts = get_named_compile_time_arg_val("num_shared_experts");
    constexpr uint32_t shared_expert_tp_factor = get_named_compile_time_arg_val("shared_expert_tp_factor");
    constexpr uint32_t tiles_per_txn = get_named_compile_time_arg_val("tiles_per_txn");

    using Cfg = moe_ring::MoeRingConfig<Ht, Nt, num_cores, has_bias, shared_expert_tp_factor, tiles_per_txn>;

    constexpr uint32_t num_banks = get_named_compile_time_arg_val("num_banks");
    constexpr uint32_t w2_pages_per_ring_core_total = get_named_compile_time_arg_val("w2_pages_per_ring_core_total");
    constexpr uint32_t metadata_ready_semaphore_id = get_named_compile_time_arg_val("metadata_ready_semaphore_id");
    constexpr uint32_t per_expert_total_tokens_cb_id = get_named_compile_time_arg_val("per_expert_total_tokens_cb_id");
    [[maybe_unused]] constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");

    // This port's knobs: blocks of DRAM reads in flight (one transaction id each, 1..15), NoC reads per block (1 =
    // a whole 2-transaction block per read: Blackhole's 16 KB burst holds it; 2 = the shared op's per-transaction
    // reads), the resident W0/W1 region in tiles (the largest ring core's routed slice, every core reserves and
    // pushes this much so the CB wraps exactly once per expert) and the W2 CB index.
    constexpr uint32_t prefetch_depth = get_named_compile_time_arg_val("prefetch_depth");
    constexpr uint32_t txns_per_read = get_named_compile_time_arg_val("txns_per_read");
    constexpr uint32_t w01_region_tiles = get_named_compile_time_arg_val("w01_region_tiles");
    constexpr uint32_t w2_cb_id = get_named_compile_time_arg_val("w2_cb_id");
    static_assert(prefetch_depth >= 1 && prefetch_depth <= 15, "one transaction id per block in flight, ids 1..15");
    static_assert(txns_per_read == 1 || txns_per_read == 2, "a block is 2 transactions");

    constexpr auto w0_w1_args = TensorAccessorArgs<0>();
    constexpr auto w2_args = TensorAccessorArgs<w0_w1_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto out_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();

    // Runtime args: the shared op's layout (dm0 / dm1 / compute share it), then this port's layer id.
    uint32_t argidx = 0;
    [[maybe_unused]] const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto out_addr = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto ring_neighbor_physical_x = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto ring_neighbor_physical_y = get_arg_val<uint32_t>(argidx++);
    uint32_t shard_to_bank[num_banks];
    for (uint32_t i = 0; i < num_banks; ++i) {
        shard_to_bank[i] = get_arg_val<uint32_t>(argidx++);
    }
    const auto layer_id = get_arg_val<uint32_t>(argidx++);

    constexpr auto cb_w01_id = tt::CBIndex::c_3;
    CircularBuffer cb_w01(cb_w01_id);
    CircularBuffer cb_w2(w2_cb_id);
    CircularBuffer cb_per_expert_total_tokens(per_expert_total_tokens_cb_id);

    constexpr uint32_t tile_size = get_tile_size(cb_w01_id);
    static_assert(get_tile_size(cb_w01_id) == get_tile_size(w2_cb_id), "both weight streams are Bfp4_b tiles");

    constexpr uint32_t tiles_per_block = Cfg::tiles_per_block;
    constexpr uint32_t bytes_per_txn = tiles_per_txn * tile_size;
    constexpr uint32_t bytes_per_block = tiles_per_block * tile_size;
    constexpr uint32_t read_bytes = txns_per_read == 1 ? bytes_per_block : bytes_per_txn;
    constexpr uint32_t reads_per_block = txns_per_read;
    constexpr uint32_t tiles_per_read = tiles_per_block / reads_per_block;
    static_assert(read_bytes <= NOC_MAX_BURST_SIZE, "a read must be one NoC packet");

    //-------------------------------------------------------------------------
    // W0/W1 compact bank-balanced layout and W2 core-major layout (moe_ring_common.h; comments in the shared dm0)
    //-------------------------------------------------------------------------
    constexpr auto w0_w1_block_offset_lut = moe_ring::make_w0_w1_block_offset_lut<Cfg, num_cores>();
    constexpr uint32_t w0_w1_bank_pages_per_expert = Cfg::w0_w1_bank_blocks_per_expert(num_banks) * tiles_per_block;
    static_assert(w0_w1_bank_pages_per_expert % tiles_per_block == 0, "bank pieces are whole blocks");
    static_assert(
        w0_w1_block_offset_lut[num_cores] <= num_banks * Cfg::w0_w1_bank_blocks_per_expert(num_banks),
        "the cores' w0_w1 slices of one expert must fit in its num_banks bank pieces");

    constexpr uint32_t w2_pages_per_logical_shard = Cfg::w2_blocks_per_expert * tiles_per_block;
    constexpr uint32_t w2_pages_total = num_cores * w2_pages_per_ring_core_total;
    static_assert(w2_pages_total % num_banks == 0, "w2 pages_total must be divisible by num_banks");
    constexpr uint32_t w2_pages_per_bank_total = w2_pages_total / num_banks;
    static_assert(w2_pages_per_bank_total % tiles_per_block == 0, "a W2 block never straddles a bank");
    constexpr uint32_t w2_region_tiles = w2_pages_per_logical_shard;
    static_assert(w2_region_tiles % tiles_per_block == 0, "the W2 region is whole blocks");
    static_assert(w01_region_tiles % tiles_per_block == 0, "the W0/W1 region is whole blocks");

    const uint32_t w0_w1_layer_offset_in_bank = layer_id * num_experts * w0_w1_bank_pages_per_expert;
    const uint32_t w2_layer_offset_in_ring_core = layer_id * num_experts * w2_pages_per_logical_shard;

    // This ring core's slice position inside every (layer, expert) W0/W1 stream.
    const uint32_t w0_w1_core_first_stream_page = w0_w1_block_offset_lut[ring_core_id] * tiles_per_block;
    const uint32_t w0_w1_core_first_shard_idx = w0_w1_core_first_stream_page / w0_w1_bank_pages_per_expert;
    const uint32_t w0_w1_core_first_piece_page =
        w0_w1_core_first_stream_page - w0_w1_core_first_shard_idx * w0_w1_bank_pages_per_expert;
    const uint32_t w2_ring_core_first_global_page =
        ring_core_id * w2_pages_per_ring_core_total + w2_layer_offset_in_ring_core;

    //-------------------------------------------------------------------------
    // Prefetch pipeline: up to prefetch_depth blocks (one trid each) in flight; the oldest is waited for and
    // pushed when the pipeline is full and at the end of each region.
    //-------------------------------------------------------------------------
    uint32_t cur_shard_idx = 0xFFFFFFFFu;  // the bank the shared read cmd-buf state is set for
    uint32_t trid_issue = 1, trid_wait = 1, pending = 0;
    CircularBuffer* pending_cb[prefetch_depth + 1];  // the CB each in-flight block belongs to (trid order)
    uint32_t pending_head = 0, pending_tail = 0;

    auto advance_trid = [](uint32_t& t) { t = (t == prefetch_depth) ? 1u : t + 1u; };
    auto set_bank_state = [&](uint32_t shard_idx) {
        if (shard_idx != cur_shard_idx) {
            const uint64_t bank_base = get_noc_addr_from_bank_id<true>(shard_to_bank[shard_idx], 0);
            noc_async_read_one_packet_set_state<true>(bank_base, read_bytes, vchannel);
            cur_shard_idx = shard_idx;
        }
    };
    auto wait_oldest = [&]() {
        noc_async_read_barrier_with_trid(trid_wait);
        pending_cb[pending_head]->push_back(tiles_per_block);
        pending_head = (pending_head == prefetch_depth) ? 0u : pending_head + 1u;
        advance_trid(trid_wait);
        --pending;
    };
    // One block: `reads_per_block` reads from bank `shard_idx` at byte offset `src` into `dst`.
    auto issue_block = [&](CircularBuffer* cb, uint32_t shard_idx, uint32_t src, uint32_t dst) {
        if (pending == prefetch_depth) {
            wait_oldest();
        }
        set_bank_state(shard_idx);
        noc_async_read_set_trid(trid_issue);
        const uint64_t bank_noc = get_noc_addr_from_bank_id<true>(shard_to_bank[shard_idx], 0);
        for (uint32_t r = 0; r < reads_per_block; ++r) {
            noc_async_read_one_packet_with_state_with_trid</*skip_ptr_update=*/false, /*skip_cmdbuf_chk=*/false>(
                bank_noc, src, dst, trid_issue);
            src += read_bytes;
            dst += read_bytes;
        }
        pending_cb[pending_tail] = cb;
        pending_tail = (pending_tail == prefetch_depth) ? 0u : pending_tail + 1u;
        advance_trid(trid_issue);
        ++pending;
    };

    //-------------------------------------------------------------------------
    // Per-expert token counts from the tilize drain core
    //-------------------------------------------------------------------------
    Semaphore<> metadata_ready_sem(metadata_ready_semaphore_id);
    metadata_ready_sem.wait_min(1);
    volatile tt_l1_ptr uint32_t* num_tokens_per_expert_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_per_expert_total_tokens.get_read_ptr());

    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        const uint32_t num_tokens = num_tokens_per_expert_ptr[expert_id];
        if (num_tokens == 0) {
            continue;  // no chunk: the compute kernel skips the expert too, nothing is streamed
        }

        // Routed experts read the core's whole compact slice; shared experts only the front-packed real prefix.
        const bool is_shared_expert = expert_id >= num_experts - num_shared_experts;
        const uint32_t w0_w1_blocks_this_expert = moe_ring::w0_w1_blocks_for_cols(
            Cfg::w0_w1_prod_cols(ring_core_id, is_shared_expert),
            Cfg::w0_w1_blocks_per_col,
            Cfg::w0_w1_blocks_per_half_col);
        const uint32_t w0_w1_expert_first_bank_page =
            w0_w1_layer_offset_in_bank + expert_id * w0_w1_bank_pages_per_expert;

        //---------------------------------------------------------------------
        // W0/W1: the whole slice into the resident region (released by compute after the expert's last chunk)
        //---------------------------------------------------------------------
        cb_w01.reserve_back(w01_region_tiles);
        {
            uint32_t dst = cb_w01.get_write_ptr();
            uint32_t shard_idx = w0_w1_core_first_shard_idx;
            uint32_t piece_page = w0_w1_core_first_piece_page;
            for (uint32_t block_id = 0; block_id < w0_w1_blocks_this_expert; ++block_id) {
                issue_block(
                    &cb_w01, shard_idx, (w0_w1_expert_first_bank_page + piece_page) * tile_size + w0_w1_addr, dst);
                dst += bytes_per_block;
                piece_page += tiles_per_block;
                if (piece_page == w0_w1_bank_pages_per_expert) {
                    ++shard_idx;
                    piece_page = 0;
                }
            }
        }
        // Drain this region's reads before the padding push so the region's pushes stay in block order.
        while (pending > 0) {
            wait_oldest();
        }
        const uint32_t w01_pad_tiles = w01_region_tiles - w0_w1_blocks_this_expert * tiles_per_block;
        if (w01_pad_tiles > 0) {
            cb_w01.push_back(w01_pad_tiles);  // unfilled tail of the region: never read by compute
        }

        //---------------------------------------------------------------------
        // W2: the full Nt-tall slice (shared experts' W2 is zero-padded to full height)
        //---------------------------------------------------------------------
        cb_w2.reserve_back(w2_region_tiles);
        {
            uint32_t dst = cb_w2.get_write_ptr();
            uint32_t global_page = w2_ring_core_first_global_page + expert_id * w2_pages_per_logical_shard;
            for (uint32_t block_id = 0; block_id < Cfg::w2_blocks_per_expert; ++block_id) {
                const uint32_t shard_idx = global_page / w2_pages_per_bank_total;
                const uint32_t in_bank_page = global_page - shard_idx * w2_pages_per_bank_total;
                issue_block(&cb_w2, shard_idx, in_bank_page * tile_size + w2_addr, dst);
                dst += bytes_per_block;
                global_page += tiles_per_block;
            }
        }
        while (pending > 0) {
            wait_oldest();
        }
    }
}
