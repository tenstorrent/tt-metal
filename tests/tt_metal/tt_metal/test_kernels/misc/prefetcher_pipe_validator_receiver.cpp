// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PrefetcherPipe validator receiver: the PrefetcherPipe counterpart of gcb_validator_receiver.cpp.
//
// Same contract and the same (bank, receiver, block) -> tile-range derivation
// (tt_metal/impl/buffers/prefetcher_matmul_design.md §3); only the consume side differs. Per
// delivered entry:
//   1. pipe.wait_front(1)
//   2. read this receiver's expected tile range from the source tensor via TensorAccessor
//   3. compare expected vs received; on mismatch DPRINT details and hang so the dispatch timeout
//      surfaces the core
//   4. pipe.pop_front(1)
//
// Differences from the GlobalCircularBuffer validator:
//   * reads through the device PrefetcherPipe class (Attached by the host) rather than remote_cb_*;
//   * no update_remote_cb_config_in_l1 / atomic barrier at exit -- the durable read cursor is
//     checkpointed by PrefetcherPipe::commit() when the object goes out of scope, and acks are
//     posted.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/tensor_accessor.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

#include "prefetcher_validator_common.h"

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t num_layers = get_arg(args::num_layers);
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);
    constexpr uint32_t print_stride = get_arg(args::print_stride);
    // Streaming mode: the prefetcher delivers each receiver's blocks ring-rotated, so the entry at
    // FIFO position `blk` is physical block (lead_block + blk) mod num_blocks. Batched delivery is
    // the identity.
    constexpr uint32_t streaming = get_arg(args::streaming);

    // ---- Runtime args ----
    // The host derives n_col_start (= ring_pos * n_per_recv_tiles) and total_n_tiles from the
    // pipes' topology and the tensor's padded shape, so this kernel stays layout-agnostic.
    uint32_t rt_idx = 0;
    const uint32_t bank_id = get_arg_val<uint32_t>(rt_idx++);           // sender's DRAM bank (diagnostic only)
    const uint32_t recv_idx_in_bank = get_arg_val<uint32_t>(rt_idx++);  // bank-local receiver index (diagnostic)
    const uint32_t k_block_w_tiles = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t total_n_tiles = get_arg_val<uint32_t>(rt_idx++);  // N / TILE_WIDTH (full tensor)
    const uint32_t n_per_recv_tiles = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t n_col_start = get_arg_val<uint32_t>(rt_idx++);  // ring_pos * n_per_recv_tiles
    const uint32_t lead_block = get_arg_val<uint32_t>(rt_idx++);   // streaming: physical block at FIFO position 0

    const auto accessor = TensorAccessor(tensor::source_tensor);
    const uint32_t tile_bytes = accessor.get_aligned_page_size();
    const uint32_t slice_bytes = n_per_recv_tiles * tile_bytes;
    const uint32_t page_bytes = k_block_w_tiles * slice_bytes;

    Scratchpad<uint32_t> scratchpad(scratch::expected);
    const uint32_t scratch_addr = scratchpad.get_base_address();

    Noc noc;
    experimental::PrefetcherPipe pipe(pipe::in);

    DPRINT(
        "PIPE_VALIDATOR_START bank={} recv_idx={} num_layers={} num_blocks={} page={} tile={}\n",
        bank_id,
        recv_idx_in_bank,
        num_layers,
        num_blocks,
        page_bytes,
        tile_bytes);

    uint32_t global_iter = 0;
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            pipe.wait_front(1);
            const auto entry = pipe.scoped_read_lock(1);
            const uint32_t page_addr = entry.get_ptr().get_address();

            // Streaming delivers each receiver's blocks ring-rotated, so FIFO position blk is
            // physical block (lead_block + blk) mod num_blocks; batched delivery is the identity.
            const uint32_t phys_blk = streaming ? ((lead_block + blk) % num_blocks) : blk;
            prefetcher_validator::read_expected_block_tiles(
                accessor,
                scratch_addr,
                tile_bytes,
                phys_blk,
                k_block_w_tiles,
                total_n_tiles,
                n_col_start,
                n_per_recv_tiles);

            const uint32_t mismatch_word =
                prefetcher_validator::first_mismatching_word(page_addr, scratch_addr, page_bytes);
            if (mismatch_word != prefetcher_validator::kNoMismatch) {
                DPRINT(
                    "PIPE_VALIDATOR_MISMATCH layer={} blk={} bank={} recv_idx={} word={} got=0x{:x} exp=0x{:x}\n",
                    layer,
                    blk,
                    bank_id,
                    recv_idx_in_bank,
                    mismatch_word,
                    prefetcher_validator::l1_word(page_addr, mismatch_word),
                    prefetcher_validator::l1_word(scratch_addr, mismatch_word));
                // Hang so the dispatch timeout surfaces this core.
                while (true) {
                    ;
                }
            }

            if (prefetcher_validator::should_log(global_iter, num_layers * num_blocks, print_stride)) {
                DPRINT(
                    "PIPE_VALIDATOR ok layer={} blk={} bank={} recv_idx={}\n", layer, blk, bank_id, recv_idx_in_bank);
            }

            pipe.pop_front(1, noc);
            ++global_iter;
        }
    }

    DPRINT("PIPE_VALIDATOR_LOOP_DONE bank={} recv_idx={}\n", bank_id, recv_idx_in_bank);

    // Bounded-poll for an entry the sender pushed past the last one this receiver consumed.
    for (uint32_t spin = 0; spin < prefetcher_validator::kExtraPollCycles; ++spin) {
        if (pipe.has_unconsumed_entries()) {
            DPRINT("PIPE_VALIDATOR_OVERFLOW: sender pushed an extra entry past the expected last one\n");
            while (true) {
                ;
            }
        }
    }

    DPRINT("PIPE_VALIDATOR_DONE ok bank={} recv_idx={}\n", bank_id, recv_idx_in_bank);
}
