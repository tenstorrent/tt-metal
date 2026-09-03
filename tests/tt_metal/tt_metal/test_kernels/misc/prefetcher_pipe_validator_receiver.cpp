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
//   * batched delivery only, so FIFO position == physical block and there is no rotation arg;
//   * no update_remote_cb_config_in_l1 / atomic barrier at exit -- the durable read cursor is
//     checkpointed by PrefetcherPipe::commit() when the object goes out of scope, and acks are
//     posted.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "api/debug/dprint.h"

namespace {

constexpr uint32_t kExtraPollCycles = 1u << 18;  // ~262k spin iterations

}  // namespace

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t scratch_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_layers = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t print_stride = get_compile_time_arg_val(3);
    // TensorAccessor compile-time args start at index 4.
    constexpr auto tensor_args = TensorAccessorArgs<4>();

    // ---- Runtime args ----
    // The host derives n_col_start (= ring_pos * n_per_recv_tiles) and total_n_tiles from the
    // pipes' topology and the tensor's padded shape, so this kernel stays layout-agnostic.
    uint32_t rt_idx = 0;
    // One kernel serves the receivers of every pipe, and a core's pipe id depends on which sender
    // drives it, so the id is a runtime arg rather than a compile-time one.
    const uint8_t prefetcher_pipe_id = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_idx++));
    const uint32_t bank_id = get_arg_val<uint32_t>(rt_idx++);           // sender's DRAM bank (diagnostic only)
    const uint32_t recv_idx_in_bank = get_arg_val<uint32_t>(rt_idx++);  // bank-local receiver index (diagnostic)
    const uint32_t bank_base_addr = get_arg_val<uint32_t>(rt_idx++);    // source tensor base addr
    const uint32_t k_block_w_tiles = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t total_n_tiles = get_arg_val<uint32_t>(rt_idx++);  // N / TILE_WIDTH (full tensor)
    const uint32_t n_per_recv_tiles = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t n_col_start = get_arg_val<uint32_t>(rt_idx++);  // ring_pos * n_per_recv_tiles

    const auto accessor = TensorAccessor(tensor_args, bank_base_addr);
    const uint32_t tile_bytes = accessor.get_aligned_page_size();
    const uint32_t slice_bytes = n_per_recv_tiles * tile_bytes;
    const uint32_t page_bytes = k_block_w_tiles * slice_bytes;

    const uint32_t scratch_addr = get_write_ptr(scratch_cb_id);

    Noc noc;
    experimental::PrefetcherPipe pipe(prefetcher_pipe_id);

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
            const uint32_t page_addr = pipe.get_read_ptr().get_address();

            // Page row h = tiles (blk*kw + h, n_col_start + n) for n in [0, n_per_recv). One
            // accessor call per tile keeps bank-routing logic out of this kernel. Batched delivery,
            // so the FIFO position is the physical block.
            uint32_t scratch_cursor = scratch_addr;
            for (uint32_t h = 0; h < k_block_w_tiles; ++h) {
                const uint32_t k_row = blk * k_block_w_tiles + h;
                const uint32_t row_page_base = k_row * total_n_tiles + n_col_start;
                for (uint32_t n = 0; n < n_per_recv_tiles; ++n) {
                    const uint64_t src_noc = accessor.get_noc_addr(row_page_base + n);
                    noc_async_read(src_noc, scratch_cursor, tile_bytes);
                    scratch_cursor += tile_bytes;
                }
            }
            noc_async_read_barrier();

            volatile tt_l1_ptr uint32_t* received = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_addr);
            volatile tt_l1_ptr uint32_t* expected = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr);
            const uint32_t words = page_bytes / sizeof(uint32_t);
            uint32_t mismatch_word = words;
            for (uint32_t w = 0; w < words; ++w) {
                if (received[w] != expected[w]) {
                    mismatch_word = w;
                    break;
                }
            }
            if (mismatch_word != words) {
                DPRINT(
                    "PIPE_VALIDATOR_MISMATCH layer={} blk={} bank={} recv_idx={} word={} got=0x{:x} exp=0x{:x}\n",
                    layer,
                    blk,
                    bank_id,
                    recv_idx_in_bank,
                    mismatch_word,
                    (uint32_t)received[mismatch_word],
                    (uint32_t)expected[mismatch_word]);
                // Hang so the dispatch timeout surfaces this core.
                while (true) {
                    ;
                }
            }

            const bool log = (global_iter < 2) || (global_iter + 1 == num_layers * num_blocks) ||
                             (print_stride > 0 && (global_iter % print_stride == 0));
            if (log) {
                DPRINT(
                    "PIPE_VALIDATOR ok layer={} blk={} bank={} recv_idx={}\n", layer, blk, bank_id, recv_idx_in_bank);
            }

            pipe.pop_front(1, noc);
            ++global_iter;
        }
    }

    DPRINT("PIPE_VALIDATOR_LOOP_DONE bank={} recv_idx={}\n", bank_id, recv_idx_in_bank);

    // Bounded-poll for an extra entry (sender overshoot). entries_sent sits one L1_ALIGNMENT below
    // entries_acked in this receiver's own config page, the same relationship wait_front relies on.
    volatile tt_l1_ptr uint32_t* pages_acked_ptr = pipe.local_pages_acked_ptr();
    volatile tt_l1_ptr uint32_t* pages_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reinterpret_cast<uint32_t>(pages_acked_ptr) - L1_ALIGNMENT);
    for (uint32_t spin = 0; spin < kExtraPollCycles; ++spin) {
        invalidate_l1_cache();
        const uint32_t sent = *pages_sent_ptr;
        const uint32_t acked = *pages_acked_ptr;
        if (sent != acked) {
            DPRINT("PIPE_VALIDATOR_OVERFLOW: sender pushed an extra entry; sent={} acked={}\n", sent, acked);
            while (true) {
                ;
            }
        }
    }

    DPRINT("PIPE_VALIDATOR_DONE ok bank={} recv_idx={}\n", bank_id, recv_idx_in_bank);
}
