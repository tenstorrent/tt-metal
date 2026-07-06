// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Validator receiver for the prefetcher-vs-matmul contract.
// See tt_metal/impl/buffers/prefetcher_matmul_design.md §3 ("Per-block source tiles") for the
// (bank, receiver, block) -> tile-range mapping this kernel derives expected
// bytes from.
//
// Per pushed page (one block per layer per receiver):
//   1. remote_cb_wait_front(1)
//   2. read the receiver's expected tile range from the source tensor via
//      TensorAccessor (bank routing handled by the accessor — no hand-rolled
//      addr arithmetic)
//   3. memcmp expected vs received page; on mismatch DPRINT details + hang
//   4. remote_cb_pop_front(1)
// After num_layers * num_blocks iterations, bounded-poll for extra pages so
// sender overshoot still surfaces.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "api/debug/dprint.h"

namespace {

constexpr uint32_t kExtraPollCycles = 1u << 18;  // ~262k spin iterations

}  // namespace

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t scratch_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_layers = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t num_senders = get_compile_time_arg_val(4);
    constexpr uint32_t print_stride = get_compile_time_arg_val(5);
    // TensorAccessor compile-time args start at index 6.
    constexpr auto tensor_args = TensorAccessorArgs<6>();

    // ---- Runtime args ----
    // Host derives `n_col_start` (= ring_pos * n_per_recv_tiles) and
    // `total_n_tiles` (= N / TILE_WIDTH) per-receiver based on the GCB topology
    // and tensor logical shape, so this kernel is layout-agnostic — the legacy
    // K-row-major and the receiver-contiguous DRAM-core layouts both reach
    // here with the right starting tile column for their ring position.
    uint32_t rt_idx = 0;
    const uint32_t bank_id = get_arg_val<uint32_t>(rt_idx++);           // sender's DRAM bank (diagnostic only)
    const uint32_t recv_idx_in_bank = get_arg_val<uint32_t>(rt_idx++);  // 0 .. num_receivers_per_sender-1 (diagnostic)
    const uint32_t bank_base_addr = get_arg_val<uint32_t>(rt_idx++);    // source tensor base addr
    const uint32_t k_block_w_tiles = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t total_n_tiles = get_arg_val<uint32_t>(rt_idx++);  // N / TILE_WIDTH (full tensor)
    const uint32_t n_per_recv_tiles = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t n_col_start = get_arg_val<uint32_t>(rt_idx++);  // ring_pos * n_per_recv_tiles
    (void)num_senders;                                             // total_n_tiles now comes directly from the host

    const auto accessor = TensorAccessor(tensor_args, bank_base_addr);
    const uint32_t tile_bytes = accessor.get_aligned_page_size();
    const uint32_t slice_bytes = n_per_recv_tiles * tile_bytes;
    const uint32_t page_bytes = k_block_w_tiles * slice_bytes;

    const uint32_t scratch_addr = get_write_ptr(scratch_cb_id);

    DPRINT(
        "VALIDATOR_START bank={} recv_idx={} num_layers={} num_blocks={} page={} tile={}\n",
        bank_id,
        recv_idx_in_bank,
        num_layers,
        num_blocks,
        page_bytes,
        tile_bytes);

    uint32_t global_iter = 0;
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            experimental::remote_cb_wait_front(remote_cb_id, 1);
            RemoteReceiverCBInterface& iface = get_remote_receiver_cb_interface(remote_cb_id);
            const uint32_t page_addr = iface.fifo_rd_ptr;

            // Read expected tiles via TensorAccessor. Per tt_metal/impl/buffers/prefetcher_matmul_design.md §3,
            // page row h = tiles (blk*kw + h, n_col_start + n) for n in [0, n_per_recv). One
            // accessor call per tile keeps bank-routing logic out of this kernel.
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

            // Byte-for-byte compare. Word-stride loop; report the first mismatching word.
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
                    "VALIDATOR_MISMATCH layer={} blk={} bank={} recv_idx={} word={} got=0x{:x} exp=0x{:x}\n",
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
                DPRINT("VALIDATOR ok layer={} blk={} bank={} recv_idx={}\n", layer, blk, bank_id, recv_idx_in_bank);
            }

            experimental::remote_cb_pop_front(remote_cb_id, 1);
            ++global_iter;
        }
    }

    DPRINT("VALIDATOR_LOOP_DONE bank={} recv_idx={}\n", bank_id, recv_idx_in_bank);

    // Bounded-poll for an extra page (sender overshoot).
    volatile tt_l1_ptr uint32_t* pages_acked_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_remote_receiver_cb_interface(remote_cb_id).aligned_pages_acked_ptr);
    volatile tt_l1_ptr uint32_t* pages_sent_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_remote_receiver_cb_interface(remote_cb_id).aligned_pages_acked_ptr - L1_ALIGNMENT);
    for (uint32_t spin = 0; spin < kExtraPollCycles; ++spin) {
        invalidate_l1_cache();
        const uint32_t sent = *pages_sent_ptr;
        const uint32_t acked = *pages_acked_ptr;
        if (sent != acked) {
            DPRINT("VALIDATOR_OVERFLOW: sender pushed an extra page; pages_sent={} pages_acked={}\n", sent, acked);
            while (true) {
                ;
            }
        }
    }

    DPRINT("VALIDATOR_DONE ok bank={} recv_idx={}\n", bank_id, recv_idx_in_bank);
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc_async_atomic_barrier();
}
