// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"

// Tensix setup kernel: resolve the fabric router connection parameters exactly the
// way a Tensix worker does — WorkerToFabricEdmSender::build_from_args() on the
// runtime args append_fabric_connection_rt_args(..., CoreType::WORKER) emitted — and
// hand them to the x280 firmware through its mailbox connection block. The
// connection is NOT opened here; the x280 opens it.
//
// Also plants a magic word in the router's producer-cursor pad (a word the router
// never touches after bring-up) so the x280 can verify which coordinate system its
// TLB window needs for the EDM core.
//
// Connection block field offsets mirror ../x280/fabric_mbox.h (FF_CONN_*). The block
// is written as one 128 B NOC write, then the VALID word last.
void kernel_main() {
    size_t idx = 0;
    const uint32_t l1_scratch = get_arg_val<uint32_t>(idx++);
    const uint32_t dram_out = get_arg_val<uint32_t>(idx++);
    const uint32_t out_size = get_arg_val<uint32_t>(idx++);
    const uint32_t l2cpu_x = get_arg_val<uint32_t>(idx++);
    const uint32_t l2cpu_y = get_arg_val<uint32_t>(idx++);
    const uint32_t conn_block = get_arg_val<uint32_t>(idx++);  // FF_MBOX_CONN (x280 address)
    const uint32_t self_x = get_arg_val<uint32_t>(idx++);      // L2CPU tile coords the router should use
    const uint32_t self_y = get_arg_val<uint32_t>(idx++);
    const uint32_t hdr_size = get_arg_val<uint32_t>(idx++);
    const uint32_t edm_noc0_x = get_arg_val<uint32_t>(idx++);
    const uint32_t edm_noc0_y = get_arg_val<uint32_t>(idx++);
    const uint32_t freeslots_sink = get_arg_val<uint32_t>(idx++);
    const uint32_t teardown_word = get_arg_val<uint32_t>(idx++);
    const uint32_t cursor_magic = get_arg_val<uint32_t>(idx++);
    const uint32_t peer_x = get_arg_val<uint32_t>(idx++);
    const uint32_t peer_y = get_arg_val<uint32_t>(idx++);
    const uint32_t peer_inbox = get_arg_val<uint32_t>(idx++);
    const uint32_t peer_hops = get_arg_val<uint32_t>(idx++);
    const uint32_t flags = get_arg_val<uint32_t>(idx++);
    const uint32_t reserved0 = get_arg_val<uint32_t>(idx++);    // unused (kept for arg layout)
    const uint32_t valid_nonce = get_arg_val<uint32_t>(idx++);  // FF_CONN_VALID value (nonzero, unique per setup)
    // ... followed by the args append_fabric_connection_rt_args(CoreType::WORKER) appended.
    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    constexpr auto out0_args = TensorAccessorArgs<0>();
    const auto out0 = TensorAccessor(out0_args, dram_out, out_size);

    const uint32_t sreg_write = static_cast<uint32_t>(sender.edm_buffer_remote_free_slots_update_addr);
    const uint32_t stream_id = (sreg_write - NOC_OVERLAY_START_ADDR) / NOC_STREAM_REG_SPACE_SIZE;
    const uint32_t sreg_read = STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);

    // Plant the coordinate-probe magic in the cursor pad word on the router.
    noc_inline_dw_write(
        get_noc_addr(sender.edm_noc_x, sender.edm_noc_y, static_cast<uint32_t>(sender.edm_copy_of_wr_counter_addr) + 8),
        cursor_magic);
    noc_async_write_barrier();

    (void)reserved0;

    volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_scratch);
    for (uint32_t i = 0; i < 32; i++) {
        b[i] = 0;
    }
    b[0x00 / 4] = sender.edm_noc_x;                                                // FF_CONN_EDM_NOC_X
    b[0x04 / 4] = sender.edm_noc_y;                                                // FF_CONN_EDM_NOC_Y
    b[0x08 / 4] = edm_noc0_x;                                                      // FF_CONN_EDM_NOC0_X
    b[0x0c / 4] = edm_noc0_y;                                                      // FF_CONN_EDM_NOC0_Y
    b[0x10 / 4] = static_cast<uint32_t>(sender.edm_buffer_base_addr);              // FF_CONN_BUFFER_BASE
    b[0x14 / 4] = sender.num_buffers_per_channel;                                  // FF_CONN_NUM_BUFFERS
    b[0x18 / 4] = sender.buffer_size_bytes;                                        // FF_CONN_BUFFER_SIZE
    b[0x1c / 4] = static_cast<uint32_t>(sender.edm_connection_handshake_l1_addr);  // FF_CONN_HANDSHAKE
    b[0x20 / 4] = static_cast<uint32_t>(sender.edm_worker_location_info_addr);     // FF_CONN_WORKER_LOC_INFO
    b[0x24 / 4] = static_cast<uint32_t>(sender.edm_copy_of_wr_counter_addr);       // FF_CONN_WR_COUNTER
    b[0x28 / 4] = sreg_write;                                                      // FF_CONN_SREG_WRITE
    b[0x2c / 4] = sreg_read;                                                       // FF_CONN_SREG_READ
    b[0x30 / 4] = self_x;                                                          // FF_CONN_SELF_NOC_X
    b[0x34 / 4] = self_y;                                                          // FF_CONN_SELF_NOC_Y
    b[0x38 / 4] = hdr_size;                                                        // FF_CONN_HDR_SIZE
    b[0x3c / 4] = freeslots_sink;                                                  // FF_CONN_FREESLOTS_SINK
    b[0x40 / 4] = teardown_word;                                                   // FF_CONN_TEARDOWN_WORD
    b[0x44 / 4] = cursor_magic;                                                    // FF_CONN_CURSOR_MAGIC
    b[0x48 / 4] = peer_x;                                                          // FF_CONN_PEER_NOC_X
    b[0x4c / 4] = peer_y;                                                          // FF_CONN_PEER_NOC_Y
    b[0x50 / 4] = peer_inbox;                                                      // FF_CONN_PEER_INBOX
    b[0x54 / 4] = peer_hops;                                                       // FF_CONN_PEER_HOPS
    b[0x58 / 4] = flags;                                                           // FF_CONN_FLAGS
    // 0x64: host-only diagnostic (not read by the firmware)
    b[0x64 / 4] = stream_id;
    b[0x7c / 4] = 0;  // FF_CONN_VALID — published separately below

    noc_async_write(l1_scratch, get_noc_addr(l2cpu_x, l2cpu_y, conn_block), 128);
    noc_async_write_barrier();
    noc_inline_dw_write(get_noc_addr(l2cpu_x, l2cpu_y, conn_block + 0x7c), valid_nonce);  // FF_CONN_VALID
    noc_async_write_barrier();

    // Same block to DRAM for the host to print.
    b[0x7c / 4] = valid_nonce;
    noc_async_write_page(0, out0, l1_scratch);
    noc_async_write_barrier();
}
