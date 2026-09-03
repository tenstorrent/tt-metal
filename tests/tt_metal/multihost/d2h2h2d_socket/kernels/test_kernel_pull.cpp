// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The receive leg, DEVICE_PULL: a Tensix core pulling its own payload out of pinned host
// RAM over PCIe, instead of waiting for the host to push it into L1.
//
// ONE PAGE PER MESSAGE, WHICH IS A CONTRACT WITH THE HOST AND NOT A CONVENIENCE
//
// The host sets page_size to the MESSAGE size and calls write() with a page count of one;
// this kernel waits for one page and reads it whole.
//
// THE DOORBELL THIS RINGS IS rdma_signal, NOT rdma_completion. "Bytes somebody else sent
// landed in your L1". On the push path that is the host; here it is this core. Which
// side writes it changes; what it means does not, and that is the point of the socket being
// a drop-in for the write.

#include <stdint.h>

#include "risc_common.h"
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

#include "tt_metal/distributed/host_uva_layout.hpp"

namespace {

inline void noc_read_page_chunked(uint32_t pcie_xy_enc, uint64_t src_pcie, uint32_t dst_l1, uint32_t size) {
    while (size) {
        uint32_t chunk = size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : size;
        noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
            NOC_INDEX, pcie_xy_enc, src_pcie, dst_l1, chunk);
        src_pcie += chunk;
        dst_l1 += chunk;
        size -= chunk;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t dst_l1_addr = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t signal_addr = get_compile_time_arg_val(2);
    // DEVICE_PULL (true) or HOST_PUSH (false). Both modes need a receiver -- the credit
    // and the doorbell are the socket's, not the transfer's -- and only the source of the
    // bytes differs. Compile-time so the unused branch costs nothing.
    constexpr bool pull_from_host = get_compile_time_arg_val(3);
    // The host writes 1 here when it has no more messages for this core. See below for why
    // a message COUNT cannot do this job.
    constexpr uint32_t stop_addr = get_compile_time_arg_val(4);
    // THE RECEIVE STATUS CONTROL REGISTER, and it is what makes a UVA store work here.
    //
    // dst_l1_addr above is a COMPILE arg -- one address for the whole run -- which is all
    // kOpSendUva ever needs, its destination being fixed by definition. A store names its
    // own address per message, and no compile arg can carry that. So the host writes an
    // instruction into this L1 word: `(offset, length)`, with the opcode implied by the
    // register. See host_uva_layout.hpp for the encoding and rx_scr_armed().
    //
    // IT IS ALSO THE DOORBELL. The word both announces the message and describes it, so the
    // host's side of this leg is ONE strict-ordered UC write rather than an advertisement
    // followed by a separate ring.
    //
    // A ZEROED WORD IS NOT AN INSTRUCTION -- rx_scr_armed() requires the magic and a
    // non-zero length. L1 is not zeroed between sweep points (one process per point), so
    // without that guard a leftover word would decode as a live receive. The X280 side paid
    // 144 phantom transfers for exactly this omission.
    constexpr uint32_t dest_word_addr = get_compile_time_arg_val(5);

    // RUNTIME, NOT COMPILE-TIME, and this is the one that would silently corrupt.
    //
    // Each core owns its own H2DSocket, and each socket's config buffer is a separate
    // MeshBuffer allocation -- so the addresses DIFFER per
    // core even though every core runs identical code. Baking one into a compile arg would
    // point all 109 cores at one core's socket: every core would decode the same ring, race
    // each other's read_ptr, and report success while delivering garbage. The host passes
    // each core its own.
    const uint32_t socket_config_addr = get_arg_val<uint32_t>(0);
    // NOT A MESSAGE COUNT -- 0 means "this core receives nothing", anything else means
    // "receive until stopped". An earlier version took the expected count and looped that
    // many times, which failed on the first real run: the host delivered 130 messages for a
    // 128-message configuration (two duplicate control words got past the sequence filter),
    // the receiver had already exited after its 16th, and the host waited forever for a
    // doorbell nobody would ring -- `core 3 doorbell stuck at 16, expected 17`.
    //
    // The duplicates are a property of the protocol this kernel sits under, not of the
    // socket, and on the push path they are harmless: the host rewrites the same bytes and
    // the doorbell simply runs ahead. A receiver whose correctness depends on the exact
    // number of messages it will be handed is the wrong shape for a data path, so it is not
    // told a number.
    const uint32_t enabled = get_arg_val<uint32_t>(1);

    // A core with no traffic is still launched, so the L1 map is identical on every core --
    // the same reason t6_host_post.cpp launches receivers with zero iterations rather than
    // branching around program creation. Returning before touching the socket is safe;
    // returning after would arm a ring nobody writes.
    if (enabled == 0) {
        return;
    }

    SocketReceiverInterface socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(socket, page_size);

    // In DEVICE_PULL the ring is pinned HOST memory, so fifo_addr is a logical anchor and
    // read_ptr is an offset against it -- not an L1 address. (read_ptr - fifo_addr) is the
    // byte offset into the host buffer; adding it to the published PCIe base is the whole
    // address translation. Getting this backwards reads the right number of bytes from the
    // wrong place and verifies as garbage, which is why it is spelled out rather than
    // folded into the call.
    const uint64_t pcie_data_addr =
        (static_cast<uint64_t>(socket.h2d.data_addr_hi) << 32) | static_cast<uint64_t>(socket.h2d.data_addr_lo);
    const uint32_t pcie_xy_enc = socket.h2d.pcie_xy_enc;

    volatile tt_l1_ptr uint32_t* signal = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_addr);
    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr);

    // How many socket polls to make before looking at the stop flag again. Large enough
    // that the check is nowhere near the hot path, small enough that shutdown is prompt.
    constexpr uint32_t kPollsPerStopCheck = 4096;

    uint32_t i = 0;
    while (stop[0] == 0) {
        // === THE RECEIVE STATUS CONTROL REGISTER =====================================
        //
        // THIS WORD IS THE DOORBELL AND THE INSTRUCTION AT ONCE. The host writes one
        // strict-ordered 8-byte UC word saying "a message is waiting, at `offset`, `length`
        // bytes" -- so the leg that used to take two writes (advertise, then ring
        // rdma_signal) takes one. The opcode is implied: there is exactly one thing a
        // receive register can mean.
        //
        // ONE NUMBER, TWO MEANINGS. The host arena is an exact mirror of this L1
        // (kArenaBytes == l1_size_per_core() == 0x180000), and the sender placed the bytes
        // in it at the offset they are to occupy here. So `offset` is both where to read in
        // the ring and where to write in L1, and this read needs no address arithmetic at
        // all.
        //
        // CHECKED BEFORE THE SOCKET, and cheaply: it is a local L1 load, where
        // socket_wait_for_pages spins on a word the host updates over PCIe. A run that never
        // issues a store never arms this and falls straight through to the legacy path.
        //
        // invalidate_l1_cache() because the host wrote it over PCIe and this core's L1 cache
        // has no idea.
        invalidate_l1_cache();
        const uint64_t scr = *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(dest_word_addr);
        if (tt::tt_metal::experimental::rx_scr_armed(scr)) {
            const uint32_t off = tt::tt_metal::experimental::rx_scr_offset(scr);
            const uint32_t len = tt::tt_metal::experimental::rx_scr_length(scr);
            noc_read_page_chunked(pcie_xy_enc, pcie_data_addr + off, off, len);
            // BEFORE the SCR is cleared and before the signal. The read is asynchronous, and
            // releasing either while it is in flight advertises bytes that have not landed.
            noc_async_read_barrier();
            // THE CONSUMER ZEROES IT. Non-zero means armed, zero means idle -- the same rule
            // the host applies to ctrl_tx/ctrl_rx, and the reason this instruction needs no
            // sequence number: freshness is a property of the word, not of remembered state.
            *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(dest_word_addr) = 0;
            // The completion, in the OTHER direction: "I have taken it". The SCR is the
            // host's doorbell to us; this is ours back to the host, and wait_delivered polls
            // it. Counted here rather than from a loop bound so an extra message increments
            // it like any other.
            *signal = ++i;
            continue;
        }
        // === end receive SCR =========================================================

        // The early-exit form (socket_api.h:273) returns false instead of spinning forever,
        // which is what makes the stop flag reachable at all: a plain socket_wait_for_pages
        // on an idle socket never comes back, and the kernel would outlive the run.
        if (!socket_wait_for_pages(socket, 1, kPollsPerStopCheck)) {
            continue;
        }

        const uint32_t dst_addr = dst_l1_addr;

        if constexpr (pull_from_host) {
            noc_read_page_chunked(
                pcie_xy_enc, pcie_data_addr + socket.read_ptr - socket.fifo_addr, dst_addr, page_size);
            // BEFORE the doorbell and before pop. ringing rdma_signal or returning credit while it is
	    // still in flight advertises bytes that have not landed -- the device-side mirror of the
	    // WC/UC ordering argument host_deliver.hpp makes for the push path.
            noc_async_read_barrier();
        } else {
            // HOST_PUSH: the host already wrote these bytes into this core's L1 FIFO, so
            // read_ptr IS an L1 address (not an offset, as it is in DEVICE_PULL) and there
            // is no PCIe read to do. The copy exists so the payload ends up at the same
            // dst_l1_addr in both modes and one verifier covers both; a real consumer would
            // read it in place and skip this.
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket.read_ptr);
            volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
            for (uint32_t w = 0; w < page_size / sizeof(uint32_t); ++w) {
                dst[w] = src[w];
            }
        }

        socket_pop_pages(socket, 1);
        // The credit. Without it the host's reserve_bytes() never unblocks and its
        // destructor's barrier(1000) times out -- the exact failure this kernel exists to
        // end.
        socket_notify_sender(socket);

        // A MONOTONIC PER-CORE COUNT, matching what the push path's ring_doorbell() writes:
        // the value after the i-th message is i+1, so a host or kernel pacing on it sees the
        // same sequence whichever side delivered the bytes. Counted here rather than taken
        // from a loop bound, so an extra message increments it like any other.
        *signal = ++i;
    }

    // Publish read_ptr/bytes_acked back into the config buffer so a host reading it sees a
    // consistent ring, and flush the outstanding credit writes before the kernel exits.
    update_socket_config(socket);
    noc_async_write_barrier();
}
