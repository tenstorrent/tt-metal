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

#include <tt-metalium/experimental/sockets/internal/host_uva_layout.hpp>

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
    // THE RECEIVE STATUS CONTROL REGISTER. dst_l1_addr above is a compile arg -- one address
    // for the whole run, which is all kOpSendUva needs. A store names its own address per
    // message, which no compile arg can carry, so the host writes `(offset, length)` into this
    // L1 word with the opcode implied by the register. See rx_scr_armed() in host_uva_layout.hpp.
    //
    // It is also the DOORBELL, so the host's side is one strict-ordered UC write rather than an
    // advertisement plus a ring. A ZEROED WORD IS NOT AN INSTRUCTION -- rx_scr_armed() requires
    // the magic and a non-zero length, because L1 is not zeroed between sweep points.
    constexpr uint32_t dest_word_addr = get_compile_time_arg_val(5);

    // RUNTIME, NOT COMPILE-TIME -- the one that would silently corrupt. Each core owns its own
    // H2DSocket and each config buffer is a separate MeshBuffer allocation, so the addresses
    // differ per core even though every core runs identical code. Baking one into a compile arg
    // would point every core at one core's socket: all decoding the same ring, racing each
    // other's read_ptr, reporting success while delivering garbage.
    const uint32_t socket_config_addr = get_arg_val<uint32_t>(0);
    // NOT A MESSAGE COUNT -- 0 means "this core receives nothing", anything else "receive until
    // stopped". A receiver whose correctness depends on the exact number of messages it will
    // be handed is the wrong shape for a data path, so it is not told a number.
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

    // In DEVICE_PULL the ring is pinned HOST memory: fifo_addr is a logical anchor and read_ptr
    // an offset against it, not an L1 address. (read_ptr - fifo_addr) plus the published PCIe
    // base is the whole translation. Backwards, it reads the right bytes from the wrong place.
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
        // ONE NUMBER, TWO MEANINGS: the host arena mirrors this L1 exactly, so `offset` is both
        // where to read in the ring and where to write in L1. Checked first because it is a
        // cheap local load. invalidate_l1_cache() because the host wrote the word over PCIe.
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
            // host's doorbell to us; this is ours back to the host. Counted here rather than
            // from a loop bound so an extra message increments it like any other.
            // Nothing polls this word: wait_delivered() polls bytes_acked in pinned host RAM
            // instead, to avoid a non-posted PCIe read contending with the payload reads.
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
            // HOST_PUSH: read_ptr IS an L1 address here, so there is no PCIe read. The copy
            // only exists so both modes land at the same dst_l1_addr for one verifier.
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
