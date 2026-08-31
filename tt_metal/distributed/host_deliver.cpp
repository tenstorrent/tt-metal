// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "host_deliver.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <ctime>
#include <mutex>
#include <sstream>

#include "host_uva_layout.hpp"
#include "host_region.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/cluster_noc_helpers.hpp>

#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
// Locating the socket's own pinned ring so the peer can RMA straight into it.
#include <cerrno>
#include <cstddef>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "tt_metal/distributed/hd_socket_descriptor.hpp"
// receiver_socket_md -- its first field is bytes_sent, the word notify_receiver() writes.
#include "tt_metal/hw/inc/hostdev/socket.h"

namespace tt::tt_metal::experimental {


namespace {

uint64_t now_ns_local() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

// A HOST->DEVICE WRITE FALLS OFF A CLIFF BETWEEN 32 KiB AND 64 KiB, so the payload write is
// chunked rather than issued whole. This is a property of the write primitive underneath
// noc_write(), not of anything here, and it is measured twice by two independent programs:
//
//   bytes     mmio_bench --h2d "write ns"      t6_host_uva diag:l1-write (verbs 1c sweep)
//   32768     6,525 ns      = 5.02 GB/s        7,231 ns min  = 4.53 GB/s
//   65536     29,912 ns     = 2.19 GB/s        79,930 ns min = 0.82 GB/s
//   1048576   3,307,132 ns  = 0.32 GB/s        3,302,615 ns min = 0.32 GB/s
//
// So 1 MiB whole costs 3.31 ms, and 1 MiB as 32 x 32 KiB costs ~32 x 6.5 us = 209 us.
//
// IT IS NOT THE DMA THRESHOLD, which is the first thing everyone reaches for and which the
// comment here used to claim. Cluster::supports_dma_operations() (tt_cluster.cpp:824)
// requires arch_ == WORMHOLE_B0, so on Blackhole it is FALSE AT EVERY SIZE and there is no
// fast path to fall off -- every noc_write of every length goes through UMD's
// write_to_device(). Whatever changes at 32 KiB is inside that call. Naming a wrong cause
// in a comment is worse than naming none, because it sends the next reader to tt-metal's
// DMA code, where nothing is wrong.
//
// 32768 is the largest size measured on the fast side of the knee, not a guess at where the
// knee is. If a future UMD moves it, the constant is the one thing to re-measure -- run
// `mmio_bench --h2d` and read the "write ns" column.
constexpr uint32_t kMaxHostWriteDefault = 32768;

// OVERRIDABLE AT RUNTIME, because the constant above was chosen from the wrong measurement.
//
// The table in host_deliver.hpp times ONE write of each size. 32768 is the largest size on
// the fast side of that knee -- but the deliverer issues chunks BACK TO BACK, and those two
// things turned out not to be the same. Measured on this tree: a lone 32 KiB write costs
// 6.94 us (4230 MB/s), while 64 KiB as two 32 KiB chunks costs 25.5 us rather than the ~14
// the single-write figure predicts, and 1 MiB as 32 chunks costs 3275 us rather than ~222.
// The header's claim that chunking buys 15x at 1 MiB is an extrapolation from the
// single-write rate; the chunked measurement matches the UNCHUNKED one.
//
// So the composition has never been swept; the chunk size is fixed at the measured default.
uint32_t max_host_write() { return kMaxHostWriteDefault; }

class DeviceDeliverer final : public Deliverer {
public:
    DeviceDeliverer(tt::tt_metal::IDevice* device, uint32_t grid_width, uint32_t cores, L1Layout layout) :
        device_id_(static_cast<uint32_t>(device->id())), layout_(layout), cores_(cores) {
        // Resolve logical -> TRANSLATED once. noc_write takes translated coords; doing the
        // lookup per message would put it in the hot path, and getting it wrong writes to
        // a real tile that is not the intended one.
        virt_.reserve(cores);
        for (uint32_t i = 0; i < cores; ++i) {
            const CoreCoord logical{i % grid_width, i / grid_width};
            virt_.push_back(device->virtual_core_from_logical_core(logical, tt::CoreType::WORKER));
        }
    }

    std::string write_payload(uint32_t core, const uint8_t* src, uint32_t bytes, uint32_t dst_l1) override {
        if (core >= cores_) {
            return "deliver: core index out of range";
        }
        // The effective address, or the layout's fixed one when no store named it.
        const uint32_t dst = (dst_l1 != 0) ? dst_l1 : layout_.payload_addr;
        const auto& v = virt_[core];
        // WC window, relaxed ordering. This is the bulk path and throughput is what
        // matters; ordering against the doorbell is the doorbell's job, not this write's.
        //
        // CHUNKED AT kMaxHostWrite, because one big noc_write is slower than the same
        // bytes in 32 KiB pieces. See the constant.
        //
        uint32_t off = 0;
        while (off < bytes) {
            const uint32_t chunk = std::min(bytes - off, max_host_write());
            tt::tt_metal::distributed::noc_write(
                device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), dst + off,
                std::string_view(reinterpret_cast<const char*>(src + off), chunk));
            off += chunk;
        }
        return {};
    }

    std::string ring_doorbell(uint32_t core, uint32_t value) override {
        if (core >= cores_) {
            return "doorbell: core index out of range";
        }
        const auto& v = virt_[core];
        // STRICT ordering, UC window. This is what makes "doorbell visible implies payload
        // landed" true. A relaxed write here can be combined ahead of the payload it
        // advertises, and the kernel then wakes on a buffer still being filled -- a torn
        // payload with nothing reporting an error.
        tt::tt_metal::distributed::noc_write_immediate(
            device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), layout_.signal_addr,
            std::string_view(reinterpret_cast<const char*>(&value), sizeof(value)));
        return {};
    }

    std::string ring_completion(uint32_t core, uint32_t value) override {
        if (core >= cores_) {
            return "completion: core index out of range";
        }
        const auto& v = virt_[core];
        // Same strict-ordering UC path as the signal. This word releases the kernel to
        // reuse its control register, so it must not be reordered ahead of anything -- and
        // unlike the signal it says nothing about payload, only that the request retired.
        tt::tt_metal::distributed::noc_write_immediate(
            device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), layout_.completion_addr,
            std::string_view(reinterpret_cast<const char*>(&value), sizeof(value)));
        return {};
    }

    uint32_t read_doorbell(uint32_t core) override {
        if (core >= cores_) {
            return 0;
        }
        const auto& v = virt_[core];
        return tt::tt_metal::distributed::noc_read_reg_u32(
            device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), layout_.signal_addr);
    }

    std::vector<uint8_t> read_payload(uint32_t core, uint32_t bytes, uint32_t src_l1) override {
        if (core >= cores_) {
            return {};
        }
        const auto& v = virt_[core];
        // Non-posted PCIe read, 22.5 ns/byte. Verification only -- never on a data path.
        return tt::tt_metal::distributed::noc_read(
            device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y),
            (src_l1 != 0) ? src_l1 : layout_.payload_addr, bytes);
    }

    uint32_t read_reg32(uint32_t core, uint64_t addr) override {
        if (core >= cores_) {
            return 0;
        }
        const auto& v = virt_[core];
        return tt::tt_metal::distributed::noc_read_reg_u32(
            device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), addr);
    }

    std::string describe() const override {
        std::ostringstream o;
        o << "device " << device_id_ << " (noc_write payload chunked at " << max_host_write()
          << " B + noc_write_immediate doorbell, no membar)";
        return o.str();
    }

private:
    uint32_t device_id_;
    L1Layout layout_;
    uint32_t cores_;
    std::vector<CoreCoord> virt_;
};

// ONE H2DSocket PER T6 CORE. The set is built once, in the constructor, on ONE thread --
// construction allocates a device config buffer and pins a host ring per core, and neither
// is something to be doing while workers are running. After that the sockets are only ever
// touched through write_payload(), and WHICH THREAD CALLS IT IS THE CALLER'S CONTRACT: this
// class does not lock, because a lock here would hide exactly the ownership bug it exists
// to make impossible. See the header.
class H2DSocketDeliverer final : public Deliverer {
public:
    H2DSocketDeliverer(
        std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh, uint32_t grid_width, uint32_t cores,
        L1Layout layout, H2DSocketConfig cfg) :
        mesh_(std::move(mesh)), layout_(layout), cores_(cores), cfg_(cfg) {
        device_id_ = static_cast<uint32_t>(mesh_->get_devices()[0]->id());
        pre_acked_.assign(cores, 0);
        const auto mode = cfg_.device_pull ? tt::tt_metal::distributed::H2DMode::DEVICE_PULL
                                           : tt::tt_metal::distributed::H2DMode::HOST_PUSH;
        sockets_.reserve(cores);
        virt_.reserve(cores);
        for (uint32_t i = 0; i < cores; ++i) {
            const CoreCoord logical{i % grid_width, i / grid_width};
            // LOGICAL coords here, unlike DeviceDeliverer. H2DSocket does the
            // logical->virtual translation itself (tt-metal h2d_socket.cpp:305); handing it
            // a translated coord would translate twice and name a different tile. The
            // virtual coords below are for the doorbell and the readback only.
            virt_.push_back(mesh_->get_devices()[0]->virtual_core_from_logical_core(
                logical, tt::CoreType::WORKER));
            sockets_.push_back(std::make_unique<tt::tt_metal::distributed::H2DSocket>(
                mesh_,
                tt::tt_metal::distributed::MeshCoreCoord{
                    tt::tt_metal::distributed::MeshCoordinate(0, 0), logical},
                tt::tt_metal::BufferType::L1,
                cfg_.fifo_size,
                mode));
            sockets_.back()->set_page_size(page_size_for());
        }
        page_size_ = sockets_.empty() ? 0 : sockets_[0]->get_page_size();

        // Ring aliasing, always. alias_on_ latches whether the mapping actually succeeded.
        {
            if (cfg_.alias_region_base == nullptr) {
                std::cerr << "ring aliasing needs a region base and none was given, so there "
                             "is nothing to alias onto. Delivery keeps the RX-arena memcpy.\n";
            } else if (const std::string e = map_rings(); !e.empty()) {
                // FATAL, NOT A FALLBACK -- corrected 2026-08-27.
                //
                // This used to announce and carry on with the memcpy path, on the reasoning
                // that a working measurement beats none. That reasoning was wrong, and the
                // comment sitting here said why in its own next sentence: "a run that
                // silently fell back would be attributed to the aliasing it never used."
                //
                // It is worse than silent. run_point() captures the binary's stdout AND
                // stderr into a shell variable and prints only its own PASS line, so a whole
                // campaign would come back labelled aliased, be filed under the same CSV
                // name as a real one, and there would be no column to tell them apart.
                // Found by sizing the ring one page too large: every aliased run would have
                // fallen back with nothing but a swallowed line to say so.
                //
                alias_error_ = "the rings could not be mapped: " + e;
                std::cerr << alias_error_ << "\n";
            } else {
                alias_on_ = true;
                std::cerr << "H2D ring aliasing ON for " << cores_ << " cores; the peer must "
                             "RMA into the ring and payload must equal fifo_size.\n";
            }
        }
    }

    ~H2DSocketDeliverer() override {
        // BEFORE the sockets are destroyed. ~H2DSocket() unlinks the shm, and unmapping after
        // that is a use of a name that no longer resolves; the mapping itself would survive,
        // but tearing down in the wrong order is how this class would start leaking a mapping
        // per run without anything failing.
        unmap_rings();
    }

    // The socket is page-granular and set_page_size() is a STATEFUL operation both sides
    // must perform in lockstep (tt-metal h2d_socket.cpp:685 and socket_api.h:246), so it
    // cannot be re-set per message. Defaulting to the PCIe alignment buys the finest legal
    // grain, which is what a variable-length sweep needs; a caller who knows its shard size
    // should say so and waste fewer tail bytes.
    //
    // ASKED FOR, NOT PROBED, AND NOT HARD-CODED. An earlier version called set_page_size()
    // in a loop from 4 B upward and caught the throw -- which works, and emits a TT_FATAL
    // log line per rejected size per socket: 4 x 109 = 436 lines of "Page size must be
    // PCIE-aligned" in front of a run that succeeded. A probe whose failures are logged by
    // the callee is not a quiet probe. hal::get_pcie_alignment() is public API and is the
    // same value h2d_socket.cpp:378 uses to validate, so this cannot disagree with it.
    uint32_t page_size_for() const {
        if (cfg_.page_size != 0) {
            return cfg_.page_size;
        }
        return tt::tt_metal::hal::get_pcie_alignment();
    }

    std::string write_payload(uint32_t core, const uint8_t* src, uint32_t bytes, uint32_t dst_l1) override {
        if (core >= cores_) {
            return "h2d-socket: core index out of range";
        }
        if (page_size_ == 0 || bytes % page_size_ != 0) {
            return "h2d-socket: payload is not a multiple of the socket page size";
        }

        // THE DESTINATION HANDOFF, AND IT MUST PRECEDE THE ADVERTISEMENT.
        //
        // On this path the host does not write the payload into L1 -- the receiver kernel
        // does, after it sees the page. So the effective address has to reach the kernel
        // before the page does. Both writes go out on the strict-ordered UC path to the same
        // tile, and the socket's bytes_sent update is what releases the kernel, so writing
        // this first is the whole ordering argument. Reversed, the kernel would write this
        // message's payload to the PREVIOUS message's address, with nothing reporting it.
        //
        // Refused rather than ignored when the layout has nowhere to put it: a store that
        // silently landed at the kernel's compile-time address would look like the feature
        // working. See kernels/t6_host_pull.cpp.
        if (dst_l1 != 0) {
            if (layout_.dest_word_addr == 0) {
                return "h2d-socket: this build has no receive-SCR word in its L1 layout, so a "
                       "store cannot be handed to the receiver kernel";
            }
            // A STORE ON THE SOCKET PATH REQUIRES THE RING TO BE THE ARENA.
            //
            // The bytes have to already sit at `dst_l1` within the ring, because that offset
            // is what the receive SCR names and what the core will read. Without aliasing the
            // host would have to memcpy them there -- and H2DSocket::write() places at the
            // ring's own write pointer, not at an offset we choose, so there is no call that
            // does it. Refused rather than approximated.
            if (!alias_on_) {
                return "h2d-socket: a store needs the aliased ring, which could not be mapped. "
                       "Unaliased, the payload is memcpy'd to the ring's write pointer and cannot "
                       "be placed at the destination offset the "
                       "receive SCR names.";
            }
            // THE RECEIVE INSTRUCTION. One 8-byte strict-ordered UC write carrying the offset
            // and the length -- the opcode is implied by the register. It must precede
            // anything that releases the core, which on this path it does: nothing else is
            // written after it, and the core is polling this very word.
            const uint64_t scr = rx_scr_encode(dst_l1, bytes);
            const auto& v = virt_[core];
            tt::tt_metal::distributed::noc_write_immediate(
                device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), layout_.dest_word_addr,
                std::string_view(reinterpret_cast<const char*>(&scr), sizeof(scr)));
            // NOTHING ELSE TO DO. The bytes are already in the ring (the peer RMA'd them
            // there), and the core finds them from the SCR rather than from bytes_sent. The
            // socket's page bookkeeping is not in the addressing path at all here.
            return {};
        }

        // ONE write() PER MESSAGE, not one per page: write() takes a page COUNT and issues a
        // single memcpy (DEVICE_PULL) or one TLB write (HOST_PUSH) plus exactly one
        // bytes_sent update, whatever the count. Looping per page here would multiply the
        // doorbell by the page count and measure a bookkeeping artefact.
        // SNAPSHOT BEFORE THE WRITE, so wait_delivered() has a reference point.
        //
        // Absolute arithmetic (expected * page_size) would be WRONG: push_bytes() and the
        // device's socket_pop_pages() both add `fifo_size - fifo_curr_size` when a message
        // wraps the ring, so bytes_acked does not advance by exactly one page per message.
        // A delta does not care -- the wrap adjustment only ever ADDS.
        //
        // Safe because there is AT MOST ONE MESSAGE IN FLIGHT PER CORE: deliver_to_l1 calls
        // write_payload -> ring_doorbell -> wait_delivered synchronously, and a core's socket
        // is only ever touched by its owning thread. Pipeline this path and the delta stops
        // being attributable and this has to become a real sequence number.
        pre_acked_[core] = sockets_[core]->bytes_acked_snapshot();

        // === ALIASED PATH — the bytes are ALREADY here ==================================
        //
        // The peer RMA'd this payload straight into the ring, so there is nothing to copy;
        // all that is owed is the notification tt-metal's write() would have sent after its
        // memcpy. This is the entire point of the exercise: on this branch the payload
        // crosses host RAM exactly once, written by the NIC, and is then read by the device.
        //
        // THE FIXED-OFFSET REQUIREMENT IS CHECKED, NOT TRUSTED. The peer writes to a fixed
        // offset (the RX arena base), so the ring's write pointer must be back at 0 every
        // time. h2d_socket.cpp:663-667 wraps write_ptr_ to 0 only when a message exactly
        // fills the ring -- `write_ptr_ + num_bytes >= fifo_curr_size_` taking the equality
        // case, leaving 0 + N - N = 0. A SHORT message advances the pointer without wrapping,
        // and every subsequent RMA would then land at the wrong place while still reporting
        // success. So a payload that is not exactly the ring size is refused by name here
        // rather than silently corrupting the stream.
        if (alias_on_) {
            if (ring_[core] == nullptr) {
                return "h2d-alias: ring for this core was never mapped";
            }
            // ONLY WHEN THE SOCKET'S POINTER IS THE ADDRESSING MECHANISM. A store names its
            // offset in the receive SCR and returns above, so it never reaches here; this
            // guard belongs to the kOpSendUva path, where the device locates the bytes via
            // read_ptr and the write pointer must therefore be back at 0.
            if (bytes != cfg_.fifo_size) {
                std::ostringstream o;
                o << "h2d-alias: payload " << bytes << " B must equal fifo_size "
                  << cfg_.fifo_size << " B -- a short message leaves the ring's write pointer "
                     "off zero and the peer's next RMA lands at the wrong offset. Size the "
                     "socket to the run's payload.";
                return o.str();
            }
            // src is expected to BE the ring (the RX arena aliased onto it). Until the
            // region-side aliasing lands, it will not be -- so this stays a check rather than
            // an assumption, and names the mismatch instead of committing bytes that are
            // somewhere else entirely.
            if (src != ring_[core]) {
                return "h2d-alias: the source is not this core's ring; the RX arena has not "
                       "been aliased onto it, so the payload is not where the device will look";
            }
            return commit_to_device(core, bytes);
        }
        // === end aliased path ===========================================================

        try {
            sockets_[core]->write(const_cast<uint8_t*>(src), bytes / page_size_);
        } catch (const std::exception& e) {
            return std::string("h2d-socket: write: ") + e.what();
        }
        return {};
    }

    std::string ring_doorbell(uint32_t core, uint32_t value) override {
        if (core >= cores_) {
            return "doorbell: core index out of range";
        }
        // write() already advertised the payload by updating bytes_sent in the core's L1
        // config buffer. When the kernel wakes on the socket that IS the doorbell and a
        // second MMIO write is pure cost; when the kernel still needs an out-of-band length
        // it is not. The flag decides, and the bench reports both.
        if (cfg_.socket_is_the_doorbell) {
            return {};
        }
        const auto& v = virt_[core];
        tt::tt_metal::distributed::noc_write_immediate(
            device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), layout_.signal_addr,
            std::string_view(reinterpret_cast<const char*>(&value), sizeof(value)));
        return {};
    }

    std::string ring_completion(uint32_t core, uint32_t value) override {
        if (core >= cores_) {
            return "completion: core index out of range";
        }
        // NOT a socket concern. rdma_completion says "the request YOU issued retired" and is
        // rung by this host after servicing the core's TX word -- it never travels the H2D
        // data path, so it keeps the strict-ordering UC write whatever the payload does.
        const auto& v = virt_[core];
        tt::tt_metal::distributed::noc_write_immediate(
            device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), layout_.completion_addr,
            std::string_view(reinterpret_cast<const char*>(&value), sizeof(value)));
        return {};
    }

    uint32_t read_doorbell(uint32_t core) override {
        if (core >= cores_) {
            return 0;
        }
        const auto& v = virt_[core];
        return tt::tt_metal::distributed::noc_read_reg_u32(
            device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), layout_.signal_addr);
    }

    // VERIFICATION ONLY, AND IT MEANS NOTHING IN DEVICE_PULL. The payload is in pinned host
    // RAM and stays there until a receiver kernel pulls it, so this reads an L1 that nothing
    // has written. The bench refuses to claim a DEVICE_PULL pass on it.
    std::vector<uint8_t> read_payload(uint32_t core, uint32_t bytes, uint32_t src_l1) override {
        if (core >= cores_) {
            return {};
        }
        const auto& v = virt_[core];
        return tt::tt_metal::distributed::noc_read(
            device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y),
            (src_l1 != 0) ? src_l1 : layout_.payload_addr, bytes);
    }

    uint32_t read_reg32(uint32_t core, uint64_t addr) override {
        if (core >= cores_) {
            return 0;
        }
        const auto& v = virt_[core];
        return tt::tt_metal::distributed::noc_read_reg_u32(
            device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), addr);
    }

    // The receiver kernel writes i+1 into signal_addr after its pull retires, so this
    // returns exactly when the payload is readable in L1 -- the same instant write_payload()
    // returns on the push path. A spin rather than a sleep: the wait is microseconds and a
    // scheduler round trip would cost more than the poll.
    std::string wait_delivered(uint32_t core, uint32_t expected) override {
        if (core >= cores_) {
            return "h2d-socket: core index out of range";
        }
        // Bounded, because an unbounded spin here is the same failure the no-receiver mode
        // used to produce: silence with no output. 5 s is far beyond any real delivery and
        // far below a human's patience for a hung sweep.
        constexpr uint64_t kTimeoutNs = 5ull * 1000 * 1000 * 1000;
        const uint64_t t0 = now_ns_local();
        uint32_t spins = 0;
        // POLL OUR OWN MEMORY, NOT THE DEVICE'S.
        //
        // The receiver posts bytes_acked into PINNED HOST RAM after its read barrier
        // (socket_api.h socket_notify_sender), so this says exactly what the L1 doorbell
        // says -- the payload is in L1 -- with a local load instead of a non-posted PCIe
        // read. Polling device L1 costs ~1 us a sample and contends with the very payload
        // reads it waits for: the protocol's invariant is that each side writes to the
        // other's memory and polls only its OWN, and the doorbell poll was the one place
        // this path broke it.
        //
        // Unsigned delta, so the counter's own 32-bit wrap costs nothing to handle.
        const uint32_t before = pre_acked_[core];
        while (static_cast<uint32_t>(sockets_[core]->bytes_acked_snapshot() - before) < page_size_) {
            // BACK OFF BETWEEN READS, BECAUSE THE READ IS NOT FREE AND NOT LOCAL.
            //
            // read_doorbell() is a non-posted PCIe read over the SAME link the
            // receiver kernels are pulling payloads across. A tight poll from every worker
            // thread contends with the transfers it is waiting for, inflating both the stage
            // it is timing and its variance.
	    // 
            // The first read still happens immediately, so an already-delivered message costs
            // exactly one read. Only a genuine wait pays the backoff.
            for (uint32_t k = 0, n = 1u << (spins < 10 ? spins : 10); k < n; ++k) {
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
            }
            if (spins < 10) {
                ++spins;
            }
            if (now_ns_local() - t0 > kTimeoutNs) {
                std::ostringstream o;
                o << "h2d-socket: core " << core << " delivery stalled -- the receiver kernel is "
                     "not running, not draining, or was given the wrong socket config address ("
                  << "bytes_acked " << sockets_[core]->bytes_acked_snapshot() << ", was " << before
                  << ", needed +" << page_size_ << ")";
                return o.str();
            }
        }
        (void)expected;
        return {};
    }

    std::string arm_receivers() override {
        if (layout_.stop_addr == 0) {
            return "h2d-socket: no stop address configured; receiver kernels cannot be armed";
        }
        const uint32_t zero = 0;
        for (uint32_t c = 0; c < cores_; ++c) {
            const auto& v = virt_[c];
            tt::tt_metal::distributed::noc_write_immediate(
                device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), layout_.stop_addr,
                std::string_view(reinterpret_cast<const char*>(&zero), sizeof(zero)));
        }
        return {};
    }

    std::string stop_receivers() override {
        if (layout_.stop_addr == 0) {
            return "h2d-socket: no stop address configured; the receiver kernels cannot be told to exit";
        }
        // Strict-ordering UC, like the doorbells: this word releases a kernel, so it must not
        // sit in a write-combining buffer while the host goes on to wait for that kernel.
        const uint32_t one = 1;
        for (uint32_t c = 0; c < cores_; ++c) {
            const auto& v = virt_[c];
            tt::tt_metal::distributed::noc_write_immediate(
                device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), layout_.stop_addr,
                std::string_view(reinterpret_cast<const char*>(&one), sizeof(one)));
        }
        return {};
    }

    std::vector<uint32_t> socket_config_addresses() const override {
        std::vector<uint32_t> out;
        out.reserve(sockets_.size());
        for (const auto& s : sockets_) {
            out.push_back(s->get_config_buffer_address());
        }
        return out;
    }

    std::string describe() const override {
        std::ostringstream o;
        o << "device " << device_id_ << " (" << cores_ << " x H2DSocket "
          << (cfg_.device_pull ? "DEVICE_PULL" : "HOST_PUSH") << ", fifo " << cfg_.fifo_size << " B, page "
          << page_size_ << " B, doorbell " << (cfg_.socket_is_the_doorbell ? "socket" : "separate UC") << ")";
        return o.str();
    }

    uint32_t page_size() const { return page_size_; }

    // Maps each socket's own pinned host ring into THIS process a second time, so the ring's
    // bytes can be reached by an address we control and registered with libfabric. The peer
    // then RMAs its payload straight into the ring and the RX-arena memcpy stops existing.
    //
    // NOTHING IS SUBSTITUTED, ONLY ALIASED. The device pulls from a NOC address tt-metal
    // derived when it pinned these pages (h2d_socket.cpp:108-117), so the pages must stay
    // exactly the ones the socket allocated. A second mapping of the same shm object gives a
    // second virtual address for the same physical pages; pointing the socket at different
    // memory instead would leave the host writing one buffer while the device reads another,
    // with nothing reporting it.
    //

    // Returns "" on success; on failure the aliasing is left off and the caller keeps the
    // memcpy path, because a half-mapped set is worse than no mapping at all.
    std::string map_rings() {
        // ORDERING, ENFORCED RATHER THAN COMMENTED. provision() pins the region
        // (host_region.cpp, PinnedMemory::Create), and Transport::connect() registers it with
        // fi_mr_reg. Both capture the PHYSICAL pages behind these addresses. MAP_FIXED after
        // either one swaps different pages in while the pin and the MR go on naming the old
        // ones -- so the NIC DMAs into memory that is no longer mapped here and the device
        // pulls from pages nothing is writing. There is no error anywhere in that sequence;
        // it surfaces as wrong payload bytes, if it surfaces at all. Hence a refusal.
        if (HostRegion::is_provisioned()) {
            return "h2d-alias: the region is already provisioned (and therefore pinned), so "
                   "overlaying the rings now would swap pages out from under the pin and the "
                   "MR. The deliverer must be constructed BEFORE HostRegion::provision().";
        }
        ring_.assign(cores_, nullptr);
        ring_bytes_.assign(cores_, 0);
        cfg_addr_.assign(cores_, 0);
        sent_.assign(cores_, 0);
        for (uint32_t c = 0; c < cores_; ++c) {
            const auto d = sockets_[c]->populate_descriptor();
            // shm_size comes off the descriptor rather than fstat so that a layout change
            // upstream disagrees with us here rather than silently mapping a different span.
            // THE OVERLAY TARGET: this core's RX arena, which is where the peer already
            // writes. The region is sharded per core already, and there is one socket per
            // core, so the correspondence is 1:1 and needs no new addressing scheme.
            uint8_t* const slot = cfg_.alias_region_base + rx_arena_offset(c);

            // BOUNDS, BECAUSE mmap ROUNDS THE LENGTH UP TO A PAGE. An arena slot is
            // kArenaBytes and the next thing after it is the FOLLOWING core's TX arena, so a
            // shm_size within a page of the slot size would overlay memory belonging to a
            // different core -- and a Tensix pushing into a TX arena that has been silently
            // replaced is corruption with no error anywhere. Refused rather than clamped: a
            // clamp would map less than the ring and leave the tail landing off the end.
            const size_t rounded = (d.shm_size + kPageBytes - 1) & ~(static_cast<size_t>(kPageBytes) - 1);
            if (rounded > kArenaBytes) {
                unmap_rings();
                std::ostringstream o;
                o << "h2d-alias: core " << c << " ring is " << d.shm_size << " B (" << rounded
                  << " B once page-rounded) but an RX arena slot is only " << kArenaBytes
                  << " B; the overlay would reach into the next core's TX arena";
                return o.str();
            }

            const int fd = ::shm_open(d.shm_name.c_str(), O_RDWR, 0);
            if (fd < 0) {
                unmap_rings();
                return "h2d-alias: shm_open(" + d.shm_name + ") failed: " + std::strerror(errno);
            }
            // MAP_FIXED, deliberately: the point is to put these pages at an address the peer
            // is ALREADY writing to. It silently replaces whatever was mapped there, which is
            // exactly why this must run before the region is pinned and registered.
            void* p = ::mmap(slot, d.shm_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
            ::close(fd);  // the mapping holds its own reference; the fd is not needed after
            if (p == MAP_FAILED) {
                unmap_rings();
                return "h2d-alias: mmap(" + d.shm_name + ") failed: " + std::strerror(errno);
            }
            if (p != slot) {
                unmap_rings();
                return "h2d-alias: MAP_FIXED did not honour the requested address";
            }
            ring_data_offset_ = d.data_offset;
            ring_[c] = static_cast<uint8_t*>(p) + d.data_offset;
            ring_bytes_[c] = d.shm_size;
            cfg_addr_[c] = d.config_buffer_address;

            // Without this the overlay is only half
            // done: provision() would still zero, and reset_banks_and_arenas() would still
            // complement-fill, every byte of this arena -- including the ring's bytes_acked
            // (at data_offset + fifo_size, h2d_socket.cpp:90) and the connector state behind
            // it.
	    //
            // fifo_size comes off the DESCRIPTOR, not off cfg_.fifo_size: the socket is the
            // authority on the size it actually built, and a disagreement between the two is
            // something to inherit rather than to paper over.
            HostRegion::declare_rx_alias(
                c, static_cast<uint64_t>(d.data_offset) + d.fifo_size, static_cast<uint64_t>(rounded));
        }
        return {};
    }

    void unmap_rings() {
        // The region's declarations go first, and unconditionally: this runs both from
        // map_rings()'s rollback paths, where a half-declared set would leave provision()
        // skipping bytes nothing is mapped over, and from the destructor, where it is a
        // no-op that costs nothing. clear_rx_aliases() is the one alias call that does not
        // refuse after provisioning, for exactly this second case.
        HostRegion::clear_rx_aliases();
        for (uint32_t c = 0; c < ring_.size(); ++c) {
            if (ring_[c] != nullptr && ring_bytes_[c] != 0) {
                // ring_ was offset by data_offset; unmap from the mapping base. data_offset is
                // 0 in every current layout, but reading it and then ignoring it here would be
                // the kind of asymmetry that survives until the day it is not 0.
                uint8_t* const slot = ring_[c] - ring_data_offset_;
                // PUT SOMETHING BACK, rather than leaving a hole in the middle of the region.
                // MAP_FIXED replaced the static array's pages here; a plain munmap would leave
                // that span unmapped, and the region is a static array that other things --
                // ~PinnedMemory unpinning the range it was given, a late scan, a verifier --
                // still address as one contiguous object. An anonymous MAP_FIXED both drops
                // the shm reference and restores a valid (zeroed, unpinned) mapping in one
                // step, so teardown order stops being able to fault.
                void* const restored = ::mmap(
                    slot, ring_bytes_[c], PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
                if (restored == MAP_FAILED) {
                    // Fall back to the plain unmap. Worse, but the alternative is leaving the
                    // shm mapped and the socket's pages alive past the socket.
                    ::munmap(slot, ring_bytes_[c]);
                }
            }
            ring_[c] = nullptr;
            ring_bytes_[c] = 0;
        }
    }

    // notify_receiver(), reimplemented. tt-metal's own is h2d_socket.cpp:
    //     bytes_sent_addr = config_buffer_address_ + offsetof(receiver_socket_md, bytes_sent);
    //     pcie_writer(&bytes_sent_, 4, bytes_sent_addr); sfence();
    // bytes_sent is the FIRST field of receiver_socket_md (tt_metal/hw/inc/hostdev/socket.h),
    // so the offset is 0. Asserted rather than assumed: a field inserted ahead of it upstream
    // would otherwise advance the wrong word and stall every receiver with no error anywhere.
    std::string commit_to_device(uint32_t core, uint32_t bytes) {
        static_assert(offsetof(receiver_socket_md, bytes_sent) == 0,
                      "bytes_sent moved within receiver_socket_md; this write targets the wrong word");
        sent_[core] += bytes;
        const auto& v = virt_[core];
        tt::tt_metal::distributed::noc_write_immediate(
            device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y),
            cfg_addr_[core] + offsetof(receiver_socket_md, bytes_sent),
            std::string_view(reinterpret_cast<const char*>(&sent_[core]), sizeof(uint32_t)));
        return {};
    }

private:
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_;
    L1Layout layout_;
    uint32_t cores_;
    H2DSocketConfig cfg_;
    uint32_t device_id_ = 0;
    uint32_t page_size_ = 0;
    std::vector<CoreCoord> virt_;
    std::vector<std::unique_ptr<tt::tt_metal::distributed::H2DSocket>> sockets_;
    // Per core, unsynchronised for the same reason nothing else here is: a core's socket
    // belongs to one thread, which is the contract H2DSocket::write itself rests on.
    mutable std::vector<uint32_t> pre_acked_;
    // Ring aliasing state. alias_on_ is latched once in the constructor rather than read from
    // the environment per message, so a variable changed mid-run cannot split one run's
    // behaviour across two paths.
    bool alias_on_ = false;
    // Non-empty means the run was asked for aliasing and could not have it. Read by the
    // factory, which refuses to hand back a deliverer rather than let the run proceed on a
    // path the operator did not ask for.
    std::string alias_error_;
    uint32_t ring_data_offset_ = 0;
    std::vector<uint8_t*> ring_;
    std::vector<size_t> ring_bytes_;
    std::vector<uint32_t> cfg_addr_;
    std::vector<uint32_t> sent_;
};

}  // namespace

std::unique_ptr<Deliverer> make_device_deliverer(
    tt::tt_metal::IDevice* device, uint32_t grid_width, uint32_t cores, L1Layout layout, std::string& error) {
    error.clear();
    if (device == nullptr) {
        error = "no device";
        return nullptr;
    }
    return std::make_unique<DeviceDeliverer>(device, grid_width, cores, layout);
}

std::unique_ptr<Deliverer> make_h2d_socket_deliverer(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device, uint32_t grid_width, uint32_t cores,
    L1Layout layout, H2DSocketConfig cfg, std::string& error) {
    error.clear();
    if (mesh_device == nullptr) {
        error = "no mesh device";
        return nullptr;
    }
    // A HOST_PUSH socket allocates a grid-wide sharded
    // FIFO per socket, so `cores` of them need cores x fifo_size of L1 on every core. The
    // allocator would fail somewhere inside the loop with a message about a buffer, not
    // about this decision, so the arithmetic is done here where the cause is legible.
    if (!cfg.device_pull) {
        const uint64_t per_core = static_cast<uint64_t>(cores) * cfg.fifo_size;
        constexpr uint64_t kUsableL1 = 1427ull << 10;  // 1.5 MiB less the allocator base
        if (per_core > kUsableL1) {
            std::ostringstream o;
            o << "HOST_PUSH with one socket per core needs " << (per_core >> 20) << " MiB of L1 on EVERY core ("
              << cores << " sockets x " << (cfg.fifo_size >> 10) << " KiB) against ~1,427 KiB usable. "
              << "Use DEVICE_PULL, or lower --cores/--fifo.";
            error = o.str();
            return nullptr;
        }
    }

    if (cfg.page_size != 0) {
        const uint32_t align = tt::tt_metal::hal::get_pcie_alignment();
        if (cfg.page_size < align || cfg.page_size % align != 0) {
            std::ostringstream o;
            o << "H2D socket page size " << cfg.page_size << " B is not a multiple of the "
              << align << " B PCIe alignment. One page per message means the page size IS the "
              << "message size, so this run's --bytes must be a multiple of " << align
              << " (the smallest usable size is " << align << ").";
            error = o.str();
            return nullptr;
        }
    }
    try {
        return std::make_unique<H2DSocketDeliverer>(std::move(mesh_device), grid_width, cores, layout, cfg);
    } catch (const std::exception& e) {
        error = std::string("H2DSocket construction failed: ") + e.what();
        return nullptr;
    }
}

double measure_ns_per_cycle(Deliverer& deliverer, uint32_t core, uint32_t sample_ms, std::string& detail) {
    detail.clear();
    auto sample = [&]() -> uint64_t {
        // LOW first: reading the low half latches the high half for readback, so the
        // other order can pair a fresh low with a stale high across a rollover.
        const uint32_t lo = deliverer.read_reg32(core, kWallClockLo);
        const uint32_t hi = deliverer.read_reg32(core, kWallClockHi);
        return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    };

    const uint64_t c0 = sample();
    const uint64_t h0 = now_ns_local();
    timespec ts{sample_ms / 1000, static_cast<long>((sample_ms % 1000) * 1000000L)};
    nanosleep(&ts, nullptr);
    const uint64_t c1 = sample();
    const uint64_t h1 = now_ns_local();

    if (c1 <= c0 || h1 <= h0) {
        detail = "wall clock did not advance -- is this core running?";
        return 0.0;
    }
    const double ns = static_cast<double>(h1 - h0);
    const double cycles = static_cast<double>(c1 - c0);
    const double ns_per_cycle = ns / cycles;

    // Sanity band. A Blackhole Tensix runs around 1 GHz, so ~1 ns/cycle; anything outside
    // 0.05..20 ns/cycle means the register read is not returning a clock and converting
    // with it would produce confident nonsense.
    if (ns_per_cycle < 0.05 || ns_per_cycle > 20.0) {
        detail = "implausible rate " + std::to_string(ns_per_cycle) + " ns/cycle -- refusing to use it";
        return 0.0;
    }
    detail = std::to_string(1.0 / ns_per_cycle) + " GHz";
    return ns_per_cycle;
}

}  // namespace tt::tt_metal::experimental
