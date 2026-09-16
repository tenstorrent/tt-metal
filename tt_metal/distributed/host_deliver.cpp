// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/internal/host_deliver.hpp>

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <ctime>
#include <sstream>
#include <stdexcept>  // std::runtime_error -- the fatal aliasing failure, A2

#include <tt-metalium/experimental/sockets/internal/host_uva_layout.hpp>
#include <tt-metalium/experimental/sockets/internal/host_region.hpp>

#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <internal/cluster_noc_helpers.hpp>

#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <cerrno>
#include <cstddef>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "tt_metal/distributed/hd_socket_descriptor.hpp"
#include "tt_metal/distributed/hd_socket_connector_state.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"

namespace tt::tt_metal::experimental {


namespace {

ttsl::Span<const std::byte> byte_span(const void* p, std::size_t n) {
    return ttsl::Span<const std::byte>(static_cast<const std::byte*>(p), n);
}

std::vector<uint8_t> to_u8(const std::vector<std::byte>& in) {
    std::vector<uint8_t> out(in.size());
    if (!in.empty()) {
        std::memcpy(out.data(), in.data(), in.size());
    }
    return out;
}

uint64_t now_ns_local() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

class H2DSocketDeliverer final : public Deliverer {
public:
    H2DSocketDeliverer(
        std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh, uint32_t grid_width, uint32_t cores,
        L1Layout layout, H2DSocketConfig cfg) :
        mesh_(std::move(mesh)), layout_(layout), cores_(cores), cfg_(cfg) {
        device_id_ = static_cast<uint32_t>(mesh_->get_devices()[0]->id());
        pre_acked_.assign(cores, 0);
        // Unconditional: there is no config that selects HOST_PUSH, and no flag that sets one.
        const auto mode = tt::tt_metal::distributed::H2DMode::DEVICE_PULL;
        sockets_.reserve(cores);
        virt_.reserve(cores);
        for (uint32_t i = 0; i < cores; ++i) {
            const CoreCoord logical{i % grid_width, i / grid_width};
            // logical coords here, unlike DeviceDeliverer. H2DSocket does the
            // logical->virtual translation itself; handing it a translated coord
	    // would translate twice and name a different tile. The virtual coords
	    // below are for the doorbell and the readback only.
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
                throw std::runtime_error("h2d-alias: the rings could not be mapped: " + e);
            } else {
                alias_on_ = true;
                std::cerr << "H2D ring aliasing ON for " << cores_ << " cores, fifo " << cfg_.fifo_size
                          << " B; the peer must RMA into the ring, and "
                             "fifo_size must be an exact multiple of the payload -- that "
                             "multiple is the send window.\n";
            }
        }
    }

    ~H2DSocketDeliverer() override {
        unmap_rings();
    }

    uint32_t page_size_for() const {
        // page size must be PCIe aligned using hal::get_pcie_alignment()
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

        // On this path the host does not write the payload into L1 -- the receiver kernel
        // does (H2D-Pull), after it sees the page. So the effective address has to reach the kernel
        // before the page does. Both writes go out on the strict-ordered UC path to the same
        // tile, and the socket's bytes_sent update is what releases the kernel, so writing
        // this first is the whole ordering argument. Reversed, the kernel would write this
        // message's payload to the previous message's address, with nothing reporting it.
        //
        // Stores are not supported: the pull kernel signals signal_addr, wait_delivered()
        // polls the socket's bytes_acked, and nothing advances either.
        if (dst_l1 != 0) {
            return "h2d-alias: store completion is not implemented";
        }

        pre_acked_[core] = sockets_[core]->bytes_acked_snapshot();

        if (alias_on_) {
            if (ring_[core] == nullptr) {
                return "h2d-alias: ring for this core was never mapped";
            }

            // The ring is now kNumAliasRingSlots * payload, and the pointer is SUPPOSED to sit at
            // slot * payload between messages. What must still hold is that the ring is an
            // exact multiple of the payload -- otherwise the wrap does not land on a slot
            // boundary and h2d_socket.cpp:663-667 never sees the exact-fill case.
            if (bytes == 0 || cfg_.fifo_size == 0 || (cfg_.fifo_size % bytes) != 0) {
                std::ostringstream o;
                o << "h2d-alias: fifo_size " << cfg_.fifo_size << " B must be a non-zero exact "
                  << "multiple of the payload " << bytes
                  << " B -- the ring is the send window, "
                     "and a partial slot leaves the write pointer off a slot boundary so the "
                     "peer's next RMA lands somewhere the device will not read. Size the socket "
                     "with kNumAliasRingSlots * payload.";
                return o.str();
            }

            // source must be a slot of this core's ring. Was `src != ring_[core]`, which
            // only admitted slot 0. The caller hands us region_.rx_slot(core, slot, length), so
            // at depth > 1 this is legitimately ring + slot * payload.
            if (src < ring_[core]) {
                return "h2d-alias: the source is below this core's ring; the RX arena has not "
                       "been aliased onto it, so the payload is not where the device will look";
            }
            const size_t slot_off = static_cast<size_t>(src - ring_[core]);
            if ((slot_off % bytes) != 0 || slot_off + bytes > cfg_.fifo_size) {
                std::ostringstream o;
                o << "h2d-alias: the source is " << slot_off << " B into this core's ring, which "
                  << "is not a " << bytes << " B slot boundary inside " << cfg_.fifo_size
                  << " B -- the RX arena is not aliased onto the ring as expected";
                return o.str();
            }

            // The peer RMA'd into slot (seq % depth). The device finds the payload through the
            // socket's read pointer, which follows bytes_sent -- advanced one payload per
            // delivery by commit_to_device(). Those two agree only if deliveries happen in
            // slot order, which BankScanner::next_rx_slot_ enforces on the receive side.
            //
            // If that ordering is ever wrong -- a reordered RMA, a stolen job servicing out of
            // turn, a depth mismatch between the three places that derive it -- the symptom
            // without this check is the device reading a slot the pointer is not on: wrong
            // payload bytes, no error anywhere. With it, it is one line naming both offsets.
            const size_t expect_off = static_cast<size_t>(sent_[core] % cfg_.fifo_size);
            if (slot_off != expect_off) {
                std::ostringstream o;
                o << "h2d-alias: OUT-OF-ORDER DELIVERY -- payload is at ring offset " << slot_off
                  << " B but the socket's write pointer is at " << expect_off << " B (bytes_sent " << sent_[core]
                  << ", fifo " << cfg_.fifo_size << ", payload " << bytes
                  << "). The device would read the wrong slot. Deliveries must reach this "
                     "function in ring order; see BankScanner::next_rx_slot_.";
                return o.str();
            }

            return commit_to_device(core, bytes);
        }

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

        // Nothing to write: the socket's own bytes_sent notification IS the doorbell here.
        (void)value;
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
        tt::tt_metal::internal::noc_write_immediate(
            device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), layout_.completion_addr,
            byte_span(&value, sizeof(value)));
        return {};
    }

    // The payload is in pinned host RAM and stays there until a receiver kernel pulls it,
    // so this reads an L1 that nothing has written. The bench refuses to claim a DEVICE_PULL pass on it.
    std::vector<uint8_t> read_payload(uint32_t core, uint32_t bytes, uint32_t src_l1) override {
        if (core >= cores_) {
            return {};
        }
        const auto& v = virt_[core];
        return to_u8(tt::tt_metal::internal::noc_read(
            device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y),
            (src_l1 != 0) ? src_l1 : layout_.payload_addr, bytes));
    }

    uint32_t read_reg32(uint32_t core, uint64_t addr) override {
        if (core >= cores_) {
            return 0;
        }
        const auto& v = virt_[core];
        return tt::tt_metal::internal::noc_read_reg_u32(
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

        // The receiver posts bytes_acked into pinned host memory after its read barrier
        // (socket_api.h socket_notify_sender), so this says exactly what the L1 doorbell
        // says -- the payload is in L1 -- with a local load instead of a non-posted PCIe
        // read. Polling device L1 costs ~1 us a sample and contends with the very payload
        // reads it waits for: the protocol's invariant is that each side writes to the
        // other's memory and polls only its OWN, and the doorbell poll was the one place
        // this path broke it.
        //
        const uint32_t want = expected != 0 ? expected : page_size_;
        const uint32_t before = pre_acked_[core];
        while (static_cast<uint32_t>(sockets_[core]->bytes_acked_snapshot() - before) < want) {
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
                o << "h2d-socket: core " << core
                  << " delivery stalled -- the receiver kernel is "
                     "not running, not draining, or was given the wrong socket config address ("
                  << "bytes_acked " << sockets_[core]->bytes_acked_snapshot() << ", was " << before << ", needed +"
                  << want << ", page_size " << page_size_ << ")";
                return o.str();
            }
        }
        return {};
    }

    std::string arm_receivers() override {
        if (layout_.stop_addr == 0) {
            return "h2d-socket: no stop address configured; receiver kernels cannot be armed";
        }
        const uint32_t zero = 0;
        for (uint32_t c = 0; c < cores_; ++c) {
            const auto& v = virt_[c];
            tt::tt_metal::internal::noc_write_immediate(
                device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), layout_.stop_addr,
                byte_span(&zero, sizeof(zero)));
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
            tt::tt_metal::internal::noc_write_immediate(
                device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y), layout_.stop_addr,
                byte_span(&one, sizeof(one)));
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
          << "DEVICE_PULL, fifo " << cfg_.fifo_size << " B, page " << page_size_ << " B, doorbell socket)";
        return o.str();
    }


    // Returns "" on success; on failure the aliasing is left off and the caller keeps the
    // memcpy path, because a half-mapped set is worse than no mapping at all.
    std::string map_rings() {
        // checks that physical pages are pinned for the MPI RMA window
        if (HostRegion::is_provisioned()) {
            return "h2d-alias: the region is already provisioned (and therefore pinned), so "
                   "overlaying the rings now would swap pages out from under the pin and the "
                   "MR. The deliverer must be constructed BEFORE HostRegion::provision().";
        }
        ring_.assign(cores_, nullptr);
        ring_bytes_.assign(cores_, 0);
        cfg_addr_.assign(cores_, 0);
        sent_.assign(cores_, 0);
        connector_.assign(cores_, nullptr);
        for (uint32_t c = 0; c < cores_; ++c) {
            const auto d = sockets_[c]->populate_descriptor();
            // shm_size comes off the descriptor rather than fstat so that a layout change
            // upstream disagrees with us here rather than silently mapping a different span.
            // this core's RX arena, which is where the peer already writes. The region is
            // sharded per core already, and there is one socket per core, so the correspondence
            // is 1:1 and needs no new addressing scheme.
            uint8_t* const slot = cfg_.alias_region_base + rx_arena_offset(c);

            // An arena slot is kArenaBytes and the next thing after it is the following core's TX
            // arena, so a shm_size within a page of the slot size would overlay memory belonging to a
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

            // The connector state rides in the SAME shm we just mapped, at an offset the
            // descriptor publishes -- so aliasing the ring aliases this too, and commit_to_device()
            // can keep the host-side counter honest without reaching into H2DSocket's privates.
            // fifo_size is: the socket is the authority on the layout it actually built.
            connector_[c] = reinterpret_cast<tt::tt_metal::distributed::HDSocketConnectorState*>(
                static_cast<uint8_t*>(p) + d.connector_state_offset);

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
            // Points into the mapping just replaced above, so it dies with the ring or it
            // dangles into an anonymous page.
            connector_[c] = nullptr;
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

        // 2 bytes_sent, and write() advances both. This reimplemented only the
        // device-side one for a long time, which is a bug that hides completely in the data path
        // and then costs a minute of teardown:
        //
        //   push_bytes()      h2d_socket.cpp:674  connector_state_->bytes_sent = bytes_sent_
        //                                         -- HOST copy, in the shm
        //   notify_receiver() h2d_socket.cpp:680  pcie_writer(..., config_buffer + offsetof(...))
        //                                         -- DEVICE copy, in L1, what the kernel polls
        //
        // Host copy FIRST, device copy second: the device write is what releases the kernel, so
        // it stays the last store here, exactly as it was.
        //
        if (connector_[core] != nullptr) {
            connector_[core]->bytes_sent = sent_[core];
        }

        const auto& v = virt_[core];
        tt::tt_metal::internal::noc_write_immediate(
            device_id_, static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y),
            cfg_addr_[core] + offsetof(receiver_socket_md, bytes_sent),
            byte_span(&sent_[core], sizeof(uint32_t)));
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
    uint32_t ring_data_offset_ = 0;
    std::vector<uint8_t*> ring_;
    std::vector<size_t> ring_bytes_;
    std::vector<uint32_t> cfg_addr_;
    std::vector<uint32_t> sent_;
    // Into the aliased shm, one per core; null when the ring is not aliased or after teardown.
    std::vector<tt::tt_metal::distributed::HDSocketConnectorState*> connector_;
};

}  // namespace

std::unique_ptr<Deliverer> make_h2d_socket_deliverer(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device, uint32_t grid_width, uint32_t cores,
    L1Layout layout, H2DSocketConfig cfg, std::string& error) {
    error.clear();
    if (mesh_device == nullptr) {
        error = "no mesh device";
        return nullptr;
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
