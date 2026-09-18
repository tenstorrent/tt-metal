// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The one region: a single statically-allocated, 2 MiB-aligned span that holds every
// core's register bank and every core's TX/RX arenas, pinned once for the TT device and
// exposed once in the MPI RMA window.
//
//   PinnedMemory::Create(...)  -> PCIDevice::map_for_dma -> TENSTORRENT_IOCTL_PIN_PAGES
//   MPI_Win_create(...)        -> ibv_reg_mr             -> the NIC's own page pin
//                                 (under osc/ucx; see ucx-for-osc-not-pml)
//
// Neither allocates and neither moves the pages; both take a reference to the same
// physical memory, which the kernel refcounts. That is what lets a Tensix core's posted
// PCIe write land directly in a buffer the NIC can already read -- no bounce, no
// per-message registration. See host_uva_layout.hpp for the layout the offsets follow.
//
// Do not call fork() after registration. ibv_reg_mr sets MADV_DONTFORK on its
// range so a child does not get COW copies of registered pages; the TT pin does not. The
// single-host two-process mode therefore uses two independently launched processes, not
// a fork -- which is also why the peer attaches by name rather than by inheritance.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <tt-metalium/experimental/sockets/internal/host_uva.hpp>
#include <cstdlib>
#include <thread>

#include <tt-metalium/experimental/sockets/internal/host_stats.hpp>  // now_ns()
#include <tt-metalium/experimental/sockets/internal/host_uva_layout.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}
namespace tt::tt_metal::experimental {
class PinnedMemory;
}

namespace tt::tt_metal::experimental {

// How the device addresses this region. Both halves come from
// PinnedMemory::get_noc_addr() and both are handed to the Tensix kernel, which
// reassembles the 64-bit destination as (hi << 32) | lo for every posted write.
//
struct DeviceView {
    uint32_t pcie_xy_enc = 0;
    uint64_t io_base = 0;  // device-side address of region byte 0

};

// Reading a control word the device wrote by DMA. An ordinary load is correct on x86 --
// PCIe writes land in coherent memory -- but the compiler must be stopped from hoisting
// it out of a poll loop, and the acquire is what orders the subsequent operand reads
// after it. This is the load half of the ordering claim of the protocol: PCIe posted
// writes from one source to one endpoint complete in order, so a visible control word
// implies the operands behind it have landed.
inline uint64_t load_acquire(const volatile uint64_t* p) {
    return __atomic_load_n(const_cast<const uint64_t*>(p), __ATOMIC_ACQUIRE);
}
inline void store_release(volatile uint64_t* p, uint64_t v) {
    __atomic_store_n(const_cast<uint64_t*>(p), v, __ATOMIC_RELEASE);
}

class HostRegion {
public:
    // Provisions the region for `cores_in_use` cores and pins the matching prefix.
    // Throws on any mismatch it can detect rather than proceeding: an over-large core
    // count, a pin the system will not grant, or a device that reports no NOC address.
    struct Grid {
        uint32_t width = 0;
        uint32_t height = 0;
    };

    static HostRegion& provision(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
        uint32_t chip,
        uint32_t cores_in_use,
        HostTopology topology,
        Grid grid);

    static bool is_provisioned();

    uint8_t* base() const { return base_; }

    // The storage is a static array, so its
    // address is fixed from program start and does not depend on provision() having run.
    //
    // This exists for exactly one caller: H2D ring aliasing has to MAP_FIXED the sockets'
    // rings over the RX arenas before provision() pins the region. pinning captures the
    // physical pages and MAP_FIXED afterwards would swap them out from under both the
    // pin and the MR -- leaving the NIC writing pages that are no longer there, with nothing
    // reporting it.
    //
    // The overlay runs before there is a HostRegion to ask, and base() is not yet available.
    // Do not use this to bypass provisioning for anything else: the returned memory is
    // unvalidated, unzeroed and unpinned until provision() runs.
    static uint8_t* reserved_base();

    // H2D ring aliasing MAP_FIXEDs an H2DSocket's own shm ring over rx_arena(core).
    static void declare_rx_alias(uint32_t core, uint64_t fill_bytes, uint64_t mapped_bytes);
    static void clear_rx_aliases();

    // How many bytes of `core`'s RX arena this region may write, and where the tail it may
    // write again begins. Both are kArenaBytes / kArenaBytes for an unaliased core, which is
    // what makes the fill loop below need no special case.
    static uint64_t rx_fill_bytes(uint32_t core);
    static uint64_t rx_tail_offset(uint32_t core);

    uint64_t pinned_bytes() const { return pinned_bytes_; }
    uint32_t cores_in_use() const { return cores_in_use_; }
    const DeviceView& device() const { return device_; }

    RegionHeader* header() const { return reinterpret_cast<RegionHeader*>(base_); }

    volatile uint64_t* reg(uint32_t core, uint32_t index) const {
        return reinterpret_cast<volatile uint64_t*>(base_ + reg_offset(core, index));
    }
    // A 64-bit word at a computed BYTE offset. Exists for the per-peer credit words, which
    // live inside register 4's line rather than at a register index of their own -- reg()
    // cannot name them.
    volatile uint64_t* reg_at(uint64_t byte_offset) const {
        return reinterpret_cast<volatile uint64_t*>(base_ + byte_offset);
    }
    volatile uint64_t* ctrl_tx(uint32_t core) const { return reg(core, kCtrlTx); }
    volatile uint64_t* ctrl_rx(uint32_t core) const { return reg(core, kCtrlRx); }

    // Slot 0 IS ctrl_rx, so a single-slot run is bit-for-bit
    // the protocol that existed before the pool -- which is also what happens at a payload
    // large enough that only one message fits an arena.
    volatile uint64_t* rx_notice(uint32_t core, uint32_t slot) const { return reg(core, rx_slot_reg(slot)); }

    // Needs the payload size because the pool is from a single RX arena at runtime
    // rather than being a fixed subdivision
    uint8_t* rx_slot(uint32_t core, uint32_t slot, uint64_t payload_bytes) const {
        return base_ + rx_slot_offset(core, slot, payload_bytes);
    }

    // `head` is advanced by senders with a one-sided atomic; `tail` is published by this
    // host as it drains. A sender may use ticket n only while n - tail < slots. There is
    // exactly one pair per core and NO per-sender state, which is what makes the receive
    // cost independent of how many hosts exist.
    uint8_t* tx_arena(uint32_t core) const { return base_ + tx_arena_offset(core); }
    uint8_t* rx_arena(uint32_t core) const { return base_ + rx_arena_offset(core); }

    // Offsets, for the parties that address the region by offset rather than pointer:
    // the NIC (an MR-relative offset) and the Tensix kernel (a device IO address).
    static uint64_t tx_arena_off(uint32_t core) { return tx_arena_offset(core); }
    static uint64_t rx_arena_off(uint32_t core) { return rx_arena_offset(core); }

    // Zeroes every control word and fills both arenas with the COMPLEMENT of what a
    // correct transfer would deposit. a destination pre-filled with the
    // complement means an unwritten word always differs, so a test cannot pass by
    // accident. Zeroing control words last is deliberate -- a bank is not armed until
    // its control word is clear, so a reader that races provisioning sees idle, never a
    // half-filled arena behind a live control word.
    void reset_banks_and_arenas(uint8_t fill = kArenaFill);

    // Compares the published header against this build's constants. The failure this
    // catches is for chips_per_host: two parties with different geometry compute different
    // offsets for the same core, each reads the wrong 64 bytes, finds them idle, and
    // reports nothing.
    std::string verify_header() const;  // empty string == agreement

    static constexpr uint8_t kArenaFill = 0xA5;

private:
    HostRegion() = default;

    // The single instance. A function-local static inside a member function rather than
    // a file-scope object in the .cpp, for one reason: the constructor stays private, so
    // there is no way to make a second HostRegion anywhere. "One region" is then a
    // property the type enforces, not a rule the .cpp happens to follow.
    static HostRegion& storage();

    uint8_t* base_ = nullptr;
    uint64_t pinned_bytes_ = 0;
    uint32_t cores_in_use_ = 0;
    uint32_t chip_ = 0;
    HostTopology topology_{};
    Grid grid_{};
    DeviceView device_{};
    std::shared_ptr<tt::tt_metal::experimental::PinnedMemory> pinned_;
};

// Reports the system's pinning limits so an over-large request fails with a number
// rather than inside an ioctl. Checked before provision() pins anything.
struct PinLimits {
    uint64_t rlimit_memlock = 0;  // RLIMIT_MEMLOCK, bytes (UINT64_MAX if unlimited)
    uint32_t max_pins = 0;        // GetMemoryPinningParameters
    uint64_t max_total_pin = 0;
    bool can_map_to_noc = false;
};
PinLimits query_pin_limits(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device);

// Each peer writes an absolute count into its OWN word of the line register 4 owns, so the
// total is their sum. A single shared word cannot express this: absolute writes from several
// peers report whichever wrote last, and the sender -- comparing against notice_sent, which
// counts messages to ALL peers -- then stops opening the gate for good.
//
// Reads every word rather than only the connected ones: the unconnected are zero, the line is
// one cache line either way, and taking a peer list here would make a hot-path read depend on
// state that can change.
//
// masked to low 32 bits, because the high half of each word now carries that peer's measured
// turnaround (see credit_pack in host_uva_layout.hpp).
inline uint64_t credit_total(const HostRegion& region, uint32_t core) {
    uint64_t sum = 0;
    for (uint32_t p = 0; p < kMaxCreditPeers; ++p) {
        sum += credit_count_of(load_acquire(region.reg_at(credit_word_offset(core, p))));
    }
    return sum;
}

// The turnaround one peer reported with its most recent credit for this core, in nanoseconds on
// that peer's clock. Not summed and not averaged across peers -- it belongs to one message, and
// the only caller that can use it is the sender of that message, which knows the target peer
//
// Zero means "no credit from this peer yet", which is indistinguishable from a genuinely
// zero turnaround. The caller treats zero as absent: deliver_to_l1 brackets a
// PCIe write and a doorbell, so a true zero does not occur.
inline uint64_t credit_turnaround_ns(const HostRegion& region, uint32_t core, uint32_t peer_host) {
    if (peer_host >= kMaxCreditPeers) {
        return 0;
    }
    return credit_turnaround_of(load_acquire(region.reg_at(credit_word_offset(core, peer_host))));
}

// Did this peer refuse the message its most recent credit frees? Attribution is free: the
// word's position in the line IS the writer's host id. Credit still means "the slot is free",
// true after a refusal; this is what lets the sender stop at the first one.
inline bool credit_refused(const HostRegion& region, uint32_t core, uint32_t peer_host) {
    if (peer_host >= kMaxCreditPeers) {
        return false;
    }
    return credit_refused_of(load_acquire(region.reg_at(credit_word_offset(core, peer_host))));
}

// Returns false on timeout, and the caller reports rather than aborts: a drain that timed out
// means the interval is wider than intended, which weakens a number without invalidating a run.
inline bool drain_credits(HostRegion& region, uint32_t cores, uint64_t want, uint64_t budget_ns, uint32_t& slow_core) {
    const uint64_t dl = now_ns() + budget_ns;
    for (uint32_t core = 0; core < cores; ++core) {
        while (credit_total(region, core) < want) {
            if (now_ns() >= dl) {
                slow_core = core;
                return false;
            }
            std::this_thread::yield();
        }
    }
    return true;
}
}  // namespace tt::tt_metal::experimental
