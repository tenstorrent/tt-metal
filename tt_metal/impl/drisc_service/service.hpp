// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host side of the DRISC data mover service: a client-counted kernel on each of a bank's two DRISC seats, shared by up
// to abi::MAX_CLIENTS client processes. The first client launches it, later ones join, and it stops when the last one
// leaves or dies. The device-side contract is hostdev/drisc_service/abi.hpp. Needs no metal device: a bare UMD cluster
// is enough.

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <tt-metalium/device_types.hpp>
#include <tt-umd/types/core_coordinates.hpp>

#include "hostdev/drisc_service/abi.hpp"
#include "impl/drisc_service/drisc_core.hpp"
#include "impl/drisc_service/seat_lock.hpp"

struct metal_SocDescriptor;

namespace tt::umd {
class Cluster;
class SysmemBuffer;
}  // namespace tt::umd

namespace tt::tt_metal::drisc_service {

class PeerWindows;

// One DRISC the service runs on, and the NoC its kernel drives.
struct Seat {
    ChipId chip;
    uint32_t bank;
    uint32_t noc;
    tt::umd::CoreCoord translated;
    SeatId id;
    uint8_t noc2axi_mask;  // NIUs DRISC firmware leaves in NOC2AXI mode; the kernel never drives these
    uint64_t bank_size;    // NoC-addressable bytes of the bank (dram_view_size)
    std::array<uint16_t, abi::BANK_ENDPOINTS> bank_noc_xy_addr;  // every endpoint of the bank, TRANSLATED
};

// The two seats of DRAM view `bank`, as named by dram_sender_logical_cores: [0] is the bank's NOC1-endpoint DRISC,
// running on NoC0; [1] is its free DRISC, running on NoC1. Throws when `bank` is not one of the chip's DRAM views,
// whatever dram_sender_logical_cores throws, or when a seat's NoC is one its NIU must keep in NOC2AXI mode.
std::array<Seat, 2> bank_seats(tt::umd::Cluster& cluster, ChipId chip, const metal_SocDescriptor& soc, uint32_t bank);

// tt-metal's runtime root, ending in '/', found the way metal finds its own firmware: the package install,
// TT_METAL_RUNTIME_ROOT, RunTimeOptions::set_root_dir(), or a working directory holding tt_metal/. Throws when none
// is found.
const std::string& runtime_root();

// The metal SoC descriptor of `chip`, built from UMD and <root>tt_metal/soc_descriptors/ without a metal device.
// `root` ends in '/'.
metal_SocDescriptor soc_descriptor(tt::umd::Cluster& cluster, ChipId chip, const std::string& root = runtime_root());

// Host memory on `chip` the service can copy to or from, zeroed. Sets `device_addr` to the address a command names
// for its first byte (pair it with SeatClient::host_endpoint). The IOVA is remapped through the iATU when it lies at or
// above 1 << 58, which the PCIe tile does not pass through. Throws when the chip has no sysmem manager or the remap
// fails.
std::unique_ptr<tt::umd::SysmemBuffer> allocate_host_buffer(
    tt::umd::Cluster& cluster, ChipId chip, size_t size, uint64_t& device_addr);

// How SeatClient::issue() and issue_batch() write a command and its doorbell into the seat's L1.
// Window, the default: through a write-combining TLB window the seat maps on its first issue and keeps, as tt-d-gen's
// link does; about 1 us per issue instead of 3 us on a host with a TSC clock. Each window holds one of the device's
// TLBs, which every process on the device shares, so a process maps at most max_windows_per_chip per chip. Seats past
// that cap, or seats that find no TLB free, issue through Shared.
// Shared: through UMD's per-device register window, holding no TLB of its own.
enum class IssuePath { Shared, Window };
inline constexpr uint32_t max_windows_per_chip = 8;

// One client's attachment to the service on one seat. Construction attaches, launching the kernel when none runs;
// destruction detaches. If the process dies first, the KMD detaches on its behalf (ExitWrite) and the OS drops its
// seat lock. Not copyable or movable; not thread-safe (one submitting thread per instance).
class SeatClient {
public:
    // Under the seat's gate: takes the client seat lock, then joins the running kernel, or launches `image` (which
    // must suit seat.noc) when the kernel is stopped, or hung while this client holds the seat alone. The kernel
    // shares the seat's bandwidth equally by bytes between its attached clients.
    //
    // Throws, attaching nothing, when an exclusive owner holds the seat, the gate stays taken, all abi::MAX_CLIENTS
    // slots stay attached for abi::ATTACH_TIMEOUT_MS, a hung kernel serves other clients, or the kernel does not ack
    // the attach in time (the message carries the seat's state, heartbeat and slot words). A failed launch leaves the
    // core in reset.
    SeatClient(
        tt::umd::Cluster& cluster,
        const Seat& seat,
        const ll_api::ElfFile& image,
        IssuePath issue_path = IssuePath::Window);

    // Drains for up to abi::DETACH_TIMEOUT_MS, detaches, and waits as long again for the kernel to free the slot.
    // Never throws: a failure is logged, and when the slot is not freed the host buffers the kernel may still write are
    // leaked. Call drain() first to get the drain failure as an exception.
    ~SeatClient();

    SeatClient(const SeatClient&) = delete;
    SeatClient& operator=(const SeatClient&) = delete;
    SeatClient(SeatClient&&) = delete;
    SeatClient& operator=(SeatClient&&) = delete;

    // True once this seat issues through a window of its own (see IssuePath); false before its first issue.
    bool has_window() const { return window_ != nullptr; }

    // Writes `command` into the slot's next queue entry and rings its doorbell. Returns false and does nothing when
    // abi::CLIENT_QUEUE_CAPACITY commands are in flight.
    // Throws, issuing nothing, when the command is not CMD_SINGLE or CMD_LOCAL, its size is 0, its source coords do not
    // fit in abi::CMD_KIND_SHIFT bits, src or dst is not 64 B aligned, src and dst overlap on one endpoint, a CMD_LOCAL
    // size is not a multiple of abi::LOCAL_SIZE_ALIGN or runs past the NoC-addressable part of the bank, or its source
    // lies in a peer window of the client that owns this seat (peer windows are never read through). Endpoints and
    // sizes are otherwise not checked.
    // Commands run concurrently and complete in issue order, with no ordering between their data: drain() between a
    // command and one that reads what it wrote.
    [[nodiscard]] bool issue(const abi::KernelCommand& command);

    // Copies `commands` into the host batch slot of the next queue entry and issues one CMD_BATCH for them. Returns
    // false and does nothing when the queue is full. Throws, issuing nothing, when `commands` is empty, longer than
    // abi::BATCH_MAX_COMMANDS, or holds a command issue() would reject.
    [[nodiscard]] bool issue_batch(std::span<const abi::KernelCommand> commands);

    // Commands issued and not yet completed (a CMD_BATCH counts once).
    uint32_t in_flight() const;

    // Spins until in_flight() is 0. Throws on `timeout`, with the counters, the seat's state and heartbeat, and the
    // slot words.
    void drain(std::chrono::milliseconds timeout) const;

    // The command block header words in L1: abi::ServiceState, and the slot-scan heartbeat.
    uint32_t state() const;
    uint32_t heartbeat() const;

    // The endpoint a command names for host memory at `device_addr` (a sysmem IOVA or iATU NoC address on this
    // chip): `device_addr` paired with the chip's PCIe tile coords.
    abi::NocEndpoint host_endpoint(uint64_t device_addr) const;

    const Seat& seat() const { return seat_; }
    uint32_t slot() const { return slot_; }
    uint32_t gen() const { return gen_; }
    // True when this client booted the kernel rather than joining it.
    bool launched() const { return launched_; }

private:
    friend class Client;
    enum class Kernel { Stopped, Running, Hung };
    Kernel classify() const;
    void join(const ll_api::ElfFile& image);
    void launch(const ll_api::ElfFile& image, uint32_t slot);
    void request_attach(uint32_t slot);
    bool acked() const;
    bool post(const abi::KernelCommand& command);
    std::string describe_state() const;
    void validate(const abi::KernelCommand& command) const;

    Seat seat_;
    PeerWindows* peers_ = nullptr;  // the owning Client's; null for a SeatClient of its own
    DriscCore core_;
    std::optional<ClientSeatLock> lock_;
    std::unique_ptr<tt::umd::SysmemBuffer> control_page_buffer_;
    std::unique_ptr<tt::umd::SysmemBuffer> batch_buffer_;
    uint64_t control_page_device_addr_ = 0;
    uint64_t batch_device_addr_ = 0;
    abi::ControlPage* control_page_ = nullptr;
    abi::KernelCommand* batch_slots_ = nullptr;
    uint32_t pcie_noc_xy_addr_ = 0;
    uint32_t slot_ = 0;
    uint32_t gen_ = 0;
    bool launched_ = false;
    uint64_t issued_ = 0;
    IssuePath issue_path_;
    bool window_tried_ = false;
    // IssuePath::Window's mapping, made on the first post(); null on Shared, past the cap, or when no TLB was free.
    std::unique_ptr<tt::umd::IoWindow> window_;
    // Last, so it fires before the host buffers the kernel writes into are freed.
    std::optional<ExitWrite> exit_write_;
};

// A client of the service on both seats of each of `banks` on one chip. Checks every bank before attaching any seat,
// then attaches in ascending bank order, NoC0 seat before NoC1 seat, so concurrent clients take seats in one order; on
// any failure detaches what it attached and rethrows. Not copyable or movable.
class Client {
public:
    // `root` ends in '/' and holds tt_metal/soc_descriptors/ and the service ELFs,
    // runtime/hw/lib/blackhole/drisc_service_noc<noc>.elf. Throws when `banks` is empty, an ELF cannot be read or has
    // no build block of this abi::ABI_VERSION, or whatever bank_seats or SeatClient throws. The ELFs are built with
    // tt-metal, not by the JIT, so TT_METAL_DISABLE_PRECOMPILED_FW=1 does not rebuild them; the first Client in a
    // process logs a warning when it is set. Every seat issues through `issue_path`.
    Client(
        tt::umd::Cluster& cluster,
        ChipId chip,
        std::vector<uint32_t> banks,
        const std::string& root = runtime_root(),
        IssuePath issue_path = IssuePath::Window);
    ~Client();

    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;
    Client(Client&&) = delete;
    Client& operator=(Client&&) = delete;

    // The attachment that runs commands on `noc` for `bank`. Throws when `bank` is not one of this client's banks or
    // `noc` is not 0 or 1.
    SeatClient& seat(uint32_t bank, uint32_t noc);

    // Waits until every seat's in_flight() is 0, within `timeout` for all of them together. Throws on timeout as
    // SeatClient::drain() does, naming the first seat still busy.
    void drain(std::chrono::milliseconds timeout) const;
    const std::vector<uint32_t>& banks() const { return banks_; }

    // The destination a command on any of this client's seats names to write `size` bytes at `addr` of core `core`
    // (TRANSLATED) on chip `peer`, over PCIe. The first range no window covers gets a 2 MiB window on `peer`, mapped
    // into this chip's IOMMU domain; the client keeps its windows until its seats have detached.
    //
    // The peer takes the writes posted (it does not wait for its DRAM's acks), on one static VC, so they land in the
    // order the seat issues them. A copy completes by a flag the same seat writes after the data, at the same peer
    // core, which the receiver, or this process, polls with a PeerFlag; done_count says only that the writes left this
    // chip.
    //
    // Throws when `peer` is this client's chip, the range crosses a 2 MiB block, `peer` has no window free, KMD refuses
    // the mapping, or the mapping lands where this chip's PCIe tile cannot reach it.
    abi::NocEndpoint peer_endpoint(ChipId peer, const tt::umd::CoreCoord& core, uint64_t addr, uint64_t size);

private:
    std::vector<uint32_t> banks_;  // ascending, unique
    // Before seats_, so destroyed after every seat has detached.
    std::unique_ptr<PeerWindows> peers_;
    std::vector<std::unique_ptr<SeatClient>> seats_;  // [2 * i + noc] serves banks_[i]
};

// A 32-bit flag in a chip's memory, read through a read-only window of its own: the receiver of a peer copy polls it,
// and so can the sender, which has the receiving chip open. Not copyable or movable.
class PeerFlag {
public:
    // The flag at `addr` of core `core` (TRANSLATED) on `chip`. Throws when the chip has no TLB window free.
    PeerFlag(tt::umd::Cluster& cluster, ChipId chip, const tt::umd::CoreCoord& core, uint64_t addr);
    ~PeerFlag();

    PeerFlag(const PeerFlag&) = delete;
    PeerFlag& operator=(const PeerFlag&) = delete;
    PeerFlag(PeerFlag&&) = delete;
    PeerFlag& operator=(PeerFlag&&) = delete;

    uint32_t value() const;
    // Spins until the flag reaches `seq` in 32-bit sequence order, so a later value also counts. Throws on `timeout`.
    void wait(uint32_t seq, std::chrono::milliseconds timeout) const;

private:
    std::unique_ptr<tt::umd::IoWindow> window_;
};

}  // namespace tt::tt_metal::drisc_service
