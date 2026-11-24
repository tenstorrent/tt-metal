// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host side of the DRISC data mover service behind internal/drisc_service/client.hpp: the seats it runs on and each
// client's attachment to one. The device-side contract is hostdev/drisc_service/abi.hpp.

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <internal/drisc_service/client.hpp>
#include <tt-metalium/vocabulary/device_types.hpp>
#include <tt-umd/types/core_coordinates.hpp>

#include "hostdev/drisc_service/abi.hpp"
#include "impl/drisc_service/drisc_core.hpp"
#include "impl/drisc_service/seat_lock.hpp"
#include "internal/tt-1xx/blackhole/noc/noc_parameters.h"

struct metal_SocDescriptor;

namespace tt::umd {
class Cluster;
}  // namespace tt::umd

namespace tt::llrt {
class RunTimeOptions;
}  // namespace tt::llrt

namespace tt::tt_metal::internal::drisc_service {

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
    abi::NocEndpoint bank_base;  // the bank's first byte, through its endpoint for `noc`
    // The chip's DRAM cores whose NIU for `noc` is in stream mode, TRANSLATED: traffic there reaches the DRISC's L1.
    std::vector<uint16_t> stream_dram_noc_xy_addr;
};

// The two seats of DRAM view `bank`, as named by dram_sender_logical_cores: [0] is the bank's NOC1-endpoint DRISC,
// running on NoC0; [1] is its free DRISC, running on NoC1. Throws when `bank` is not one of the chip's DRAM views,
// whatever dram_sender_logical_cores throws, or when a seat's NoC is one its NIU must keep in NOC2AXI mode.
std::array<Seat, NUM_NOCS> bank_seats(
    tt::umd::Cluster& cluster, ChipId chip, const metal_SocDescriptor& soc, uint32_t bank);

// This process's RunTimeOptions, built once: construction reads every TT_METAL_* variable. Its root dir is found the
// way metal finds its own firmware.
const tt::llrt::RunTimeOptions& runtime_options();

// The metal SoC descriptor of `chip`, built from UMD and metal's SoC descriptor YAML without a metal device.
metal_SocDescriptor soc_descriptor(tt::umd::Cluster& cluster, ChipId chip);

// Unmaps pages from map_host_pages().
struct UnmapPages {
    size_t size;
    void operator()(void* va) const noexcept;
};

// What a SeatClient does, documented there. Construction attaches, launching the kernel when none runs; destruction
// detaches. If the process dies first, the KMD detaches on its behalf (ExitWrite) and the OS drops its seat lock.
class SeatClientImpl {
public:
    // Under the seat's gate: takes the client seat lock, then joins the running kernel, or launches `image` (which
    // must suit seat.noc) when the kernel is stopped, or hung while this client holds the seat alone. The kernel
    // shares the seat's bandwidth equally by bytes between its attached clients.
    //
    // Throws, attaching nothing, when an exclusive owner holds the seat, the gate stays taken, all abi::MAX_CLIENTS
    // slots stay attached for abi::ATTACH_TIMEOUT_MS, a hung kernel serves other clients, or the kernel does not ack
    // the attach in time (the message carries the seat's state, heartbeat and slot words). A failed launch leaves the
    // core in reset.
    SeatClientImpl(
        tt::umd::Cluster& cluster,
        const Seat& seat,
        const ll_api::ElfFile& image,
        IssuePath issue_path = IssuePath::Window);

    ~SeatClientImpl();

    SeatClientImpl(const SeatClientImpl&) = delete;
    SeatClientImpl& operator=(const SeatClientImpl&) = delete;
    SeatClientImpl(SeatClientImpl&&) = delete;
    SeatClientImpl& operator=(SeatClientImpl&&) = delete;

    // True once this seat issues through a window of its own (see IssuePath); false before its first issue.
    bool has_window() const { return window_ != nullptr; }

    // Writes `command` into the slot's next queue entry and rings its doorbell, and returns its ticket for done().
    // Returns nothing and does nothing when abi::CLIENT_QUEUE_CAPACITY commands are in flight. A CMD_SINGLE whose
    // source and destination both lie in this seat's bank, in 16 B units, is sent as the CMD_LOCAL that copies it.
    // Throws, issuing nothing, when the command is not CMD_SINGLE or CMD_LOCAL, its size is 0, its source coords do not
    // fit in abi::CMD_KIND_SHIFT bits, src or dst is not 64 B aligned, src and dst overlap on one endpoint, a CMD_LOCAL
    // size is not a multiple of abi::LOCAL_SIZE_ALIGN or runs past the NoC-addressable part of the bank. Endpoints
    // and sizes are otherwise not checked.
    // Commands run concurrently and complete in issue order, with no ordering between their data: drain() between a
    // command and one that reads what it wrote.
    [[nodiscard]] std::optional<uint64_t> issue(const abi::KernelCommand& command);

    // Copies `commands` into the host batch slot of the next queue entry and issues one CMD_BATCH for them, and returns
    // its one ticket. Returns nothing and does nothing when the queue is full. Throws, issuing nothing, when `commands`
    // is empty, longer than abi::BATCH_MAX_COMMANDS, or holds a command issue() would reject.
    [[nodiscard]] std::optional<uint64_t> issue_batch(std::span<const abi::KernelCommand> commands);

    // Commands issued and not yet completed (a CMD_BATCH counts once). Submitting thread only, like completed() and
    // done().
    uint32_t in_flight() const;

    // Commands the kernel has completed (a CMD_BATCH counts once), from the control page in host memory. 32-bit, so it
    // wraps.
    uint32_t done_count() const;

    // Commands completed so far (a CMD_BATCH counts once). 64-bit, so it does not wrap.
    uint64_t completed() const { return issued_ - in_flight(); }

    // True once the command behind `ticket` has completed. Does not block.
    bool done(uint64_t ticket) const { return completed() >= ticket; }

    // Spins until in_flight() is 0. Throws on `timeout`, with the counters, the seat's state and heartbeat, and the
    // slot words.
    void drain(std::chrono::milliseconds timeout) const;

    // abi::ServiceState, from L1 over PCIe.
    uint32_t state() const;

    // The kernel's slot-scan heartbeat, from the control page in host memory: the kernel publishes it at least every
    // abi::CONTROL_PAGE_HEARTBEAT_TICKS while this client is attached, so a value that stops changing means the kernel
    // stopped. 0 until the first publish. Costs no PCIe read and stays readable when the chip hangs, unlike the live
    // counter in L1 (l1_heartbeat()), which the client reads only before attach and in describe_state().
    uint32_t heartbeat() const;

    // The seat's state, L1 heartbeat and slot words, and the control page's done_count, for error messages.
    std::string describe_state() const;

    // The endpoint a command names for host memory at `device_addr` (a sysmem IOVA on this chip): `device_addr` paired
    // with the chip's PCIe tile coords.
    abi::NocEndpoint host_endpoint(uint64_t device_addr) const;
    abi::NocEndpoint bank_endpoint(uint64_t offset) const { return seat_.bank_base.at_offset(offset); }

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
    std::optional<uint64_t> post(const abi::KernelCommand& command);
    uint8_t endpoint_flags(uint16_t noc_xy_addr) const;
    bool in_own_bank(uint16_t noc_xy_addr, uint64_t addr, uint32_t size) const;
    // The command as sent: a CMD_SINGLE within this seat's bank becomes CMD_LOCAL. Throws if issue() would reject it.
    abi::KernelCommand prepare(const abi::KernelCommand& command) const;
    // The kernel's live heartbeat counter in L1, over PCIe. For when the control page copy has nothing yet (classify(),
    // before attach) or may lag (describe_state()).
    uint32_t l1_heartbeat() const;
    void validate(const abi::KernelCommand& command, bool src_in_own_bank) const;

    Seat seat_;
    static constexpr uint8_t own_bank_endpoint = 1;
    static constexpr uint8_t stream_mode_endpoint = 2;
    std::array<uint8_t, size_t{1} << abi::CMD_KIND_SHIFT> endpoint_flags_{};  // indexed by noc_xy_addr
    PeerWindows* peers_ = nullptr;                                            // the owning Client's
    DriscCore core_;
    std::optional<SeatLock> lock_;
    // Pinned on exit_write_'s fd, so the KMD sends the detach before unpinning them.
    std::unique_ptr<void, UnmapPages> control_page_memory_;
    std::unique_ptr<void, UnmapPages> batch_memory_;
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

// Test access to a SeatClient's internals.
struct SeatClientAccess {
    static const SeatClientImpl& impl(const SeatClient& client) { return *client.impl_; }
};
inline const Seat& seat_of(const SeatClient& client) { return SeatClientAccess::impl(client).seat(); }

}  // namespace tt::tt_metal::internal::drisc_service
