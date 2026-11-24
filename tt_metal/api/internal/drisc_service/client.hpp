// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Client side of the DRISC data mover service: a client-counted kernel on each of a bank's two DRISC seats, shared by
// up to abi::MAX_CLIENTS client processes. The first client launches it, later ones join, and it stops when the last
// one leaves or dies. Needs no metal device: a bare UMD cluster is enough.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <tt-metalium/vocabulary/device_types.hpp>
#include <internal/drisc_service/commands.hpp>

namespace tt::umd {
class Cluster;
class SysmemBuffer;
}  // namespace tt::umd

namespace tt::tt_metal::internal::drisc_service {

// tt-metal's runtime root, ending in '/', found the way metal finds its own firmware: the package install,
// TT_METAL_RUNTIME_ROOT, RunTimeOptions::set_root_dir(), or a working directory holding tt_metal/. Throws when none
// is found.
const std::string& runtime_root();

// Host memory on `chip` the service can copy to or from, zeroed. Sets `device_addr` to its IOVA, the address a command
// names for its first byte (pair it with SeatClient::host_endpoint). Throws when the chip has no sysmem manager or the
// IOVA lies at or above 1 << 58, which the PCIe tile does not pass through.
std::unique_ptr<tt::umd::SysmemBuffer> allocate_host_buffer(
    tt::umd::Cluster& cluster, ChipId chip, size_t size, uint64_t& device_addr);

// How SeatClient::issue() and issue_batch() write a command and its doorbell into the seat's L1.
// Window, the default: through a write-combining TLB window the seat maps on its first issue and keeps; about 1 us per
// issue instead of 3 us on a host with a TSC clock. Each window holds one of the device's TLBs, which every process on
// the device shares, so a process maps at most max_windows_per_chip per chip. Seats past that cap, or seats that find
// no TLB free, issue through Shared.
// Shared: through UMD's per-device register window, holding no TLB of its own.
enum class IssuePath { Shared, Window };
inline constexpr uint32_t max_windows_per_chip = 16;

class SeatClientImpl;

// One client's attachment to the service on one seat, made by a Client. Attaching launches the kernel when none runs;
// destruction detaches. If the process dies first, the KMD detaches on its behalf and the OS drops its seat lock. Not
// copyable or movable. One submitting thread per instance; other threads may call only done_count(), heartbeat(),
// state() and describe_state().
class SeatClient {
public:
    // Drains for up to abi::DETACH_TIMEOUT_MS, detaches, and waits as long again for the kernel to free the slot, after
    // which nothing the kernel issued for this client is in flight; only then does it unpin its host buffers. Never
    // throws: a failure is logged, and when the slot is not freed the host buffers the kernel may still write are
    // leaked. Call drain() first to get the drain failure as an exception.
    ~SeatClient();

    SeatClient(const SeatClient&) = delete;
    SeatClient& operator=(const SeatClient&) = delete;
    SeatClient(SeatClient&&) = delete;
    SeatClient& operator=(SeatClient&&) = delete;

    // True once this seat issues through a window of its own (see IssuePath); false before its first issue.
    bool has_window() const;

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
    uint64_t completed() const;

    // True once the command behind `ticket` has completed. Does not block.
    bool done(uint64_t ticket) const;

    // Spins until in_flight() is 0. Throws on `timeout`, with the counters, the seat's state and heartbeat, and the
    // slot words.
    void drain(std::chrono::milliseconds timeout) const;

    // abi::ServiceState, from L1 over PCIe.
    uint32_t state() const;

    // The kernel's slot-scan heartbeat, from the control page in host memory: the kernel refreshes it every few
    // milliseconds while this client is attached, so a value that stops changing means the kernel stopped. 0 until the
    // first publish. Costs no PCIe read and stays readable when the chip hangs.
    uint32_t heartbeat() const;

    // The seat's state, L1 heartbeat and slot words, and the control page's done_count, for error messages.
    std::string describe_state() const;

    // The endpoint a command names for host memory at `device_addr` (a sysmem IOVA on this chip): `device_addr` paired
    // with the chip's PCIe tile coords.
    abi::NocEndpoint host_endpoint(uint64_t device_addr) const;

    // The endpoint a command names for byte `offset` of this seat's bank, through the bank's endpoint for the seat's
    // NoC.
    abi::NocEndpoint bank_endpoint(uint64_t offset) const;

    uint32_t slot() const;
    uint32_t gen() const;
    // True when this client booted the kernel rather than joining it.
    bool launched() const;

private:
    friend class Client;
    friend struct SeatClientAccess;
    explicit SeatClient(std::unique_ptr<SeatClientImpl> impl);

    std::unique_ptr<SeatClientImpl> impl_;
};

// A client of the service on both seats of each of `banks` on one chip. Checks every bank before attaching any seat,
// then attaches in ascending bank order, NoC0 seat before NoC1 seat, so concurrent clients take seats in one order; on
// any failure detaches what it attached and rethrows. Not copyable or movable.
class Client {
public:
    // `root` ends in '/' and holds the service ELFs, runtime/hw/lib/blackhole/drisc_service_noc<noc>.elf. Throws when
    // `banks` is empty or names a bank the chip does not have, an ELF cannot be read or is of another service ABI, or
    // attaching a seat fails: an exclusive owner holds it, its gate stays taken, all abi::MAX_CLIENTS slots stay
    // attached, a hung kernel serves other clients, or the kernel does not ack the attach in time (the message carries
    // the seat's state, heartbeat and slot words). The ELFs are built with tt-metal, not by the JIT, so
    // TT_METAL_DISABLE_PRECOMPILED_FW=1 does not rebuild them; the first Client in a process logs a warning when it is
    // set. Every seat issues through `issue_path`.
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

private:
    std::vector<uint32_t> banks_;                     // ascending, unique
    std::vector<std::unique_ptr<SeatClient>> seats_;  // [2 * i + noc] serves banks_[i]
};

}  // namespace tt::tt_metal::internal::drisc_service
