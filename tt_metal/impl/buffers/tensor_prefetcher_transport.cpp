// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/tensor_prefetcher_transport.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-logger/tt-logger.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/buffers/h2d_socket_internal.hpp"
#include "impl/buffers/tensor_prefetcher_manager.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::distributed {

namespace {

// How long the host worker pauses when no TL1 request ring has room.
constexpr std::chrono::milliseconds kRequestRingRetryInterval{1};

uint32_t read_bytes_acked(
    const MeshDevice& mesh_device, IDevice* device, const CoreCoord& sender_logical, uint32_t control_base) {
    uint32_t bytes_acked = 0;
    read_dram_core_l1(
        mesh_device,
        device,
        sender_logical,
        control_base + kCceRingMetaOffset + offsetof(RingMeta, bytes_acked),
        std::as_writable_bytes(std::span(&bytes_acked, 1)));
    return bytes_acked;
}

}  // namespace

H2DSocketTransport::H2DSocketTransport(
    MeshDevice& mesh_device,
    const std::vector<IDevice*>& devices,
    const std::vector<CoreCoord>& sender_logical_cores,
    uint32_t socket_config_l1_addr,
    uint32_t socket_data_l1_addr) {
    const Hal& hal = MetalContext::instance(mesh_device.impl().get_context_id()).hal();
    const uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    page_bytes_ = tt::align(kRequestPageBytes, pcie_alignment);
    const uint32_t fifo_size = tt::align(page_bytes_ * TensorPrefetcherManager::kSocketFifoPages, pcie_alignment);

    sockets_.reserve(devices.size() * sender_logical_cores.size());

    std::shared_ptr<MeshDevice> mesh_device_shared = mesh_device.shared_from_this();
    const uint64_t dram_l1_noc_offset = hal.get_l1_noc_offset(HalProgrammableCoreType::DRAM);

    for (IDevice* device : devices) {
        const MeshCoordinate device_coord = mesh_device.get_view().find_device(device->id());
        for (const CoreCoord& sender_logical : sender_logical_cores) {
            // Uniform L1 layout per DRAM core: only one socket lives on each core (the one for that core's
            // own sender), at the offsets the manager carved.
            std::unique_ptr<H2DSocket> socket = experimental::detail::create_h2d_socket_for_dram_recv(
                mesh_device_shared,
                MeshCoreCoord(device_coord, sender_logical),
                fifo_size,
                socket_config_l1_addr,
                socket_data_l1_addr,
                dram_l1_noc_offset);
            socket->set_page_size(page_bytes_);
            sockets_.push_back(std::move(socket));
        }
    }
}

H2DSocketTransport::~H2DSocketTransport() = default;

bool H2DSocketTransport::try_write(uint32_t endpoint, std::vector<uint8_t>& page) {
    return experimental::detail::try_write(*sockets_.at(endpoint), page.data(), 1);
}

bool H2DSocketTransport::target_exited(uint32_t /*endpoint*/) { return false; }

bool H2DSocketTransport::gave_up_on(uint32_t /*endpoint*/) const { return false; }

bool H2DSocketTransport::give_up_on_stalled(
    uint32_t /*endpoint*/, uint8_t /*command*/, std::chrono::steady_clock::duration /*without_progress*/) {
    return false;
}

// Yield rather than busy-spin at 100% CPU until a receiver frees a page.
void H2DSocketTransport::pause_after_stalled_pass() const { std::this_thread::yield(); }

std::optional<std::chrono::steady_clock::duration> H2DSocketTransport::gave_up_after() const { return std::nullopt; }

void H2DSocketTransport::log_sender_statuses() const {}

std::vector<std::string> H2DSocketTransport::failed_senders() const { return {}; }

uint32_t H2DSocketTransport::page_bytes() const { return page_bytes_; }

uint32_t H2DSocketTransport::config_buffer_address(uint32_t endpoint) const {
    return sockets_.at(endpoint)->get_config_buffer_address();
}

Tl1RingTransport::Tl1RingTransport(
    const MeshDevice& mesh_device,
    const std::vector<IDevice*>& devices,
    const std::vector<CoreCoord>& sender_logical_cores,
    uint32_t control_base,
    std::chrono::steady_clock::duration no_progress_limit,
    bool give_up_on_running_senders) :
    mesh_device_(mesh_device),
    control_base_(control_base),
    no_progress_limit_(no_progress_limit),
    give_up_on_running_senders_(give_up_on_running_senders) {
    endpoints_.reserve(devices.size() * sender_logical_cores.size());
    for (IDevice* device : devices) {
        for (const CoreCoord& sender_logical : sender_logical_cores) {
            endpoints_.push_back(Endpoint{.device = device, .sender_logical = sender_logical});
        }
    }
}

// Runs on the worker thread, where a throw would terminate: the manager builds every page page_bytes() long and
// only passes endpoints it provisioned. A failed device access leaves the page unsent; the first is logged.
bool Tl1RingTransport::try_write(uint32_t endpoint_index, std::vector<uint8_t>& page) {
    Endpoint& endpoint = endpoints_[endpoint_index];
    try {
        if (endpoint.bytes_sent - endpoint.bytes_acked >= kCceRequestRingBytes) {
            endpoint.bytes_acked =
                read_bytes_acked(mesh_device_, endpoint.device, endpoint.sender_logical, control_base_);
            if (endpoint.bytes_sent - endpoint.bytes_acked >= kCceRequestRingBytes) {
                return false;
            }
        }
        // Strict ordering: each write completes before the next is issued, so a ring page lands before the count
        // that covers it. A DRAM core's writes are otherwise relaxed.
        const uint32_t slot = (endpoint.bytes_sent / kRequestPageBytes) % kCceRequestRingSlots;
        write_dram_core_l1(
            mesh_device_,
            endpoint.device,
            endpoint.sender_logical,
            control_base_ + kCceRequestRingOffset + (slot * kRequestPageBytes),
            std::as_bytes(std::span(page.data(), kRequestPageBytes)),
            tt::umd::IoOrdering::Strict);
        // A separate write after the page's, so the kernel never sees a count that covers a page still in flight.
        const uint32_t bytes_sent = endpoint.bytes_sent + kRequestPageBytes;
        write_dram_core_l1(
            mesh_device_,
            endpoint.device,
            endpoint.sender_logical,
            control_base_ + kCceRingMetaOffset + offsetof(RingMeta, bytes_sent),
            std::as_bytes(std::span(&bytes_sent, 1)),
            tt::umd::IoOrdering::Strict);
        endpoint.bytes_sent = bytes_sent;
    } catch (const std::exception& error) {
        if (!endpoint.write_error_logged) {
            endpoint.write_error_logged = true;
            log_error(
                tt::LogMetal,
                "Tensor prefetcher: writing a request page to sender {} on device {} failed: {}",
                endpoint.sender_logical.str(),
                endpoint.device->id(),
                error.what());
        }
        return false;
    }
    return true;
}

// A status that cannot be read is unknown, not exited: the worker keeps the sender's pages until its no-progress limit.
bool Tl1RingTransport::target_exited(uint32_t endpoint_index) {
    Endpoint& endpoint = endpoints_[endpoint_index];
    try {
        const CceSenderStatus status =
            read_cce_sender_status(mesh_device_, endpoint.device, endpoint.sender_logical, control_base_);
        return status.state == static_cast<uint32_t>(CceSenderState::Stopped) ||
               status.state == static_cast<uint32_t>(CceSenderState::Failed);
    } catch (const std::exception& error) {
        if (!endpoint.status_error_logged) {
            endpoint.status_error_logged = true;
            log_error(
                tt::LogMetal,
                "Tensor prefetcher: reading the status of sender {} on device {} failed: {}",
                endpoint.sender_logical.str(),
                endpoint.device->id(),
                error.what());
        }
        return false;
    }
}

bool Tl1RingTransport::gave_up_on(uint32_t endpoint_index) const { return endpoints_[endpoint_index].given_up; }

// A sender frees a ring slot only once it has handled the command in it, so a ring also takes no page while its sender
// sits in a legitimate credit, CQ or drain wait, which wait_budget_cycles = 0 never bounds. Without an operation
// timeout, every page stays while the sender may still run: dropped, a page loses its data, and a dropped STOP leaves
// the sender running for good. With one, the page is dropped, and stop()'s WaitProgramDone times out on that sender.
bool Tl1RingTransport::give_up_on_stalled(
    uint32_t endpoint_index, uint8_t command, std::chrono::steady_clock::duration without_progress) {
    if (without_progress < no_progress_limit_) {
        return false;
    }
    Endpoint& endpoint = endpoints_[endpoint_index];
    const std::string sender_name =
        fmt::format("sender {} on device {}", endpoint.sender_logical.str(), endpoint.device->id());
    if (!give_up_on_running_senders_ && sender_running(endpoint)) {
        if (!endpoint.retention_logged) {
            endpoint.retention_logged = true;
            log_warning(
                tt::LogMetal,
                "Tensor prefetcher: the request ring of {} took no page for {:.0f} s; keeping its command {} page and "
                "later pages while it runs, because TT_METAL_OPERATION_TIMEOUT_SECONDS is unset",
                sender_name,
                std::chrono::duration<float>(without_progress).count(),
                command);
        }
        return false;
    }
    log_error(
        tt::LogMetal,
        "Tensor prefetcher: the request ring of {} took no page for {:.0f} s; dropping its command {} page and every "
        "later page to it but STOP",
        sender_name,
        std::chrono::duration<float>(without_progress).count(),
        command);
    endpoint.given_up = true;
    gave_up_on_any_sender_ = true;
    return true;
}

// Each ring pass reads sender TL1 over the device link, so the worker pauses rather than only yielding.
void Tl1RingTransport::pause_after_stalled_pass() const { std::this_thread::sleep_for(kRequestRingRetryInterval); }

std::optional<std::chrono::steady_clock::duration> Tl1RingTransport::gave_up_after() const {
    if (!gave_up_on_any_sender_) {
        return std::nullopt;
    }
    return no_progress_limit_;
}

void Tl1RingTransport::log_sender_statuses() const {
    for (const Endpoint& endpoint : endpoints_) {
        try {
            const CceSenderStatus status =
                read_cce_sender_status(mesh_device_, endpoint.device, endpoint.sender_logical, control_base_);
            log_error(
                tt::LogMetal,
                "Tensor prefetcher: sender {} on device {}: {}",
                endpoint.sender_logical.str(),
                endpoint.device->id(),
                format_cce_sender_status(status));
        } catch (const std::exception& error) {
            log_error(
                tt::LogMetal,
                "Tensor prefetcher: reading the status of sender {} on device {} failed: {}",
                endpoint.sender_logical.str(),
                endpoint.device->id(),
                error.what());
        }
    }
}

std::vector<std::string> Tl1RingTransport::failed_senders() const {
    std::vector<std::string> failed_senders;
    for (const Endpoint& endpoint : endpoints_) {
        try {
            const CceSenderStatus status =
                read_cce_sender_status(mesh_device_, endpoint.device, endpoint.sender_logical, control_base_);
            if (status.state == static_cast<uint32_t>(CceSenderState::Failed)) {
                std::string unread_pages;
                try {
                    unread_pages = std::to_string(unconsumed_pages(endpoint));
                } catch (const std::exception&) {
                    unread_pages = "an unknown number of";
                }
                failed_senders.push_back(fmt::format(
                    "sender {} on device {}: {}; {} request page(s) left unread in its ring, counting stop's DRAIN and "
                    "STOP",
                    endpoint.sender_logical.str(),
                    endpoint.device->id(),
                    format_cce_sender_status(status),
                    unread_pages));
            }
        } catch (const std::exception& error) {
            failed_senders.push_back(fmt::format(
                "sender {} on device {}: status unreadable: {}",
                endpoint.sender_logical.str(),
                endpoint.device->id(),
                error.what()));
        }
    }
    if (!failed_senders.empty()) {
        log_sender_statuses();
    }
    return failed_senders;
}

uint32_t Tl1RingTransport::page_bytes() const { return kRequestPageBytes; }

// The kernel acks a page only once it has handled it, Failed included, so the device's count, not the cached one,
// marks where the sender stopped reading.
uint32_t Tl1RingTransport::unconsumed_pages(const Endpoint& endpoint) const {
    const uint32_t bytes_acked =
        read_bytes_acked(mesh_device_, endpoint.device, endpoint.sender_logical, control_base_);
    return (endpoint.bytes_sent - bytes_acked) / kRequestPageBytes;
}

// A status that cannot be read is unknown and counts as running. Called on the host worker thread, so a failed read is
// not thrown; target_exited, which reads the same status earlier in the pass, logs it.
bool Tl1RingTransport::sender_running(const Endpoint& endpoint) const {
    try {
        const uint32_t state =
            read_cce_sender_status(mesh_device_, endpoint.device, endpoint.sender_logical, control_base_).state;
        return state == static_cast<uint32_t>(CceSenderState::Running) ||
               state == static_cast<uint32_t>(CceSenderState::WaitingCq) ||
               state == static_cast<uint32_t>(CceSenderState::Draining);
    } catch (const std::exception&) {
        return true;
    }
}

CceSenderStatus read_cce_sender_status(
    const MeshDevice& mesh_device, IDevice* device, const CoreCoord& sender_logical, uint32_t control_base) {
    CceSenderStatus status{};
    read_dram_core_l1(
        mesh_device,
        device,
        sender_logical,
        control_base + kCceSenderStatusOffset,
        std::as_writable_bytes(std::span(&status, 1)));
    return status;
}

std::string format_cce_sender_status(const CceSenderStatus& status) {
    return fmt::format(
        "state {} code {} request {} target 0x{:x} receiver {} sent {} acked {}",
        status.state,
        status.code,
        status.request_index,
        status.target_state_addr,
        status.receiver,
        status.sent,
        status.acked);
}

}  // namespace tt::tt_metal::distributed
