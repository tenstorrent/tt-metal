// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/tensor_prefetcher_transport.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
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

std::vector<uint32_t> read_sender_words(
    const MeshDevice& mesh_device,
    IDevice* device,
    const CoreCoord& sender_logical,
    uint32_t local_addr,
    uint32_t num_words) {
    const DramCoreL1Address source = dram_core_l1_address(mesh_device, device, sender_logical, local_addr);
    std::vector<uint32_t> words(num_words, 0);
    MetalContext::instance(mesh_device.impl().get_context_id())
        .get_cluster()
        .read_core(
            words.data(),
            num_words * static_cast<uint32_t>(sizeof(uint32_t)),
            tt_cxy_pair(device->id(), source.virtual_core),
            source.noc_addr);
    return words;
}

// Strict ordering: each write completes before the next is issued, so a ring page lands before the count that covers
// it. Cluster::write_core would otherwise resolve a DRAM core to relaxed ordering.
void write_sender_bytes(
    const MeshDevice& mesh_device,
    IDevice* device,
    const CoreCoord& sender_logical,
    uint32_t local_addr,
    const void* bytes,
    uint32_t num_bytes) {
    const DramCoreL1Address target = dram_core_l1_address(mesh_device, device, sender_logical, local_addr);
    MetalContext::instance(mesh_device.impl().get_context_id())
        .get_cluster()
        .write_core(
            bytes,
            num_bytes,
            tt_cxy_pair(device->id(), target.virtual_core),
            target.noc_addr,
            tt::umd::IoOrdering::Strict);
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

uint32_t H2DSocketTransport::page_bytes() const { return page_bytes_; }

uint32_t H2DSocketTransport::config_buffer_address(uint32_t endpoint) const {
    return sockets_.at(endpoint)->get_config_buffer_address();
}

Tl1RingTransport::Tl1RingTransport(
    const MeshDevice& mesh_device,
    const std::vector<IDevice*>& devices,
    const std::vector<CoreCoord>& sender_logical_cores,
    uint32_t control_base) :
    mesh_device_(mesh_device), control_base_(control_base) {
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
    const uint32_t ring_meta_addr = control_base_ + kCceRingMetaOffset;
    try {
        if (endpoint.bytes_sent - endpoint.bytes_acked >= kCceRequestRingBytes) {
            endpoint.bytes_acked = read_sender_words(
                                       mesh_device_,
                                       endpoint.device,
                                       endpoint.sender_logical,
                                       ring_meta_addr + offsetof(RingMeta, bytes_acked),
                                       1)
                                       .front();
            if (endpoint.bytes_sent - endpoint.bytes_acked >= kCceRequestRingBytes) {
                return false;
            }
        }
        const uint32_t slot = (endpoint.bytes_sent / kRequestPageBytes) % kCceRequestRingSlots;
        write_sender_bytes(
            mesh_device_,
            endpoint.device,
            endpoint.sender_logical,
            control_base_ + kCceRequestRingOffset + (slot * kRequestPageBytes),
            page.data(),
            kRequestPageBytes);
        // A separate write after the page's, so the kernel never sees a count that covers a page still in flight.
        const uint32_t bytes_sent = endpoint.bytes_sent + kRequestPageBytes;
        write_sender_bytes(
            mesh_device_,
            endpoint.device,
            endpoint.sender_logical,
            ring_meta_addr + offsetof(RingMeta, bytes_sent),
            &bytes_sent,
            sizeof(bytes_sent));
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

uint32_t Tl1RingTransport::page_bytes() const { return kRequestPageBytes; }

CceSenderStatus read_cce_sender_status(
    const MeshDevice& mesh_device, IDevice* device, const CoreCoord& sender_logical, uint32_t control_base) {
    const std::vector<uint32_t> words = read_sender_words(
        mesh_device,
        device,
        sender_logical,
        control_base + kCceSenderStatusOffset,
        sizeof(CceSenderStatus) / sizeof(uint32_t));
    CceSenderStatus status{};
    std::memcpy(&status, words.data(), sizeof(status));
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
