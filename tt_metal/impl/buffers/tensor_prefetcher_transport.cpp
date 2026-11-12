// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/tensor_prefetcher_transport.hpp"

#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_align.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "impl/buffers/h2d_socket_internal.hpp"
#include "impl/buffers/tensor_prefetcher_manager.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::distributed {

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

std::optional<std::string> H2DSocketTransport::failure_summary(uint32_t /*dropped_request_pages*/) const {
    return std::nullopt;
}

uint32_t H2DSocketTransport::page_bytes() const { return page_bytes_; }

uint32_t H2DSocketTransport::config_buffer_address(uint32_t endpoint) const {
    return sockets_.at(endpoint)->get_config_buffer_address();
}

}  // namespace tt::tt_metal::distributed
