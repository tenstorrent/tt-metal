// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host side of the X280 -> host D2HSocket experiment (iteration 1).
// Constructs the socket on chip 0's L2CPU tile, then polls for pages the firmware pushes.

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

#include "x280_metal_transport.hpp"
#include "x280_address_map.h"

#include "common/heart_beat.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <thread>
#include <vector>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

namespace {
constexpr uint32_t kFifoSize = 4096;          // bytes; PCIe-aligned
constexpr uint32_t kPageSize = 64;            // bytes per message
constexpr uint32_t kConfigAddr = 0x08133000;  // LIM; must avoid 0x08131C00 (heartbeat) and 0x08132000 (table) TODO: Put this in a common spot, both sides need this.
constexpr auto kRunTime = std::chrono::seconds(30);

std::optional<Heartbeat> read_heartbeat(
    const std::shared_ptr<x280::X280MetalTransport>& transport, 
    int chip_id, 
    const tt::umd::CoreCoord& core
) {
    Heartbeat beat;
    transport->read_block(chip_id, core, X280_STANDALONE_HEARTBEAT_ADDR, &beat, sizeof(Heartbeat));
    if (beat.magic == static_cast<uint64_t>(X280_STANDALONE_HEARTBEAT_MAGIC)) {
        return beat;
    }

    return std::nullopt;
}

}  // namespace

int main() {
    auto mesh = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 1)));

    if (!experimental::GetMemoryPinningParameters(*mesh).can_map_to_noc) {
        std::cerr << "Host memory cannot be mapped to NOC on this system\n";
        mesh->close();
        return 1;
    }

    // Translated NOC coordinate of the first unharvested L2CPU (X280) tile on this chip.
    const ChipId chip = mesh->get_device(MeshCoordinate(0, 0))->id();
    auto l2cpu_cores = MetalContext::instance().get_cluster().get_soc_desc(chip).get_cores(
        tt::CoreType::L2CPU, tt::CoordSystem::TRANSLATED);
    if (l2cpu_cores.empty()) {
        std::cerr << "No L2CPU tiles on chip " << chip << "\n";
        mesh->close();
        return 1;
    }
    const MeshCoreCoord sender(MeshCoordinate(0, 0), CoreCoord(l2cpu_cores.front().x, l2cpu_cores.front().y));
    std::cout << "X280 tile (translated): " << l2cpu_cores.front().x << "," << l2cpu_cores.front().y << "\n";


    D2HSocket socket(*mesh, sender, kFifoSize, kConfigAddr);
    socket.set_page_size(kPageSize);
    std::cout << "Socket ready. Config block at LIM 0x" << std::hex << socket.get_config_buffer_address() << std::dec
              << ". Waiting for pages...\n";

    auto transport = x280::X280MetalTransport::bind(*mesh);

    auto heart_beat = read_heartbeat(
        transport,
        chip,
        l2cpu_cores.front()
    );

    // Check the hartbeat shit.
    if (!heart_beat.has_value()) {
        std::cout << "No heartbeat \n";
        return 1;
    }    

    std::vector<uint32_t> page(kPageSize / sizeof(uint32_t));
    const auto deadline = std::chrono::steady_clock::now() + kRunTime;
    uint32_t pages_read = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        if (socket.has_data()) {
            socket.read(page.data(), 1);
            ++pages_read;
            std::cout << "page " << pages_read << ": word0=0x" << std::hex << page[0] << std::dec << "\n";
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // std::cout << "Read " << pages_read << " pages; device bytes_sent=" << socket.bytes_sent() << "\n";
    mesh->close();
    return 0;
}
