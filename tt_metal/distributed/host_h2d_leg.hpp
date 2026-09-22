// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The H2D leg: one H2DSocket per core in DEVICE_PULL, its ring aliased over that core's RX
// arena. The peer's RMA lands the frame in the ring, so publishing is a counter bump.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tt_metal/distributed/host_tasks.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace tt::tt_metal::experimental {

class H2DLeg {
public:
    struct Config {
        uint32_t cores = 0;
        uint32_t grid_width = 0;
        uint32_t page_bytes = 0;
        uint32_t ring_pages = 1;
        uint8_t* alias_region_base = nullptr;
    };

    // Must run before HostRegion::provision(); see RingAlias.
    static std::unique_ptr<H2DLeg> create(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh, const Config& cfg, std::string& err);

    ~H2DLeg();

    H2DLeg(const H2DLeg&) = delete;
    H2DLeg& operator=(const H2DLeg&) = delete;

    // Releases an arrived frame to the device. The bytes are already in the ring, so this
    // only advances bytes_sent. False means the ring has no room yet.
    bool publish(const DeliverTask& task);

    // Frames the device has pulled since the last call, per core. Drives the H2H credit.
    uint32_t drained(uint32_t core);

    // Per core, for the receiver kernel's create_receiver_socket_interface() arg.
    std::vector<uint32_t> config_addresses() const;

    std::string describe() const;
    std::string first_error() const;

private:
    H2DLeg();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental
