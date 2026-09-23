// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The D2H leg: one D2HSocket per core, its FIFO aliased over that core's TX arena so the
// device's pages land inside the pinned region the transport already reads from.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tt_metal/distributed/host_tasks.hpp"
#include <tt-metalium/experimental/sockets/host_uva_frame.hpp>
#include <tt-metalium/experimental/sockets/host_uva.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace tt::tt_metal::experimental {

class D2HLeg {
public:
    struct Config {
        uint32_t cores = 0;
        uint32_t grid_width = 0;
        uint32_t payload_bytes = 0;
        // Ring depth in frames. Must equal what the H2H window and the peer's ring use.
        uint32_t ring_pages = 1;
        // HostRegion::reserved_base(). The overlay must precede provision(), which pins.
        // L1 word this leg pokes with the far device's consumed count; 0 disables.
        uint32_t consumed_addr = 0;
        uint8_t* alias_region_base = nullptr;
    };

    // Must be called BEFORE HostRegion::provision(): MAP_FIXED replaces the pages behind
    // the arenas, and a pin taken first would go on naming the old ones.
    static std::unique_ptr<D2HLeg> create(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh, const Config& cfg, std::string& err);

    ~D2HLeg();

    D2HLeg(const D2HLeg&) = delete;
    D2HLeg& operator=(const D2HLeg&) = delete;

    // Hands each complete frame to `sink`, stopping at the first refusal so the caller can
    // apply backpressure. Non-blocking; returns how many frames were accepted.
    using Sink = std::function<bool(const SendTask&)>;
    uint32_t poll(const Sink& sink);

    // Frees `pages` of this core's FIFO once the transport has finished reading them.
    // Contiguous prefix only: bytes_acked is one counter and cannot free a page by name.
    void retire(uint32_t core, uint32_t pages);

    // Publishes the far device's consumed count to this core, for tt_uva_sync().
    void credit(uint32_t core, uint64_t pages);

    uint32_t page_size() const;
    uint32_t cores() const;
    // Per core, for the kernel's create_sender_socket_interface() runtime arg.
    std::vector<uint32_t> config_addresses() const;
    std::string describe() const;
    std::string first_error() const;

private:
    D2HLeg();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental
