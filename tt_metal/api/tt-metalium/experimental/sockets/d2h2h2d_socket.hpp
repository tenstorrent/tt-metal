// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// chip -> host -> host -> chip: D2HSocket --frames--> H2HSocket --frames--> H2DSocket.
// No threads. One caller-driven poll() drives all three.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <tt-metalium/experimental/sockets/internal/host_d2h_leg.hpp>
#include <tt-metalium/experimental/sockets/internal/host_h2d_leg.hpp>
#include <tt-metalium/experimental/sockets/internal/host_h2h_socket.hpp>
#include <tt-metalium/experimental/sockets/internal/host_l1_map.hpp>
#include <tt-metalium/experimental/sockets/internal/host_region.hpp>

namespace tt::tt_metal {
class IDevice;
namespace distributed {
class MeshDevice;
}
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

class D2H2H2DSocket {
public:
    struct Config {
        HostTopology topo{};
        uint32_t chip = 0;
        uint32_t cores = 0;
        uint32_t grid_width = 0;
        uint32_t grid_height = 0;
        // One size for the socket's life: the ring is ring_pages x (payload + trailer).
        uint32_t payload_bytes = 0;
        uint32_t ring_pages = kNumAliasRingSlots;
        uint32_t send_window = 0;
        // Give delivery its own L1 buffer, so a core can send and receive at once.
        bool bidirectional = false;
    };

    struct Counters {
        uint64_t sent = 0;
        uint64_t received = 0;
        uint64_t drained = 0;
        uint64_t retired = 0;
    };

    // Collective, and order-critical: the legs' rings must be overlaid before the region
    // is pinned, and the window created after. That order lives here and nowhere else.
    static std::unique_ptr<D2H2H2DSocket> create(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh,
        tt::tt_metal::IDevice* device,
        const Config& cfg,
        std::string& err);

    ~D2H2H2DSocket();

    D2H2H2DSocket(const D2H2H2DSocket&) = delete;
    D2H2H2DSocket& operator=(const D2H2H2DSocket&) = delete;

    // One non-blocking pass through all three legs. Returns units of progress; zero means
    // the caller may back off.
    uint32_t poll();

    const L1MapNew& l1() const;
    const Counters& counters() const;
    HostRegion& region() const;
    D2HLeg& d2h() const;
    H2HSocket& h2h() const;
    H2DLeg& h2d() const;

    std::string barrier();
    bool failed() const;
    std::string first_error() const;
    std::string describe() const;

private:
    D2H2H2DSocket();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental
