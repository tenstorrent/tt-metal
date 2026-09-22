// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The host-to-host hop. Forwards each [payload|trailer] frame verbatim into the peer's RX
// arena; the trailer doubles as the arrival flag, so no separate notice is sent.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "tt_metal/distributed/host_tasks.hpp"
#include <tt-metalium/experimental/sockets/host_uva.hpp>

namespace tt::tt_metal::experimental {

class H2HSocket {
public:
    struct Config {
        HostTopology topo{};
        uint32_t chip = 0;
        uint32_t cores = 0;
        uint32_t page_bytes = 0;
        uint32_t ring_pages = 1;
        uint8_t* region_base = nullptr;
        uint64_t region_bytes = 0;
        // 0 => cores * ring_pages. Caps frames in flight across the whole socket.
        uint32_t send_window = 0;
    };

    // Collective: creates the window, so every rank must call it at the same point.
    static std::unique_ptr<H2HSocket> create(const Config& cfg, std::string& err);
    ~H2HSocket();

    H2HSocket(const H2HSocket&) = delete;
    H2HSocket& operator=(const H2HSocket&) = delete;

    // Queue a frame the D2H leg produced. False means the window is full; the caller
    // retries, which is what propagates backpressure to the device.
    bool submit(const SendTask& task);

    // Frees the D2H page behind a frame whose put has retired locally.
    using Retire = std::function<void(uint32_t core, uint32_t pages)>;
    // Hands an arrived frame to the H2D leg. False means it could not be taken this lap.
    using Deliver = std::function<bool(const DeliverTask&)>;

    // One non-blocking pass: start queued sends, retire completed ones, harvest arrivals.
    uint32_t poll(const Retire& retire, const Deliver& deliver);

    // Called by the H2D leg once the device has consumed a delivered frame. Posts the
    // credit that lets the origin reuse its slot.
    void consumed(uint32_t core, uint32_t pages);

    // Frames of this core's the peers have consumed. One peer per core today, so the
    // sum is that peer's count.
    uint64_t credit_total(uint32_t core) const;

    std::string barrier();
    bool failed() const;
    std::string first_error() const;
    std::string describe() const;

private:
    H2HSocket();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental
