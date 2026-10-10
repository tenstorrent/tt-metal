// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// E2H: ERISC -> host. Zero copy: each D2H socket's shm FIFO is aliased onto its Tx arena.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tt_metal/distributed/erisc_bridge_tasks.hpp"

namespace tt::tt_metal::distributed {
class D2HSocket;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental {

class E2HLeg {
public:
    struct Config {
        uint32_t arenas = 1;          // (link, sender channel) pairs in the region
        uint32_t chans_per_link = 1;  // flat across every VC, or arena indices collide
        uint32_t packet_capacity = 0;
        uint32_t ring_pages = 1;
        uint32_t link_idx = 0;
        uint8_t* alias_region_base = nullptr;  // where the sockets' shm is already mapped
    };

    // One socket per bridged sender channel, already aliased: sockets -> alias -> create().
    static std::unique_ptr<E2HLeg> create(
        std::vector<std::unique_ptr<tt::tt_metal::distributed::D2HSocket>> socks, const Config& cfg, std::string& err);

    ~E2HLeg();
    E2HLeg(const E2HLeg&) = delete;
    E2HLeg& operator=(const E2HLeg&) = delete;

    // A sink returning false did not take the task; it is re-offered on the next poll.
    using Sink = std::function<bool(const BridgeSendTask&)>;
    uint32_t poll(const Sink& sink);

    // Pages stay owned until retired; crediting earlier lets the ERISC overwrite them.
    void retire(uint32_t arena, uint32_t pages);

    uint32_t page_size() const;
    uint64_t unarmed() const;       // polls that found nothing landed yet
    uint64_t out_of_order() const;  // frames whose ordering_cntr or lap tag was not the next expected
    uint64_t drained_total() const;
    std::string describe() const;

private:
    E2HLeg();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental
