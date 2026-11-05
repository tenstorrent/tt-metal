// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"

namespace tt::tt_fabric::capture {

// A chip the reader found: what capture matches a manifest's chip and routers against.
struct FoundChip {
    // The reader's id for the chip, which reads address it by.
    int32_t chip = 0;
    // The chip's unique id, as the manifest's asic_id holds it.
    uint64_t asic_id = 0;
    // The architecture, spelled as the manifest's arch.name, such as "blackhole".
    std::string arch;
    // Each Ethernet channel's core, indexed by channel. Null for a channel the chip doesn't have.
    std::vector<std::optional<CoreXY>> eth_cores_by_channel;
};

// An Ethernet core topology discovery skipped, such as one whose base firmware heartbeat stalled. Its chip was still
// found, and the core can still be read.
struct DiscoverySkippedCore {
    uint64_t asic_id = 0;
    CoreXY core;
    std::string reason;

    bool operator==(const DiscoverySkippedCore&) const = default;
};

// What a reader knows about the chips it can reach, as plain data.
struct ClusterFacts {
    // The id the cluster descriptor gives this host, which the manifest records as run.host_name. Null when the
    // descriptor has none.
    std::optional<std::string> cluster_id;
    std::vector<FoundChip> chips;
    std::vector<DiscoverySkippedCore> skipped_cores;
};

// How capture reaches the chips. Reads only: nothing capture does writes to a chip. A read that fails throws, with a
// message saying why.
class DeviceReader {
public:
    virtual ~DeviceReader() = default;

    // Reads out.size() bytes of a core's L1, from `address`.
    virtual void read_l1(int32_t chip, CoreXY core, uint64_t address, std::span<std::byte> out) = 0;
    // Reads one 32-bit register.
    virtual uint32_t read_register(int32_t chip, CoreXY core, uint64_t address) = 0;
};

}  // namespace tt::tt_fabric::capture
