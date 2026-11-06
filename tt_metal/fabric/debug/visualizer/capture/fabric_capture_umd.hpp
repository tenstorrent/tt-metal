// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_device.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"

namespace tt::umd {
class Cluster;
}  // namespace tt::umd

namespace tt::tt_fabric::capture {

// What UMD's topology discovery found on `cluster`, regarding chips and their ethernet cores.
ClusterFacts collect_umd_cluster_facts(tt::umd::Cluster& cluster);

// Reads chips through a UMD cluster of its own, for reading only:
//   - it maps no hugepages, so large remote reads go through the Ethernet core's buffer;
//   - it's never started or closed, so it resets nothing and writes nothing to a chip;
//   - discovery skips an Ethernet core whose heartbeat has stalled instead of failing, since a hung run is what capture
//     is for.
// A read is addressed by UMD's chip id and the core's translated coordinates.
class UmdDeviceReader final : public DeviceReader {
public:
    // Opens every chip this process can see. `cluster_id` names the host as the run named it (TT_METAL_CLUSTER_ID);
    // null leaves UMD's default, the host name. Fails if the chips' unique ids aren't the ones they report, since
    // capture finds the manifest's chips by them.
    explicit UmdDeviceReader(std::optional<std::string> cluster_id);
    ~UmdDeviceReader() override;
    UmdDeviceReader(const UmdDeviceReader&) = delete;
    UmdDeviceReader& operator=(const UmdDeviceReader&) = delete;

    void read_l1(int32_t chip, CoreXY core, uint64_t address, std::span<std::byte> out) override;
    uint32_t read_register(int32_t chip, CoreXY core, uint64_t address) override;

    tt::umd::Cluster& get_cluster() { return *cluster_; }

private:
    std::unique_ptr<tt::umd::Cluster> cluster_;
};

}  // namespace tt::tt_fabric::capture
