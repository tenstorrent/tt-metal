// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "impl/context/context_types.hpp"
#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"
#include "impl/streaming_profiler/streaming_profiler_host_probe.hpp"
#include "hostdev/streaming_profiler_common.h"

namespace tt {
class Cluster;
}

namespace tt::tt_metal {
class Program;
class IDevice;

namespace streaming_profiler {

inline uint32_t packed_xy(const CoreCoord& c) {
    return (static_cast<uint32_t>(c.y) << 16) | static_cast<uint32_t>(c.x);
}

// One core as a kernel placement, the host's NoC writes and the NoC 0 grid each name it.
struct CoreCoords {
    CoreCoord logical, virt, phys;
};

// The idle-eth pusher's L1, carved from the top of IDLE_ETH UNRESERVED: socket config, ctrl words (done/heartbeat,
// go, stop), one frame slot, the linked-core scratch, the tile table, the sample ring. The tile measurement runs the
// pusher's kernel in measure-only mode, so it shares the carve.
struct EthL1 {
    uint32_t cfg = 0, ctrl = 0, stage = 0, scratch = 0, table = 0, ring = 0;
};
constexpr uint32_t kEthPointUs = 1000;  // the open segment's line reaches the host at least this often
// Tile table (hostdev EthTileTable): the header, a coordinate per Tensix tile and an int64 offset per tile.
constexpr uint32_t kEthTableBytes = 4096;
constexpr uint32_t kEthTableMaxTiles =
    (kEthTableBytes / sizeof(uint32_t) - kernel_profiler::ETH_TILE_XY_0) / (1 + kernel_profiler::ETH_TILE_OUT_WORDS);

// The idle-eth kernel over its L1 carve: the resident pusher, or the same kernel reading the tile table once and
// exiting.
KernelHandle create_pusher_kernel(Program& program, const EthL1& l1, const CoreCoords& core, bool measure_only);

// The device-to-device sync's use of the devices. At boot it measures each chip's tile clock offsets before any
// relay or pusher is on the NoC and plans the eth links; once the receiver drains the sockets it launches the link
// ends (resident kernels, or the fabric routers' roles) and the host probe; at quiesce it stops them. The stamps and
// clock samples travel the D2H path like every record and are consumed by D2dSyncConsumer.
class SyncDevices {
public:
    struct Device {
        uint32_t chip_id = 0;
        IDevice* device = nullptr;
        std::vector<CoreCoords> tensix;  // the compute grid in core index order
        std::vector<CoreCoords>
            eth;  // the idle eth cores, the pusher first: the origin the tile offsets are solved against
        double frequency_ghz = 0.0;
    };

    SyncDevices(ContextId context_id, const EthL1& eth_l1, uint32_t aeth_unreserved, uint32_t aeth_unres_size);
    ~SyncDevices();
    SyncDevices(const SyncDevices&) = delete;
    SyncDevices& operator=(const SyncDevices&) = delete;

    // Devices are indexed in the order added, the CaptureContext's device order.
    uint32_t add_device(Device d);
    void truncate(uint32_t n);
    // Reads every tile from every idle eth core and solves the offsets into cap.tile_offset. Nothing else of ours
    // may be on the NoC.
    void measure_tiles(uint32_t di, CaptureContext::Device& cap);
    // Every eligible eth link between the devices, all of a pair's links.
    void plan_links();
    // The host probe on the root chip: the first device with an idle eth core.
    void start_probe();
    // MUST follow the receiver's ingest threads draining the sockets: a link end stamps into a ring an eth pusher
    // ships, and a pusher blocked on a full FIFO wedges the end.
    void launch_links();
    // The probe, then every link end: the sender first (its last round still echoes off the live receiver).
    void stop(tt::Cluster& cluster);
    const std::vector<CaptureContext::Link>& links() const { return links_; }
    uint32_t root_dev() const { return root_dev_; }

private:
    struct DeviceState {
        Device d;
    };
    struct ResidentSync {
        std::unique_ptr<Program> ps, pr;
        IDevice* dev_a = nullptr;
        IDevice* dev_b = nullptr;
        CoreCoord virt_a, virt_b;
        uint32_t chip_a = 0, chip_b = 0;
        uint32_t stop_a = 0, stop_b = 0;
    };
    std::vector<double> solve_tiles(uint32_t di);
    void stop_links(tt::Cluster& cluster);

    const ContextId context_id_;
    const EthL1 eth_l1_;
    const uint32_t aeth_unreserved_, aeth_unres_size_;  // ACTIVE_ETH unreserved region: the link ends' L1
    std::vector<DeviceState> devices_;
    std::vector<CaptureContext::Link> links_;
    std::vector<ResidentSync> link_syncs_;
    bool fabric_link_sync_ = false;  // fabric is on: the routers on the planned links run the ends, nothing is launched
    std::shared_ptr<HostProbe> host_probe_;
    uint32_t root_dev_ = 0;
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
