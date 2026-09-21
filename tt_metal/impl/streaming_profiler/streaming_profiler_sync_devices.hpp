// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The device-to-device sync's device-side arm: the host clock and the probe that ties the root chip's refclk to it,
// the link planner the fabric routers consult for their role, and SyncDevices, which runs the tile mesh, the links
// and the probe for a capture.
#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "impl/context/context_types.hpp"
#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_engine.hpp"
#include "hostdev/streaming_profiler_common.h"

namespace tt {
class Cluster;
}
namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal {
class Program;
class IDevice;

namespace streaming_profiler {

// The host's clocks as the profiler reads them: the TSC, which every record's host time is in (tenths of a nanosecond
// of it, the API's host_clock), its rate, and its relation to std::chrono::steady_clock as the probe measures it.
// The host TSC, fenced, in ticks.
int64_t tsc_now() noexcept;
// TSC ticks per nanosecond, measured once per process against CLOCK_MONOTONIC_RAW.
double tsc_ticks_per_ns();
// host_clock units (tenths of a nanosecond) per TSC tick.
double units_per_tsc();
int64_t clock_ns(clockid_t id);

// The host TSC on CLOCK_MONOTONIC, one line between two NTP slews: mono_ns = mono0 + (tsc - tsc0) * ns_per_tick.
struct SteadySegment {
    int64_t tsc0 = 0, mono0 = 0;
    double ns_per_tick = 0.0;
    bool ok = false;
    int64_t mono_of(int64_t tsc) const { return mono0 + std::llrint(static_cast<double>(tsc - tsc0) * ns_per_tick); }
};

// The host TSC on steady_clock as the host probe measures it: one segment for the process, readable from any thread
// and cached per thread. No capture is involved, so the API's steady_time() reads it with no device open.
class SteadyView {
public:
    static void set(const SteadySegment& segment) noexcept;
    // A TSC/CLOCK_MONOTONIC pair taken here stands in until a probe publishes a segment.
    static int64_t mono_ns(int64_t tsc) noexcept;
};

// The root chip's refclk on the host TSC: tsc = a + b * refclk, fitted over the recent probe bursts.
struct HostLine {
    double a = 0.0, b = 0.0;
    double sigma_ns = 0.0;  // rms residual of the burst points about the line, in ns
    uint32_t bursts = 0;
    bool ok = false;
    double tsc_of(double refclk) const { return a + b * refclk; }
};

// Ties ONE chip's refclk to the host: bursts of back-to-back reads of that chip's PCIe-tile count-from-reset timer
// (the same distributed ordinary clock the eth tiles count, one NoC hop from the PCIe entry, nobody else's latch),
// each read bracketed by fenced TSC reads through a static TLB window (720 ns round trip), the tightest kept, a line
// fitted across bursts. Every other chip reaches this one through the eth link sync, so this is the fleet's only
// host relation; each burst's line becomes a node of the sync engine's host series. The same thread pairs TSC with
// CLOCK_MONOTONIC for the steady_clock view.
class HostProbe {
public:
    // Writes the host series of `map` while it runs.
    HostProbe(tt::Cluster& cluster, uint32_t chip_id, ClockMap& map);
    ~HostProbe();
    HostProbe(const HostProbe&) = delete;
    HostProbe& operator=(const HostProbe&) = delete;

    uint32_t chip_id() const { return chip_id_; }
    HostLine line() const;
    SteadySegment steady() const;
    // Ends the reads; the last line and segment stay readable. Must precede the device's teardown.
    void stop();

private:
    struct BurstPoint {
        double tsc, refclk;  // means of the kept reads
    };
    void run();
    uint32_t read_cfr_lo();
    bool burst(BurstPoint& out);
    void refit();
    void steady_pair();

    tt::Cluster& cluster_;
    const uint32_t chip_id_;
    ClockMap& map_;
    uint32_t pcie_x_ = 0, pcie_y_ = 0;  // translated
    tt::umd::TlbWindow* window_ = nullptr;
    uint32_t cfr_hi_ = 0, cfr_lo_last_ = 0;
    double ticks_per_ns_ = 0.0;
    std::deque<BurstPoint> points_;
    std::deque<std::pair<int64_t, int64_t>> pairs_;  // (tsc, mono)
    mutable std::mutex mu_;
    HostLine line_;
    SteadySegment steady_;
    uint64_t bursts_ = 0, reads_ = 0, kept_ = 0;
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

// Which eth links of a chip pair carry the device-to-device link sync and which end of each sends. The device side
// is tools/profiler/sync/eth_ptp_link.hpp on the constants of hostdev/streaming_profiler_common.h; without fabric
// the profiler runs the ends as resident kernels, with fabric the routers on the chosen links run them
// (fabric_erisc_router.cpp, LINK_SYNC_ROLE).
namespace link_sync {

enum class Role : uint32_t { None = 0, Sender = 1, Receiver = 2 };
// The profiler is on and TT_METAL_STREAMING_PROFILER_LINK_SYNC is not 0. Off leaves the link sync out of a profiler
// session: no links planned, every router's role None. The rest of the profiler runs as usual; it is how the sync's
// own cost is measured.
bool enabled();

struct Link {
    uint32_t chip_a = 0, chip_b = 0;  // chip_a < chip_b; chip_a's end sends
    CoreCoord eth_a, eth_b;           // logical eth cores
};

// Every eligible link between two connected chips, in the cluster's order: the lower chip's eth cores connected to
// the higher and the cores they connect to; with fabric on, only links whose two cores hold routers. The sync runs
// over all of them and averages a pair's links, so their path asymmetries average too.
std::vector<Link> links_between(const tt::Cluster& cluster, uint32_t chip_x, uint32_t chip_y);
// What the router on this eth core does for the sync.
Role role_of(const tt::Cluster& cluster, uint32_t chip, const CoreCoord& eth_logical);

}  // namespace link_sync

inline uint32_t packed_xy(const CoreCoord& c) {
    return (static_cast<uint32_t>(c.y) << 16) | static_cast<uint32_t>(c.x);
}

// One core as a kernel placement, the host's NoC writes and the NoC 0 grid each name it.
struct CoreCoords {
    CoreCoord logical, virt, phys;
};

// The idle-eth pusher's L1, carved from the top of IDLE_ETH UNRESERVED: the two socket configs, ctrl words
// (done/heartbeat, go, stop), one frame slot, the linked-core scratch, the sample ring, the sync ring of its clock
// model's points. link_ring is the link ends' sync ring on every active eth core (0 without active eth).
struct EthL1 {
    uint32_t cfg = 0, sync_cfg = 0, ctrl = 0, stage = 0, scratch = 0, ring = 0, sync_ring = 0, link_ring = 0;
};
constexpr uint32_t kEthPointUs = 1000;  // the open segment's line reaches the host at least this often

// The idle-eth pusher kernel over its L1 carve.
KernelHandle create_pusher_kernel(Program& program, const EthL1& l1, const CoreCoords& core);

// The device-to-device sync's use of the devices. At boot it measures each chip's tile clock offsets before any
// relay or pusher is on the NoC and plans the eth links; once the receiver drains the sockets it launches the link
// ends (resident kernels, or the fabric routers' roles) and the host probe; at quiesce it stops them. The stamps and
// clock samples travel the D2H path like every record and are consumed by the SyncEngine.
class SyncDevices {
public:
    struct Device {
        uint32_t chip_id = 0;
        IDevice* device = nullptr;
        std::vector<CoreCoords> tensix;  // the compute grid in core index order
        std::vector<CoreCoords>
            eth;  // the idle eth cores, the pusher first: the origin the tile offsets are solved against
        std::vector<CoreCoords> linked;  // the active eth cores the pusher drains, in their roster order
        double frequency_ghz = 0.0;
    };

    SyncDevices(ContextId context_id, const EthL1& eth_l1, uint32_t aeth_unreserved, uint32_t aeth_unres_size);
    ~SyncDevices();
    SyncDevices(const SyncDevices&) = delete;
    SyncDevices& operator=(const SyncDevices&) = delete;

    // Devices are indexed in the order added, the CaptureContext's device order.
    uint32_t add_device(Device d);
    void truncate(uint32_t n);
    // Every lane's offset into the pusher's wall domain, from the chip's tile clocks, into cap.tile_offset.
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
    void stop_links(tt::Cluster& cluster);
    // The two resident kernels of a link, compiled before either launches; false with the link skipped if one does
    // not compile.
    bool launch_link_ends(const CaptureContext::Link& L, uint32_t link_l1, ResidentSync& out);

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
