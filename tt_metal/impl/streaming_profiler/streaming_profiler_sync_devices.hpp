// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The device-to-device sync's device-side arm: the host clock and the probe that ties the root chip's refclk to it,
// the link planner the fabric routers consult for their role, and SyncDevices, which runs the tile mesh, the links
// and the probe for a capture.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "impl/context/context_types.hpp"
#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_engine.hpp"
#include "hostdev/streaming_profiler_common.h"
#include "hostdev/streaming_profiler_sync.h"

namespace tt {
class Cluster;
}
namespace tt::umd {
class IoWindow;
}

namespace tt::tt_metal {
class Program;
class IDevice;
class Hal;

namespace streaming_profiler {

// The host TSC, fenced, in ticks.
int64_t tsc_now() noexcept;
// host_clock units (tenths of a nanosecond) per TSC tick, in 32-bit fixed point: host_clock is the TSC scaled by
// exactly this, so every conversion between the two is integer arithmetic at any uptime.
int64_t units_per_tsc_q32();
// The same scale as a double, exactly units_per_tsc_q32() / 2^32.
double units_per_tsc();
// A TSC tick on host_clock, rounded to the nearest unit, and back.
int64_t units_of_tsc(int64_t tsc);
int64_t tsc_of_units(int64_t units);

// A host TSC tick on steady_clock as the host probe measures it: the clock map's steady series (ClockMap::steady_ns),
// readable from any thread. No capture is involved, so the API's steady_time() reads it with no device open; a
// TSC/CLOCK_MONOTONIC pair this thread takes once stands in until a probe has taken one.
int64_t steady_mono_ns(int64_t tsc) noexcept;

// The root chip's refclk on the host TSC, both from the clock map's bases: tsc = a + b * refclk, fitted over the recent
// probe bursts.
struct HostLine {
    double a = 0.0, b = 0.0;
    double sigma_ns = 0.0;  // rms residual of the burst points about the line, in ns
    uint32_t bursts = 0;
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

    // Ends the reads. Must precede the device's teardown.
    void stop();

private:
    struct BurstPoint {
        double tsc, refclk;  // means of the kept reads
    };
    void run();
    BurstPoint burst();
    void refit();
    void steady_pair();

    tt::Cluster& cluster_;
    const uint32_t chip_id_;
    ClockMap& map_;
    std::unique_ptr<tt::umd::IoWindow> window_;
    uint32_t cfr_hi_ = 0, cfr_lo_last_ = 0;
    double ticks_per_ns_ = 0.0;
    int64_t rtt_floor_ = std::numeric_limits<int64_t>::max();  // fastest read round trip seen, TSC ticks
    std::deque<BurstPoint> points_;
    int64_t pair_tsc_ = 0, pair_mono_ = 0;  // the newest steady pair
    double ns_per_tick_ = 0.0;              // CLOCK_MONOTONIC ns per TSC tick between the two newest pairs
    HostLine line_;
    uint64_t bursts_ = 0, reads_ = 0, kept_ = 0;
    std::mutex stop_mutex_;
    std::condition_variable stop_cv_;
    bool stop_ = false;
    std::thread thread_;
};

// Which eth links of a chip pair carry the device-to-device link sync and which end of each sends. The device side
// is tools/profiler/sync/eth_ptp_link.hpp on the layouts of hostdev/streaming_profiler_sync.h; without fabric
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
// The link end's kernel_profiler::LinkSyncL1 on every active eth core: the top of ACTIVE_ETH UNRESERVED, the same place
// whether a resident kernel or a router runs the end.
uint32_t l1_addr(const Hal& hal);

}  // namespace link_sync

inline uint32_t packed_xy(const CoreCoord& c) {
    return (static_cast<uint32_t>(c.y) << 16) | static_cast<uint32_t>(c.x);
}
// The cores in (y, x) order, so a choice among them is stable run to run.
std::vector<CoreCoord> sorted_yx(const std::unordered_set<CoreCoord>& cores);

// One core as a kernel placement, the host's NoC writes and the NoC 0 grid each name it.
struct CoreCoords {
    CoreCoord logical, virt, phys;
};

// The idle-eth pusher's L1, carved from the top of IDLE_ETH UNRESERVED: the two socket configs, ctrl words
// (done/heartbeat, go, stop), one frame slot, the linked-core scratch, the 64 B its PLL reads land in, the sync ring of
// its clock model's points. link_ring is the link ends' sync ring on every active eth core.
struct EthL1 {
    uint32_t cfg = 0, sync_cfg = 0, ctrl = 0, stage = 0, scratch = 0, pll = 0, sync_ring = 0, link_ring = 0;
};

// The idle-eth pusher kernel over its L1 carve, reading AICLK's PLL on the ARC tile at `arc` (translated), and the
// drainer that ships its ring (and the chip's eth cores' frames) from a second idle core: both cores carve their own
// L1 alike, so one EthL1 addresses either.
KernelHandle create_pusher_kernel(Program& program, const EthL1& l1, const CoreCoords& core, const CoreCoord& arc);
KernelHandle create_drainer_kernel(Program& program, const EthL1& l1, const CoreCoords& core, const CoreCoords& pusher);

// The device-to-device sync's use of the devices. At boot it measures each chip's tile clock offsets before any
// relay or pusher is on the NoC and plans the eth links; once the receiver drains the sockets it launches the link
// ends (resident kernels, or the fabric routers' roles) and the host probe; at quiesce it stops them. The stamps and
// clock samples travel the D2H path like every record and are consumed by the SyncEngine.
class SyncDevices {
public:
    struct Device {
        uint32_t chip_id = 0;
        IDevice* device = nullptr;
        CoreCoords pusher;               // the pusher's idle eth core: the origin the tile offsets are solved against
        std::vector<CoreCoords> tensix;  // the compute grid in core index order
        std::vector<CoreCoords> linked;  // the active eth cores the pusher drains, in their roster order
        CoreCoords drainer;              // the idle eth core whose anchors audit the pusher's clock model
    };

    explicit SyncDevices(ContextId context_id);
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
    // The host probe on the root chip, the first device.
    void start_probe();
    // MUST follow the receiver's ingest threads draining the sockets: a link end stamps into a ring an eth pusher
    // ships, and a pusher blocked on a full FIFO wedges the end.
    void launch_links();
    // The probe, then every link end: the sender first (its last round still echoes off the live receiver).
    void stop(tt::Cluster& cluster);
    const std::vector<CaptureContext::Link>& links() const { return links_; }

private:
    void stop_links(tt::Cluster& cluster);
    // The two resident kernels of a link, sender then receiver, both compiled before either launches.
    std::pair<std::unique_ptr<Program>, std::unique_ptr<Program>> launch_link_ends(const CaptureContext::Link& L);

    const ContextId context_id_;
    std::vector<Device> devices_;
    std::vector<CaptureContext::Link> links_;
    // The kernel_profiler::LinkSyncL1 of every link end while the links run, the same address on every end.
    std::optional<uint32_t> link_l1_;
    // Each link's resident sender and receiver, parallel to links_; empty when the fabric routers run the ends.
    std::vector<std::pair<std::unique_ptr<Program>, std::unique_ptr<Program>>> resident_;
    std::unique_ptr<HostProbe> host_probe_;
    bool fabric_link_sync_ = false;  // fabric is on: the routers on the planned links run the ends, nothing is launched
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
