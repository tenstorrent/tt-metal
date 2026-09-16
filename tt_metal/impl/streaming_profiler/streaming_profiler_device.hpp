// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streaming profiler device layer: relay bring-up, host<->device clock sync, quiesce, and the teardown
// completeness check for one MeshDevice's local Blackhole devices.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include "impl/context/context_types.hpp"
#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"
#include "impl/streaming_profiler/streaming_profiler_host_probe.hpp"

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
class MeshCoordinate;
class D2HSocket;
}  // namespace distributed
class Program;
class IDevice;

namespace streaming_profiler {

// One device whose relays came up: the sockets the receiver ingests and what its consumers need to decode them.
struct CapturedDevice {
    uint32_t chip_id = 0;
    int numa_node = -1;                                            // the node the sockets bind their FIFOs to
    std::vector<std::unique_ptr<distributed::D2HSocket>> sockets;  // the relays' in relay order, then the eth pushers'
    uint32_t n_relay_sockets = 0;
    CaptureContext::Device ctx;
    DeviceClock clock;
};

// What quiesce() reports per (device index, socket index): Drained, the relay pushed its last page and waits on
// its socket barrier; Done, it saw every byte acked.
enum class RelayState { Running, Drained, Done };
using RelayStateFn = std::function<void(uint32_t device_index, uint32_t socket_index, RelayState)>;

// The relays of one capture: up to kMaxRelays DRISCs per device, each sweeping a band of the worker grid into its
// own socket. Destruction releases the spool, so it must precede the mesh allocator's.
class Devices {
public:
    Devices();
    ~Devices();
    Devices(const Devices&) = delete;
    Devices& operator=(const Devices&) = delete;

    // Brings the relays up on every eligible local Blackhole device and syncs each clock. A device that fails is
    // logged and left unarmed, so its markers are overwritten rather than blocked on.
    std::vector<CapturedDevice> boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    // Stops every relay through its stop word (1 = quiesce, then 2 = release the NIU once it is done), reporting
    // each relay's states through `on_state` (may be empty), then disarms the device's producers. A relay that does
    // not finish within 10 s is a fault. The resident idle FW is left alone.
    void quiesce(const RelayStateFn& on_state);
    // After the relays swept to empty and the capture detached: the producer-owned stall counters, and every
    // worker lane's own tail against the consumed-words mirror `heads` (empty when nothing decoded the device).
    void verify_completeness(uint32_t device_index);
    // Launch the planned boot-time eth link syncs. MUST be called after the host receiver's ingest threads
    // are draining the sockets, or the armed sync kernels wedge on a FIFO no reader empties.
    void run_link_sync();
    // Start the idle-eth clock pushers sampling. MUST be called after the receiver's ingest threads are draining
    // the sockets: a pusher sampling earlier fills its FIFO before any consumer attaches, and the consumers then
    // start behind frames the device is already overwriting.
    void release_eth_pushers();
    // Stop every resident link sync at quiesce: the sender first (its final round still echoes off the live
    // receiver), then the receiver, each confirmed by its done word.
    void stop_link_syncs(tt::Cluster& cluster);
    // The eth link syncs launch_link_sync() ran at boot, for the consumers' CaptureContext.
    const std::vector<CaptureContext::Link>& links() const { return links_; }
    // The root chip's refclk on the host TSC (null when no chip has an eth tracker), and that chip's index.
    uint32_t root_dev() const { return root_dev_; }

private:
    static constexpr uint32_t kMaxRelays = 8;

    struct Relay {
        // Launched outside the command queue: a DRAM-only program touches no fast-dispatch resource, so it stays
        // resident across every workload, while going through the CQ would deadlock the first Finish().
        std::unique_ptr<Program> program;
        CoreCoord logical;
        CoreCoord virt;
    };
    struct WorkerCore {
        CoreCoord logical, physical, virt;
    };
    // An idle-eth core that pushes its own profiler ring (and its linked active eth core) over its own socket.
    // Out of the relay roster entirely; enumerated to the decoder as a standard 5-lane core with idle siblings.
    struct EthPusher {
        CoreCoord logical, virt, phys;
        std::unique_ptr<Program> program;
        uint32_t sock_idx = 0;  // index into out.sockets, after the relays
        // A second idle eth core that reads the Tensix tiles in this pusher's own column at boot: the pusher reaches
        // those over a one-ring path whose split differs from every other tile's; the helper, in another column,
        // reaches them over two rings like the rest, and every eth tile's wall clock is the same clock.
        bool has_helper = false;
        CoreCoord helper_logical, helper_virt;
        // The chip's active eth cores this pusher drains (their rings are NoC-read, their heads written back):
        // they run the fabric router and can spend no cycles on egress, so the idle sibling carries them.
        struct Linked {
            CoreCoord logical, virt;
            uint32_t xy = 0;       // packed virtual XY, the frame identity the decoder resolves
            uint32_t prof_l1 = 0;  // that core type's profiler L1 base (ACTIVE_ETH)
        };
        std::vector<Linked> linked;
    };
    struct DeviceCtx {
        uint32_t chip_id = 0;
        IDevice* device = nullptr;
        CapturedDevice out;
        std::vector<WorkerCore> cores;  // the compute grid, row-major, so a relay's band is a contiguous run
        Relay relays[kMaxRelays];
        uint32_t n_relays = 0;
        std::vector<EthPusher> eth;  // idle-eth clock pushers, one socket each

        DeviceCtx();
        ~DeviceCtx();
        DeviceCtx(DeviceCtx&&) noexcept;
    };
    // The DRISC L1 layout every relay shares: staging slots, per-core scratch, the done (with heartbeat) and stop
    // words, and the socket config at the top.
    struct RelayL1 {
        uint32_t stage_base = 0, n_stage = 0, core_records = 0, done = 0, stop = 0, cfg = 0;
    };

    bool boot_device(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        DeviceCtx& ctx,
        const distributed::MeshCoordinate& coord);
    void enumerate_worker_grid(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx);
    // Relay count, each relay's DRAM view and core, and a check that firmware left that core's NIUs in stream
    // mode. False: no relay can run on this device.
    bool choose_relay_cores(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx);
    void reserve_spool();
    // Configures the relay's TLB window, builds its socket, launches it and confirms its heartbeat. False means
    // capture must be abandoned for this device.
    bool launch_relay(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        DeviceCtx& ctx,
        const distributed::MeshCoordinate& coord,
        uint32_t d);
    // Idle-eth cores as padded standard cores in the decode roster (never the relay roster); false = none.
    void enumerate_eth_cores(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx);
    // Builds the eth core socket, launches the pusher and confirms its heartbeat. False: this pusher is dropped;
    // the capture continues without it.
    bool launch_eth_pusher(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        DeviceCtx& ctx,
        const distributed::MeshCoordinate& coord,
        uint32_t k);
    void write_eth_ctrl_word(const DeviceCtx& ctx, const CoreCoord& virt, uint32_t index, uint32_t value);
    // The per-tile wall-clock offsets the pusher measured before its heartbeat started, into the capture context.
    void read_tile_offsets(DeviceCtx& ctx);
    // One-shot device<->device link sync at boot: the eth sync kernels on every connected active-eth pair of local
    // devices, whose SYNC-ZONE zones the idle pushers then drain. Only when fabric is DISABLED: after fabric init
    // those cores hold live routers, and a launch onto one would write a launch message into a router.
    void plan_link_sync();
    // PROFILER_ARMED on every core the relays drain: set once they are up (producers boot unarmed and never block on
    // a full ring until then), cleared once every relay is done so a producer blocked on a full ring is released.
    void set_producers_armed(const DeviceCtx& ctx, bool armed);
    void write_ctrl_word(const DeviceCtx& ctx, const CoreCoord& virt, uint32_t index, uint32_t value);
    // A DRISC L1 address as the host reaches it over the NoC.
    uint64_t relay_noc_addr(uint32_t l1) const { return drisc_l1_noc_ + (l1 - drisc_l1_base_); }

    ContextId context_id_{0};
    uint64_t prof_l1_ = 0;        // Tensix profiler L1 base (control vector, then the per-RISC rings)
    uint32_t drisc_l1_base_ = 0;  // DRISC L1 unreserved region, and its NoC-addressable base
    uint64_t drisc_l1_noc_ = 0;
    uint32_t slot_bytes_ = 0;  // staging slot; mirrors the relay kernel's kSlotWords
    RelayL1 l1_;
    // Idle-eth pusher L1 (IDLE_ETH): the profiler base, and carved from the top of UNRESERVED: socket config,
    // ctrl words (done/heartbeat, stop), one frame slot, the linked-core scratch, the tile table.
    bool eth_ok_ = false;
    uint64_t eth_prof_l1_ = 0;
    uint32_t eth_cfg_ = 0, eth_ctrl_ = 0, eth_stage_ = 0, eth_scratch_ = 0, eth_table_ = 0, eth_ring_ = 0;
    bool aeth_ok_ = false;  // ACTIVE_ETH profiler base resolved: the pusher can drain active eth cores
    uint64_t aeth_prof_l1_ = 0;
    uint32_t aeth_unreserved_ = 0, aeth_unres_size_ = 0;  // ACTIVE_ETH unreserved region: the resident sync stop word
    // Resident link sync programs (a sender+receiver pair per link), launched after the receiver is up and
    // stopped at quiesce. Kept alive here so the Program objects outlive the run, like the relays and pushers.
    struct ResidentSync {
        std::unique_ptr<Program> ps, pr;
        IDevice* dev_a = nullptr;
        IDevice* dev_b = nullptr;
        CoreCoord virt_a, virt_b;
        uint32_t chip_a = 0, chip_b = 0;
        uint32_t stop_a = 0, stop_b = 0;
    };
    std::vector<ResidentSync> link_syncs_;
    bool fabric_link_sync_ = false;  // fabric is on: the routers on the planned links run the ends, nothing is launched
    // GDDR spool: the HAL's PROFILER DRAM region, which MetalEnv sizes for the spool when the streaming profiler
    // is on, so it lies below every allocator's unreserved base. Bytes 0 = direct push.
    uint32_t spool_bytes_ = 0;
    uint32_t spool_addr_ = 0;
    std::vector<DeviceCtx> devices_;
    std::vector<CaptureContext::Link> links_;
    std::shared_ptr<HostProbe> host_probe_;  // on the root chip: the lowest device with an eth tracker
    uint32_t root_dev_ = 0;
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
