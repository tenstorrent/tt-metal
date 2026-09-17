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
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include "impl/context/context_types.hpp"
#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_devices.hpp"

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
    // Start the idle-eth clock pushers sampling. MUST be called after the receiver's ingest threads are draining
    // the sockets: a pusher sampling earlier fills its FIFO before any consumer attaches, and the consumers then
    // start behind frames the device is already overwriting.
    void release_eth_pushers();
    ContextId context_id() const { return context_id_; }
    // The sync's device-side arm; valid once boot() returned a device.
    SyncDevices& sync() { return *sync_; }

private:
    static constexpr uint32_t kMaxRelays = 8;

    // One D2H drainer of a device, a DRISC relay or the idle-eth pusher: a resident program on a core no workload
    // uses, its own socket, and two control words the host drives it with. Launched outside the command queue: a
    // DRAM-only or idle-eth program touches no fast-dispatch resource, so it stays up across every workload, while
    // going through the CQ would deadlock the first Finish().
    struct Drainer {
        std::unique_ptr<Program> program;
        CoreCoord logical, virt;
        uint32_t sock_idx = 0;    // into CapturedDevice::sockets
        uint64_t state_addr = 0;  // the Drained/Done word as the host addresses it; the heartbeat is the word after
        uint64_t stop_addr = 0;
    };
    struct WorkerCore {
        CoreCoord logical, physical, virt;
    };
    // The idle-eth core that pushes its own profiler ring and its linked active eth cores' over its own socket. Out
    // of the relay roster entirely; enumerated to the decoder as a standard 5-lane core with idle siblings.
    struct EthPusher : Drainer {
        CoreCoord phys;
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
        std::vector<Drainer> relays;    // at most kMaxRelays; their sockets are the prefix of out.sockets
        std::optional<EthPusher> eth;
        std::vector<SyncDevices::EthCore> idle_eth;  // every idle eth core, the pusher first

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
    // Idle-eth cores as padded standard cores in the decode roster (never the relay roster).
    void enumerate_eth_cores(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx);
    // Builds the eth core socket, launches the pusher and confirms its heartbeat. False: this pusher is dropped;
    // the capture continues without it.
    bool launch_eth_pusher(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        DeviceCtx& ctx,
        const distributed::MeshCoordinate& coord);
    // Stops one drainer through its stop word and reports Drained, then Done, through `on_state`; one that does not
    // finish within 10 s is a fault.
    void stop_drainer(
        uint32_t device_index, const DeviceCtx& ctx, const Drainer& r, const char* what, const RelayStateFn& on_state);
    void write_eth_ctrl_word(const DeviceCtx& ctx, const CoreCoord& virt, uint32_t index, uint32_t value);
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
    bool eth_ok_ = false;  // an idle-eth pusher fits its L1
    uint64_t eth_prof_l1_ = 0;
    EthL1 eth_l1_;
    bool aeth_ok_ = false;  // ACTIVE_ETH profiler base resolved: the pusher can drain active eth cores
    uint64_t aeth_prof_l1_ = 0;
    // GDDR spool: the HAL's PROFILER DRAM region, which MetalEnv sizes for the spool when the streaming profiler
    // is on, so it lies below every allocator's unreserved base. Bytes 0 = direct push.
    uint32_t spool_bytes_ = 0;
    uint32_t spool_addr_ = 0;
    std::vector<DeviceCtx> devices_;
    std::unique_ptr<SyncDevices> sync_;
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
