// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streaming profiler device layer: the D2H drainers' bring-up and quiesce, the decode roster, and the teardown
// completeness check for one MeshDevice's local Blackhole devices.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
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
class Hal;

namespace streaming_profiler {

// One device whose relays came up: the sockets the receiver ingests and what its consumers need to decode them.
struct CapturedDevice {
    uint32_t chip_id = 0;
    int numa_node = -1;                                            // the node the sockets bind their FIFOs to
    std::vector<std::unique_ptr<distributed::D2HSocket>> sockets;  // the relays' in relay order, then the pusher's two
    uint32_t n_relay_sockets = 0;
    uint32_t sync_socket = UINT32_MAX;  // the pusher's sync socket: the sync's records, for the sync engine alone
    CaptureContext::Device ctx;
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

    // Brings the drainers up on every eligible local Blackhole device. A device that fails is logged and left
    // unarmed, so its markers are overwritten rather than blocked on; one whose bring-up throws stops every drainer
    // already up and returns nothing.
    std::vector<CapturedDevice> boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    // Stops every drainer that came up through its stop word (1 = quiesce, then 2 = release the NIU once it is
    // done), reporting each relay's states through `on_state`, then disarms the device's producers. Empty
    // `on_state` means no consumer drains the sockets, and the host acks their pages itself. A drainer that does
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
    // uses, its own socket, and a control block the host drives it with (done and heartbeat words, then the stop
    // word one kRelayCtrlWordStride up). Launched outside the command queue: a DRAM-only or idle-eth program touches
    // no fast-dispatch resource, so it stays up across every workload, while going through the CQ would deadlock
    // the first Finish().
    struct Drainer {
        std::unique_ptr<Program> program;
        CoreCoords core;
        uint32_t sock_idx = 0;  // into CapturedDevice::sockets; n_sockets of them, the pusher's sync socket second
        uint32_t n_sockets = 1;
        uint64_t state_addr = 0;  // the control block as the host addresses it
        uint64_t stop_addr = 0;
    };
    struct DrainerL1 {
        HalProgrammableCoreType core_type;
        uint32_t cfg = 0;       // the socket config
        uint32_t sync_cfg = 0;  // the pusher's sync socket config; 0 = one socket
        uint32_t fifo_bytes = 0;
    };
    // A 5-lane core in the decode roster and the L1 base of its control vector. A blocking producer is armed for the
    // capture and waits on a full ring for its drainer; the pusher's linked routers are left non-blocking and
    // overwrite instead, so a router never wedges while the pusher is briefly behind.
    struct Producer : CoreCoords {
        uint64_t prof_l1 = 0;
        bool blocking = false;
    };
    struct DeviceCtx {
        uint32_t chip_id = 0;
        IDevice* device = nullptr;
        CapturedDevice out;
        std::vector<Producer> producers;  // the worker grid row-major, then the pusher, then its linked cores
        uint32_t n_workers = 0;           // the relays' bands cover this prefix
        std::vector<Drainer> relays;      // at most kMaxRelays; their sockets are the prefix of out.sockets
        std::optional<Drainer> pusher;
        std::vector<CoreCoords> idle_eth;  // every idle eth core, the pusher first

        DeviceCtx();
        ~DeviceCtx();
        DeviceCtx(DeviceCtx&&) noexcept;
    };
    // The DRISC L1 layout every relay shares: staging slots, per-core scratch, the done (with heartbeat) and stop
    // words, and the socket config at the top.
    struct RelayL1 {
        uint32_t stage_base = 0, n_stage = 0, core_records = 0, done = 0, stop = 0, cfg = 0;
    };

    // The DRISC L1 layout every relay shares; false when the region cannot hold a relay.
    bool carve_relay_l1(const Hal& hal);
    // The idle-eth pusher's carve and the ACTIVE_ETH region the link ends own; a region too small leaves the eth
    // side off and the relays unaffected.
    void carve_eth_l1(const Hal& hal, uint32_t& aeth_unreserved, uint32_t& aeth_unres_size);
    bool boot_device(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        DeviceCtx& ctx,
        const distributed::MeshCoordinate& coord);
    void enumerate_worker_grid(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx);
    // Idle-eth cores as padded standard cores in the decode roster (never the relay roster).
    void enumerate_eth_cores(DeviceCtx& ctx);
    // Registers a core with the decoder (its XY and five lanes) and returns its producer record.
    Producer& enroll(DeviceCtx& ctx, const CoreCoords& core, uint64_t prof_l1, bool blocking);
    void zero_control(const DeviceCtx& ctx, const Producer& p);
    // Relay count, each relay's DRAM view and core, and a check that firmware left that core's NIUs in stream
    // mode. False: no relay can run on this device.
    bool choose_relay_cores(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx);
    void reserve_spool();
    std::unordered_map<std::string, uint32_t> relay_compile_args(uint32_t chip, uint32_t d) const;
    // False means capture must be abandoned for this device.
    bool launch_relay(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        DeviceCtx& ctx,
        const distributed::MeshCoordinate& coord,
        uint32_t d);
    // False: this pusher is dropped; the capture continues without it.
    bool launch_eth_pusher(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        DeviceCtx& ctx,
        const distributed::MeshCoordinate& coord);
    // Builds the drainer's socket, zeroes its control block, launches `program` resident on its core and waits for
    // its heartbeat. False: nothing drains from that core; the caller decides what that costs the capture.
    bool launch_drainer(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        DeviceCtx& ctx,
        const distributed::MeshCoordinate& coord,
        Drainer& d,
        const DrainerL1& l1,
        std::unique_ptr<Program> program,
        std::string_view what);
    // quiesce() for one device: its drainers that came up, then its producers disarmed, then the relays released.
    void stop_device(uint32_t device_index, const DeviceCtx& ctx, const RelayStateFn& on_state);
    // Stops one drainer through its stop word and reports Drained, then Done, through `on_state`; one that does not
    // finish within 10 s is a fault.
    void stop_drainer(
        uint32_t device_index,
        const DeviceCtx& ctx,
        const Drainer& r,
        std::string_view what,
        const RelayStateFn& on_state);
    // PROFILER_ARMED on every blocking producer: set once the drainers are up (producers boot unarmed and never
    // block on a full ring until then), cleared once every drainer is done so a producer blocked on a full ring is
    // released.
    void set_producers_armed(const DeviceCtx& ctx, bool armed);
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
