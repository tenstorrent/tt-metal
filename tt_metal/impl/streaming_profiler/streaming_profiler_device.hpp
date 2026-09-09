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

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
class MeshCoordinate;
class D2HSocket;
class MeshBuffer;
}  // namespace distributed
class Program;
class IDevice;

namespace streaming_profiler {

// One device whose relays came up: the sockets the receiver ingests and what its consumers need to decode them.
struct CapturedDevice {
    uint32_t chip_id = 0;
    int numa_node = -1;                                            // the node the sockets bind their FIFOs to
    std::vector<std::unique_ptr<distributed::D2HSocket>> sockets;  // one per relay, in relay order
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
    Devices() = default;
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
    struct DeviceCtx {
        uint32_t chip_id = 0;
        IDevice* device = nullptr;
        CapturedDevice out;
        std::vector<WorkerCore> cores;  // the compute grid, row-major, so a relay's band is a contiguous run
        Relay relays[kMaxRelays];
        uint32_t n_relays = 0;

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
    // Relay count, each relay's DRAM view and core, then every relay's NIU into stream mode. False: no relay can
    // run on this device.
    bool choose_relay_cores(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx);
    void reserve_spool(const std::shared_ptr<distributed::MeshDevice>& mesh_device, IDevice* device);
    // Configures the relay's TLB window, builds its socket, launches it and confirms its heartbeat. False means
    // capture must be abandoned for this device.
    bool launch_relay(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        DeviceCtx& ctx,
        const distributed::MeshCoordinate& coord,
        uint32_t d);
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
    // GDDR spool: one replicated mesh buffer with one interleaved page per DRAM bank, so the same window is
    // reserved in every bank of every device. Bytes 0 = direct push.
    std::shared_ptr<distributed::MeshBuffer> spool_buffer_;
    uint32_t spool_bytes_ = 0;
    uint32_t spool_addr_ = 0;
    std::vector<DeviceCtx> devices_;
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
