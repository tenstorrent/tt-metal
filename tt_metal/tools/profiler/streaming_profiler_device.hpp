// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streaming profiler device layer: relay bring-up, host<->device clock sync, quiesce, and the teardown
// completeness check for one MeshDevice's local Blackhole devices.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hostdev/streaming_profiler_common.h"

#include <tt-metalium/core_coord.hpp>
#include "tools/profiler/streaming_profiler_receiver.hpp"

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

// The step a failed bring-up was in, for the message that disables the profiler.
const std::string& bringup_step();

// The relays of one capture. boot() brings them up on every eligible local Blackhole device, syncs each clock
// and returns the receiver's view of every device that came up; quiesce() stops them at teardown through the
// relay's stop word (1 = quiesce, 2 = release the NIU) and leaves the resident idle FW alone; verify() checks
// capture completeness against the receiver's consumed mirrors. Destruction releases the spool, so it must
// precede the mesh allocator's.
class Devices {
public:
    Devices() = default;
    ~Devices();
    Devices(const Devices&) = delete;
    Devices& operator=(const Devices&) = delete;

    std::vector<ReceiverDeviceConfig> boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    // Waits for the producer rings to drain (unblocking back-pressure if they do not), stops every relay and
    // reports each finished stream to the receiver when there is one.
    void quiesce(Receiver* receiver);
    void verify(Receiver& receiver);

private:
    // One DRISC relay per DRAM view, up to kMaxRelays, each sweeping a slice of the worker grid into its own socket.
    static constexpr uint32_t kMaxRelays = 8;
    static constexpr uint32_t kPageSize = 64;
    static constexpr uint32_t kNRisc = 5;

    struct DeviceCtx {
        uint32_t chip_id = 0;
        std::unique_ptr<distributed::D2HSocket> sockets[kMaxRelays];  // moved into the receiver by boot()
        uint32_t nl = 0;                                              // lanes = num_cores * NRISC
        // Launched outside the command queue: a DRAM-only program touches no fast-dispatch resource, so it stays
        // resident across every workload, while going through the CQ would deadlock the first Finish().
        std::unique_ptr<Program> relay_program[kMaxRelays];
        IDevice* device = nullptr;
        CoreCoord drisc_logical[kMaxRelays];
        CoreCoord drisc_virtual[kMaxRelays];
        uint64_t drisc_l1_noc[kMaxRelays] = {};  // NoC-addressable base of each DRISC L1 window
        uint32_t drisc_l1_base[kMaxRelays] = {};
        uint32_t stop_addr[kMaxRelays] = {};  // host writes 1 to quiesce, 2 to release the NIU
        uint32_t done_addr[kMaxRelays] = {};  // relay publishes 0xD09E**** once its last page is out
        uint32_t n_drisc = 0;                 // relays (= sockets) in use, [1, kMaxRelays]; set once by boot_device
        // Per core index: logical (x,y) [the public Core::coord] and virtual [what the SRC lane resolves to].
        std::vector<std::pair<uint32_t, uint32_t>> core_logical, core_virt;
        std::unordered_map<uint32_t, uint32_t> core_of_xy;  // packed (y<<16)|x -> dense core index
        bool active = false;
        bool clock_synced = false;
        double freq_ghz = 0.0;  // measured sync frequency (cycles/ns); aiclk fallback
        uint64_t anchor_ticks = 0;
        int64_t anchor_host_ns = 0;  // std::chrono::steady_clock at anchor_ticks; the capture start when unsynced

        DeviceCtx();
        ~DeviceCtx();
        DeviceCtx(DeviceCtx&&) noexcept;
    };

    // Stream mode (1) or NOC2AXI (0) for these DRISCs' NIUs, in one launch (see the .cpp).
    static void set_drisc_niu_mode(IDevice* device, const std::vector<CoreCoord>& drisc_logicals, uint32_t stream);
    // Set PROFILER_ARMED on the cores the relays drain, once they are up; producers boot unarmed and never block
    // on a full ring until then.
    void arm_producers(DeviceCtx& ctx);
    void report_unarmed(uint32_t device_id);
    bool wait_producer_rings_drained(DeviceCtx& ctx, std::chrono::milliseconds budget);
    void disarm_producer_backpressure(DeviceCtx& ctx);
    // State the boot_device() steps below hand each other.
    struct BootPlan {
        uint64_t prof_l1 = 0;  // Tensix profiler L1 base (control vector, then the per-RISC rings)
        uint64_t num_cores = 0;
        uint32_t slot_bytes = 0;   // staging slot size; mirrors the relay kernel's kSlotWords
        uint32_t spool_bytes = 0;  // 0 = direct push
        uint32_t spool_addr = 0;
        std::vector<uint32_t> coords;        // core index -> packed (y<<16)|x, the relay's poll list
        std::vector<uint8_t> zero_ctrl;      // a zeroed profiler control vector, reused for every write
        std::vector<uint32_t> banks;         // DRAM view per relay
        std::vector<CoreCoord> relay_cores;  // logical DRAM core per relay
    };

    bool boot_device(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        DeviceCtx& ctx,
        const distributed::MeshCoordinate& coord);
    void enumerate_worker_grid(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx, BootPlan& plan);
    bool choose_relay_banks(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx, BootPlan& plan);
    void reserve_spool(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx, BootPlan& plan);
    // Configures the relay's TLB window, builds its socket, launches it and confirms its heartbeat. False
    // means capture must be abandoned for this device; the caller disarms the producers.
    bool launch_relay(
        const std::shared_ptr<distributed::MeshDevice>& mesh_device,
        DeviceCtx& ctx,
        const distributed::MeshCoordinate& coord,
        const BootPlan& plan,
        uint32_t d);
    // After the relays swept to empty and the receiver drained: compare every worker lane's own tail
    // against the receiver's consumed-words mirror.
    void verify_completeness(DeviceCtx& ctx, uint32_t device_index, Receiver& receiver);

    std::vector<DeviceCtx> devices_;
    // GDDR spool: one replicated mesh buffer with one interleaved page per DRAM bank, so the same window is
    // reserved in every bank of every device. nullptr in direct-push runs.
    std::shared_ptr<distributed::MeshBuffer> spool_buffer_;
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
