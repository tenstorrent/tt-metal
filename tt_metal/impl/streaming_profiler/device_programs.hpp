// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "impl/context/context_types.hpp"
#include "impl/streaming_profiler/capture_context.hpp"

namespace tt {
class Cluster;
}
struct metal_SocDescriptor;

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
class D2HSocket;
}  // namespace distributed
class Program;
class IDevice;
class Hal;
class MetalContext;

namespace streaming_profiler {

// Whether this process captures: the profiler is enabled on Blackhole with DRAM programmable cores.
bool can_capture(const MetalContext& mc);

std::vector<CoreCoord> sorted_yx(const std::unordered_set<CoreCoord>& cores);

struct CoreCoords {
    CoreCoord logical, virt, phys;
};
CoreCoords locate(tt::Cluster& cluster, uint32_t chip, const CoreCoord& logical, CoreType type);

uint64_t host_l1_addr(const Hal& hal, HalProgrammableCoreType core, HalL1MemAddrType type);
void zero_l1(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint64_t addr, uint32_t bytes);
void zero_profiler_control(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint64_t addr);
// Firmware holds a DRAM view endpoint's NIU in NOC2AXI: a read the core issues on it never goes out, and one
// arriving on it goes to GDDR.
uint8_t dram_view_endpoint_noc_mask(const metal_SocDescriptor& soc, const CoreCoord& logical);

// Compiles before it launches: launching onto an idle eth core that holds no valid binary wedges the core and can take
// the host down.
void launch_resident(IDevice* device, Program& program);

struct CapturedSocket {
    distributed::D2HSocket* socket = nullptr;  // owned by the DevicePrograms that booted it
    uint32_t dev = 0;
    uint32_t index = 0;
    int numa_node = -1;
    bool sync = false;
};

enum class RelayState { Running, AwaitingAcks, Done };
using RelayStateFn = std::function<void(uint32_t device_index, uint32_t socket_index, RelayState)>;

// Destruction releases the spool, so it must precede the mesh allocator's.
class DevicePrograms {
public:
    DevicePrograms();
    ~DevicePrograms();
    DevicePrograms(const DevicePrograms&) = delete;
    DevicePrograms& operator=(const DevicePrograms&) = delete;

    // Returns false, after logging why, where the profiler can't capture.
    [[nodiscard]] bool boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    const CaptureContext& capture_context() const { return capture_; }
    const std::vector<CapturedSocket>& sockets() const { return sockets_; }
    // Call it only once the receiver's ingest threads drain the sockets, or the eth relay fills its FIFOs before any
    // consumer attaches.
    void start();
    // Runs once: later calls do nothing, and the destructor calls it if it hasn't run.
    void quiesce(const RelayStateFn& on_state);
    void verify_completeness() const;

private:
    // A core whose kernel runs for the whole capture, launched outside the command queue, where it would deadlock the
    // first Finish().
    struct ResidentCore {
        std::unique_ptr<Program> program;
        CoreCoords core;
        std::string name;
        uint64_t ctrl = 0;
        uint32_t socket_index = 0;
        uint32_t socket_count = 0;
    };
    struct Producer : CoreCoords {
        CoreType type = CoreType::WORKER;
        uint64_t prof_l1 = 0;
    };
    struct SocketSpec {
        uint32_t cfg = 0;
        uint32_t fifo_bytes = 0;
        bool sync = false;
    };
    struct DeviceCtx {
        distributed::MeshDevice* mesh;
        distributed::MeshCoordinate coord;
        IDevice* device;
        uint32_t chip_id = 0;
        uint32_t worker_count = 0;
        int numa_node = -1;
        // The relays' sockets in relay order, then the eth relay's sync socket and, with eth zones, its frames socket.
        std::vector<std::unique_ptr<distributed::D2HSocket>> sockets;
        // Filled while the device boots on its own thread, then appended to DevicePrograms::sockets_ in device order.
        std::vector<CapturedSocket> captured;
        CaptureContext::Device capture;
        std::optional<ResidentCore> ruler;
        // The worker grid row-major, which the relays split into bands, then the tracker, then its linked cores.
        std::vector<Producer> producers;
        std::vector<ResidentCore> relays;
        ResidentCore tracker;
        ResidentCore eth_relay;

        std::span<const Producer> workers() const { return std::span(producers).first(worker_count); }
        const Producer& tracker_producer() const { return producers[worker_count]; }
        std::span<const Producer> drained_by_eth_relay() const {
            return std::span(producers).subspan(worker_count + 1);
        }
        // The ruler rides in the eth relay's program.
        bool ruler_running() const { return ruler && eth_relay.program; }

        DeviceCtx(distributed::MeshDevice& mesh, const distributed::MeshCoordinate& coord);
        ~DeviceCtx();
        DeviceCtx(DeviceCtx&&) noexcept;
    };
    struct LinkEnd {
        IDevice* device = nullptr;
        uint32_t chip = 0;
        CoreCoord eth, virt;
        uint64_t prof_l1 = 0;
        bool sender = false;
    };
    // Idle eth L1, which the tracker, the eth relay and the ruler share, plus link and link_ring on the link-end cores.
    struct EthL1 {
        uint32_t frames_cfg = 0, sync_cfg = 0, ctrl = 0, stage = 0, scratch = 0, sync_ring = 0, link = 0, link_ring = 0;
    };
    // The GDDR spool is the HAL's PROFILER DRAM region, which MetalEnv sizes for it when the streaming profiler is on,
    // so it lies below every allocator's unreserved base.
    struct DriscL1 {
        uint32_t stage_base = 0, core_records = 0, ctrl_local = 0, cfg = 0, spool_addr = 0, spool_bytes = 0;
        uint64_t ctrl = 0;
    };

    tt::Cluster& get_cluster() const;
    void carve_l1(const Hal& hal);
    void enumerate_worker_grid(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx);
    void enumerate_eth_cores(DeviceCtx& ctx);
    Producer& add_producer(DeviceCtx& ctx, const CoreCoords& core, CoreType type, uint64_t prof_l1);
    void choose_relay_cores(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx);
    void start_resident(
        DeviceCtx& ctx,
        ResidentCore& resident,
        HalProgrammableCoreType core_type,
        const std::vector<SocketSpec>& specs,
        std::unique_ptr<Program> program);
    std::unique_ptr<Program> relay_program(const DeviceCtx& ctx, uint32_t relay_index);
    std::unique_ptr<Program> tracker_program(const DeviceCtx& ctx);
    std::unique_ptr<Program> eth_relay_program(const DeviceCtx& ctx);
    void plan_links();
    std::array<LinkEnd, 2> link_ends(const CaptureContext::Link& link) const;
    void launch_links();
    // Stops each sender before its receiver so the sender's last round still echoes off a live one.
    void stop_links();
    enum class StopStage { Relays, Ruler, Tracker, EthRelay };
    static std::vector<const ResidentCore*> running_cores(const DeviceCtx& ctx, StopStage stage);
    void request_stop(const DeviceCtx& ctx, const ResidentCore& resident);
    void await_stop(
        uint32_t device_index, const DeviceCtx& ctx, const ResidentCore& resident, const RelayStateFn& on_state);
    // Clearing PROFILER_ARMED once every relay is done releases a producer blocked on a full ring.
    void write_producers_armed(const DeviceCtx& ctx, uint32_t armed);

    uint64_t prof_l1_ = 0;
    ContextId context_id_{0};
    DriscL1 drisc_l1_;
    EthL1 eth_l1_;
    std::vector<DeviceCtx> devices_;
    CaptureContext capture_;
    std::vector<CapturedSocket> sockets_;
    // Empty when the fabric routers run the ends.
    std::vector<std::unique_ptr<Program>> resident_link_programs_;
    bool fabric_link_sync_ = false;
    bool links_running_ = false;
    bool quiesced_ = false;
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
