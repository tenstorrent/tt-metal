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
namespace tt::llrt {
class RunTimeOptions;
}

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

// Returns whether this process profiles its mesh devices, which requires the profiler to be enabled on Blackhole with
// DRAM programmable cores.
bool can_capture(const Hal& hal, const llrt::RunTimeOptions& rtoptions);
bool can_capture(const MetalContext& mc);

std::vector<CoreCoord> sorted_yx(const std::unordered_set<CoreCoord>& cores);

struct CoreCoords {
    CoreCoord logical, virt, phys;
};
CoreCoords locate_core(tt::Cluster& cluster, uint32_t chip, const CoreCoord& logical, CoreType type);

void zero_l1(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint64_t addr, uint32_t bytes);
void zero_profiler_control(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint64_t addr);

// Compiles the program, then launches it without waiting for it. It compiles first because launching onto an idle eth
// core that holds no valid binary wedges the core and can take the host down.
void launch_resident(IDevice* device, Program& program);

struct CapturedSocket {
    distributed::D2HSocket* socket = nullptr;  // owned by the DevicePrograms that booted it
    uint32_t device_index = 0;
    uint32_t socket_index = 0;
    int numa_node = -1;
    bool sync_socket = false;
};

enum class RelayState { Running, AwaitingAcks, Done };
using RelayStateFn = std::function<void(uint32_t device_index, uint32_t socket_index, RelayState)>;

// Runs a capture's resident kernels on each chip. Relays on DRAM cores ship the worker cores' profiler rings to the
// host. On the idle eth cores, the wall-clock core measures the chip's wall clock against its refclk, and the eth relay
// ships the eth cores' profiler rings and the clock sync's records. With the sync check on, a third idle eth core, the
// check core, samples both clocks to measure the sync's accuracy. The active eth cores on the two sides of a synced
// link run its link sync ports. Destroying a DevicePrograms stops these kernels and closes their sockets, so it must be
// destroyed while its mesh is open.
class DevicePrograms {
public:
    DevicePrograms();
    ~DevicePrograms();
    DevicePrograms(const DevicePrograms&) = delete;
    DevicePrograms& operator=(const DevicePrograms&) = delete;

    // Launches the relays, wall-clock core, eth relay and check core on every local chip and opens their sockets.
    // Returns false, after logging why, where the profiler can't run.
    [[nodiscard]] bool boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    const CaptureContext& capture_context() const { return capture_; }
    const std::vector<CapturedSocket>& sockets() const { return sockets_; }
    // Sets the go word of every chip's wall-clock core, eth relay and check core, and starts the link sync. Call it
    // only once the receiver's ingest threads are draining the sockets, or the eth relay fills its FIFOs before any
    // consumer attaches.
    void start();
    // Stops the link sync ports and every resident core, reports each socket's relay state to `on_state`, and disarms
    // the producers. Only the first call does anything, and the destructor makes that call if nothing else has.
    void quiesce(const RelayStateFn& on_state);
    void verify_completeness(uint32_t device_index);

private:
    struct L1Range {
        uint64_t addr = 0;
        uint32_t bytes = 0;
    };
    // A core whose kernel runs for the whole capture. It is launched outside the command queue, where it would deadlock
    // the first Finish().
    struct ResidentCore {
        std::unique_ptr<Program> program;
        CoreCoords core;
        std::string name;
        uint64_t ctrl = 0;
        // L1 besides the control block that start_resident zeroes before it starts the core.
        std::optional<L1Range> stale_l1;
        uint32_t socket_index = 0;
        uint32_t socket_count = 0;
    };
    struct Producer : CoreCoords {
        CoreType type = CoreType::WORKER;
        uint64_t control_vector_l1 = 0;
    };
    struct SocketSpec {
        uint32_t cfg = 0;
        uint32_t fifo_bytes = 0;
        bool sync_socket = false;
    };
    struct DeviceCtx {
        distributed::MeshDevice* mesh;
        distributed::MeshCoordinate coord;
        IDevice* device;
        uint32_t index = 0;
        uint32_t chip_id = 0;
        uint32_t worker_count = 0;
        int numa_node = -1;
        // The relays' sockets in relay order, then the eth relay's sync socket and its frames socket.
        std::vector<std::unique_ptr<distributed::D2HSocket>> sockets;
        // Filled while the device boots on its own thread, then appended to DevicePrograms::sockets_ in device order.
        std::vector<CapturedSocket> captured;
        std::optional<ResidentCore> check;
        // The worker grid in row-major order, which the relays split into bands, then the wall-clock core, then the eth
        // cores the eth relay drains.
        std::vector<Producer> producers;
        std::vector<ResidentCore> relays;
        ResidentCore wall_clock;
        ResidentCore eth_relay;

        const Producer& wall_clock_producer() const { return producers[worker_count]; }
        std::span<const Producer> drained_by_eth_relay() const {
            return std::span(producers).subspan(worker_count + 1);
        }

        DeviceCtx(distributed::MeshDevice& mesh, const distributed::MeshCoordinate& coord);
        ~DeviceCtx();
        DeviceCtx(DeviceCtx&&) noexcept;
    };
    struct LinkPort {
        const DeviceCtx& ctx;
        const Producer& producer;
    };
    // The idle eth L1 layout shared by the wall-clock core, the eth relay and the check core, plus link and link_ring
    // on the cores that run link sync ports.
    struct EthL1 {
        uint32_t frames_cfg = 0, sync_cfg = 0, ctrl = 0, stage = 0, sample_ring = 0, sync_ring = 0, link = 0,
                 link_ring = 0;
    };
    // The L1 addresses of each relay's buffers on its DRAM core, and the address and size of its GDDR spool, which are
    // 0 when the spool is off. The spool is the HAL's PROFILER DRAM region, which MetalEnv sizes for it when the
    // streaming profiler is on, so it lies below every allocator's unreserved base.
    struct DriscL1 {
        uint32_t stage_base = 0, core_records = 0, ctrl = 0, cfg = 0, spool_addr = 0, spool_bytes = 0;
        uint64_t host_ctrl = 0;
    };

    void carve_l1();
    void enumerate_worker_grid(DeviceCtx& ctx);
    void enumerate_eth_cores(DeviceCtx& ctx);
    Producer& add_producer(DeviceCtx& ctx, const CoreCoords& core, CoreType type, uint64_t control_vector_l1);
    void choose_relay_cores(DeviceCtx& ctx);
    void start_resident(
        DeviceCtx& ctx,
        ResidentCore& resident,
        HalProgrammableCoreType core_type,
        const std::vector<SocketSpec>& specs,
        std::unique_ptr<Program> program);
    std::unique_ptr<Program> relay_program(const DeviceCtx& ctx, uint32_t relay_index);
    std::unique_ptr<Program> wall_clock_program(const DeviceCtx& ctx);
    std::unique_ptr<Program> check_program(const DeviceCtx& ctx);
    std::unique_ptr<Program> eth_relay_program(const DeviceCtx& ctx);
    void plan_links();
    // Returns the link's transmitter port, then its receiver port.
    std::array<LinkPort, 2> link_ports(const CaptureContext::Link& link) const;
    void launch_links();
    // Stops each transmitter before its receiver, so the transmitter's last round is still echoed by a running
    // receiver.
    void stop_links();
    void await_stop(const DeviceCtx& ctx, const ResidentCore& resident, const RelayStateFn& on_state);
    // Writes `armed` to PROFILER_ARMED on every producer of the chip. Clearing it once every relay is done releases a
    // producer blocked on a full ring.
    void write_producers_armed(const DeviceCtx& ctx, uint32_t armed);

    uint64_t control_vector_l1_ = 0;
    MetalContext* mc_ = nullptr;
    DriscL1 drisc_l1_;
    EthL1 eth_l1_;
    std::vector<DeviceCtx> devices_;
    CaptureContext capture_;
    std::vector<CapturedSocket> sockets_;
    // Empty when the fabric routers run the ports.
    std::vector<std::unique_ptr<Program>> resident_link_programs_;
    bool fabric_link_sync_ = false;
    bool links_running_ = false;
    bool quiesced_ = false;
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
