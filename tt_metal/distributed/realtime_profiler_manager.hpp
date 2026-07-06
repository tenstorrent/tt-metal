// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "context/context_types.hpp"
#include "tt_metal/common/broadcast_ring.hpp"
#include "tt_metal/impl/dispatch/data_collection.hpp"

namespace tt::tt_metal {

class IDevice;
class Program;
class DataCollector;
class RealtimeProfilerTracyHandler;

namespace distributed {

class D2HSocket;
class MeshDevice;

// L1 carve-out addresses for the reserved RT-profiler tensix core. The ring buffer
// (BRISC->NCRISC handoff) and the D2H socket sender config sit in a single carve-out
// anchored past dispatch_mem_map's UNRESERVED, bypassing the user-space allocator.
struct RealtimeProfilerCoreL1Addrs {
    uint32_t base = 0;
    uint32_t ring_buffer = 0;
    uint32_t socket_config = 0;
};

// Owns the full RT-profiler subsystem for a single MeshDevice: per-device state, the
// background receiver thread, the host-side Tracy handler, and the host-device sync
// handshake.
//
// Lifecycle:
//   * Constructor runs the eligibility gate, brings up D2H sockets and BRISC/NCRISC
//     kernels on each eligible device, and starts the receiver thread.
//   * shutdown() (or destruction) signals receiver termination, joins the thread, and
//     drops the Tracy handler. Idempotent.
//   * trigger_sync_check() asks the receiver to run a sync handshake only on devices
//     whose last finish/init sync was at least 60s ago (each device tracked separately),
//     or on the first finish-path attempt after init (so short runs still get FINISH_SYNC),
//     Called from the FD command queue's finish path.
//     Constructor init uses the same interval process-wide per chip_id to throttle full
//     run_sync + SYNC_CHECK when reopening meshes on the same chips.
//
//     Init host-device sync (run_sync + constructor SYNC_CHECK) shards work across
//     devices using a small worker pool (up to hardware_concurrency).
class RealtimeProfilerManager : private tt::RealtimeProfilerCallbackListener {
public:
    explicit RealtimeProfilerManager(const std::shared_ptr<MeshDevice>& mesh_device);
    ~RealtimeProfilerManager() override;

    RealtimeProfilerManager(const RealtimeProfilerManager&) = delete;
    RealtimeProfilerManager& operator=(const RealtimeProfilerManager&) = delete;
    RealtimeProfilerManager(RealtimeProfilerManager&&) = delete;
    RealtimeProfilerManager& operator=(RealtimeProfilerManager&&) = delete;

    // Idempotent: writes terminate flag, joins receiver thread, releases Tracy handler,
    // and notifies deactivation. Safe to call multiple times.
    void shutdown();

    // Requests the receiver to perform the handshake if at least one device needs a sync
    // only on those devices (others are unchanged). No-op when no devices are active,
    // the Tracy handler has been released, or every device was synced within the last
    // 60 seconds (except the first finish-path sync after init, which always runs).
    void trigger_sync_check();

    // RT-profiler diagnostics
    uint32_t peak_fifo_pages() const { return peak_fifo_pages_.load(std::memory_order_relaxed); }
    uint32_t host_fifo_capacity_pages() const;
    uint64_t num_published_records() const { return num_published_records_.load(std::memory_order_relaxed); }
    uint64_t num_published_batches() const { return num_published_batches_.load(std::memory_order_relaxed); }
    uint32_t ring_full_wait_count() const;  // reads device L1
    size_t num_active_devices() const { return devices_.size(); }

private:
    struct DeviceState {
        IDevice* device = nullptr;
        uint32_t chip_id = 0;
        MeshCoordinate mesh_coord = MeshCoordinate(0);
        CoreCoord realtime_profiler_core;
        std::unique_ptr<D2HSocket> socket;
        // Owns the BRISC+NCRISC realtime-profiler program so its kernels remain alive for
        // the lifetime of the manager. If this goes out of scope while the kernels are
        // still running, downstream tooling (e.g. tt-inspector) loses the kernel metadata.
        std::unique_ptr<Program> realtime_profiler_program;
        RealtimeProfilerCoreL1Addrs core_l1;
        bool fifo_reached_capacity = false;
        uint64_t first_timestamp = 0;
        int64_t sync_host_start = 0;
        double sync_frequency = 0.0;
        uint32_t sync_request_addr = 0;
        uint32_t sync_host_ts_addr = 0;
        int64_t sync_host_time_before = 0;
        // Updated after a successful finish-path or init SYNC_CHECK handshake; used to
        // throttle redundant finish syncs (minimum 60s between attempts per device).
        std::optional<std::chrono::steady_clock::time_point> last_finish_sync_at;
        // After init, the first finish-path sync bypasses the throttle so short runs still
        // get a FINISH_SYNC pair even when finish runs within the minimum interval of init.
        // Cleared after the first successful finish-path handshake for this device.
        bool pending_first_unthrottled_finish_sync = false;
        enum class FinishSyncPhase : uint8_t { Idle, AwaitingDelay, AwaitingResponse };
        FinishSyncPhase finish_sync_phase = FinishSyncPhase::Idle;
        std::chrono::steady_clock::time_point finish_sync_request_at;
        std::chrono::steady_clock::time_point finish_sync_deadline;

        DeviceState();
        ~DeviceState();
        DeviceState(DeviceState&& o) noexcept;
        DeviceState& operator=(DeviceState&&) = delete;
        DeviceState(const DeviceState&) = delete;
        DeviceState& operator=(const DeviceState&) = delete;
    };

    // Set up the D2H socket and launch the BRISC/NCRISC kernels on each eligible local device. Devices failing the
    // eligibility gate or socket creation are skipped.
    void initialize_devices(const std::shared_ptr<MeshDevice>& mesh_device);
    void run_sync(DeviceState& dev_state, uint32_t num_samples);
    void run_init_sync();

    using RecordRing = BroadcastRing<tt::ProgramRealtimeRecord>;

    enum class ConsumerStopMode : uint8_t { Running, StopWithoutDrain, DrainThenStop };

    struct Consumer {
        Consumer(
            RecordRing::Reader reader,
            tt::ProgramRealtimeProfilerCallback callback,
            tt::ProgramRealtimeProfilerCallbackHandle handle) :
            reader(std::move(reader)), callback(std::move(callback)), handle(handle) {}
        RecordRing::Reader reader;
        tt::ProgramRealtimeProfilerCallback callback;
        tt::ProgramRealtimeProfilerCallbackHandle handle;
        std::atomic<ConsumerStopMode> stop_mode{ConsumerStopMode::Running};
        uint64_t dropped = 0;
        std::thread thread;
    };

    void run_consumer(Consumer& consumer);
    void stop_consumer(Consumer& consumer, ConsumerStopMode stop_mode);
    void on_callback_registered(
        tt::ProgramRealtimeProfilerCallbackHandle handle, const tt::ProgramRealtimeProfilerCallback& callback) override;
    void on_callback_unregistered(tt::ProgramRealtimeProfilerCallbackHandle handle) override;

    // Receiver thread entry point: drain every device socket, advance the finish-sync handshake, and
    // publish decoded records to the ring for consumer threads to read.
    void run_receiver();
    uint64_t run_receiver_loop();
    uint64_t drain_receiver_on_shutdown();
    // One drain pass over every device; wakes readers if any pages were read. Returns the number of pages read.
    uint32_t drain_all_devices(
        bool scan_sync_marker, std::vector<uint32_t>& page_buf, std::vector<tt::ProgramRealtimeRecord>& record_buf);
    // Returns the number of pages read from one device.
    uint32_t drain_device_pages(
        DeviceState& dev_state,
        bool scan_sync_marker,
        std::vector<uint32_t>& page_buf,
        std::vector<tt::ProgramRealtimeRecord>& record_buf);
    // Decode program records from drained pages and publish them to the broadcast ring.
    void publish_pages(
        const DeviceState& dev_state,
        const uint32_t* page_buf,
        uint32_t num_pages,
        std::vector<tt::ProgramRealtimeRecord>& records);
    enum SyncRequest : uint32_t { Clear = 0, Set = 1 };
    static void write_sync_request(DeviceState& dev_state, SyncRequest value);
    [[nodiscard]] bool has_active_finish_sync() const;
    void start_finish_syncs(std::chrono::steady_clock::time_point now);
    void advance_finish_sync(DeviceState& dev_state, std::chrono::steady_clock::time_point now);
    void service_finish_sync(std::chrono::steady_clock::time_point now, bool allow_start);
    void notify_finish_sync_waiters();

    // ContextId of the owning MeshDevice, captured in the constructor. All MetalContext
    // accesses inside this manager must go through MetalContext::instance(context_id_)
    // so a non-default (mock / coexistence) context routes through its own HAL, cluster
    // and dispatch_mem_map instead of leaking to the silicon DEFAULT_CONTEXT_ID via the
    // bare instance() inline fallback. See #38445 / #39849 for the broader migration.
    ContextId context_id_;
    const DataCollector* data_collector_ = nullptr;
    std::vector<DeviceState> devices_;
    std::thread receiver_thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> finish_sync_requested_{false};
    std::atomic<bool> finish_sync_busy_{false};
    std::atomic<std::chrono::steady_clock::rep> last_sync_request_at_{0};

    std::mutex finish_sync_wait_mu_;
    std::condition_variable finish_sync_cv_;
    std::unique_ptr<RealtimeProfilerTracyHandler> tracy_handler_;

    // Receiver diagnostics
    std::atomic<uint32_t> peak_fifo_pages_{0};        // peak D2H FIFO usage
    std::atomic<uint64_t> num_published_records_{0};  // count of records published to the ring
    std::atomic<uint64_t> num_published_batches_{0};  // count of batches published to the ring

    static constexpr size_t kMaxConsumerBatchPerDevice = 1u << 15;  // max batch size per device
    static constexpr size_t kMaxConsumerBatchCap = 1u
                                                   << 20;  // hard cap on total batch size (corresponds to 32 devices)
    static constexpr size_t kRingHeadroomBatches = 4;
    static constexpr size_t kMaxRingCapacity = 1u << 22;

    std::optional<RecordRing> ring_;
    std::mutex consumers_mutex_;
    std::unordered_map<tt::ProgramRealtimeProfilerCallbackHandle, std::unique_ptr<Consumer>> consumers_;
};

}  // namespace distributed
}  // namespace tt::tt_metal
