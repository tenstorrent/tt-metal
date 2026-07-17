// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_trace_id.hpp>

#include "impl/buffers/tensor_prefetcher_request.hpp"

namespace tt::tt_metal {

class IDevice;
class MeshTensor;
class Program;

namespace distributed {

class MeshDevice;

// Long-lived Tensor prefetcher (DRISC) for a single MeshDevice. Holds the
// per-device Programs, the per-(device, sender) H2D sockets, and the host worker
// thread that drains the request queue. It does NOT own or hold the queued tensors
// or GCBs alive — queue() serializes each request into socket pages and keeps only
// those bytes, so the caller must keep the tensors and GCB alive until stop() (see
// the public tensor_prefetcher.hpp note).
//
// Single-prefetcher-at-a-time invariant: start() asserts is_active() is false.
//
// Lifecycle:
//   * start(config) builds one Program per IDevice in the mesh, with a DRISC
//     kernel on every DRAM sender core. Allocates one H2D socket per (device,
//     sender). Launches the programs (non-blocking — kernels park on
//     socket_wait_for_pages) and spawns the host worker thread.
//   * queue(gcb, subset, tensors) serializes the request into one or more
//     fixed-size socket pages (splitting when the tensor list overflows a page)
//     and pushes them onto the internal queue in order. The host worker thread
//     fans each page out to every socket whose mesh coord is in `subset` via
//     non-blocking try_write so one slow socket can't starve the others. The
//     caller is responsible for keeping tensors and the GCB alive until stop()
//     (see the public tensor_prefetcher.hpp note).
//   * stop() pushes a zero-tensor request targeting the full mesh, joins the
//     worker thread (the kernel exits on `num_entries == 0`), WaitProgramDone
//     on each device, releases per-cycle resources.
//   * Destructor calls stop().
class TensorPrefetcherManager {
public:
    // `lock_api_function` grabs the owning MeshDevice's api_mutex_ (bound from
    // MeshDeviceImpl::lock_api). start()/queue()/stop() take it for the duration
    // of the call so prefetcher operations serialize against the rest of the
    // device API, mirroring MeshCommandQueueBase. enqueue_cq_signal_and_wait() also
    // takes it, but only around its manager-state snapshot — it must release the lock
    // before the dispatcher write, which re-locks the same (non-recursive) api_mutex_.
    TensorPrefetcherManager(MeshDevice* mesh_device, std::function<std::lock_guard<std::mutex>()> lock_api_function);
    ~TensorPrefetcherManager();

    TensorPrefetcherManager(const TensorPrefetcherManager&) = delete;
    TensorPrefetcherManager& operator=(const TensorPrefetcherManager&) = delete;
    TensorPrefetcherManager(TensorPrefetcherManager&&) = delete;
    TensorPrefetcherManager& operator=(TensorPrefetcherManager&&) = delete;

    void start(const experimental::TensorPrefetcherConfig& config);

    // When `cq_id`'s command queue is mid trace-capture, the serialized request pages are
    // captured into trace_requests_ keyed by that trace's MeshTraceId instead of being sent
    // immediately; they are (re)sent on every replay_trace() of that trace. Otherwise the
    // pages are queued for immediate fan-out. `cq_id` == std::nullopt resolves to the
    // current/default command queue (see MeshDevice::mesh_command_queue).
    void queue(
        const experimental::GlobalCircularBuffer& gcb,
        const std::optional<MeshCoordinateRangeSet>& device_subset,
        const std::vector<experimental::TensorPrefetcherInput>& tensors,
        std::optional<uint8_t> cq_id = std::nullopt);

    // Re-queue every request captured under `trace_id` for immediate fan-out. No-op if no
    // prefetcher requests were captured during that trace's capture. Called from the trace
    // replay path so a captured request is re-sent on each trace execution.
    void replay_trace(const MeshTraceId& trace_id);

    // Drop the requests captured under `trace_id`. Called when the trace is released.
    void release_trace(const MeshTraceId& trace_id);

    // Make the prefetcher wait until all work currently enqueued on command queue
    // `cq_id` has landed before it reads DRAM. Bumps a host-side per-CQ counter,
    // has the dispatcher write the new value into every DRAM core's signal slot
    // (ordered after prior CQ work), and queues a WAIT_CQ request so each kernel
    // blocks until that value is observed. Must be called synchronously on the
    // host thread that enqueues the data writes (after them, before the dependent
    // prefetch request).
    void enqueue_cq_signal_and_wait(uint8_t cq_id, const std::optional<MeshCoordinateRangeSet>& device_subset);

    void stop();

    bool is_active() const { return active_; }

private:
    // ---- Constants shared with the kernel side ----
    // The request page wire format (header + entry table + deduplicated layout table)
    // and the fixed payload size kRequestPageBytes live in
    // impl/buffers/tensor_prefetcher_request.hpp so the host and the kernel agree on
    // both the byte layout and the payload size. A Queue call whose tensors overflow one
    // page is split across multiple pages.

    // FIFO depth — how many in-flight requests a single socket can hold before
    // back-pressuring. kSocketFifoPages × align_up(kRequestPageBytes, pcie) per socket.
    // Scaled inversely with kRequestPageBytes (128 × 128 B == the previous 16 × 1024 B) so
    // shrinking the page to one-matmul granularity keeps the per-socket DRISC L1 FIFO at the
    // same footprint while allowing 8× more small in-flight requests.
    static constexpr uint32_t kSocketFifoPages = 128;

    struct Request {
        // One logical socket page, materialized per sender. Either size num_senders_ (a
        // PREFETCH page — each sender carries only its own slice of the per-receiver streaming
        // rotation table, so the bytes differ per sender) or size 1 (STOP / WAIT_CQ — no
        // rotation, identical bytes broadcast to every sender). worker_loop indexes
        // sender_pages[sender_pages.size() == 1 ? 0 : s].
        std::vector<std::vector<uint8_t>> sender_pages;
        std::vector<MeshCoordinate> target_devices;
    };

    void worker_loop();
    void enumerate_dram_senders();
    void build_and_launch_programs(uint32_t stage_ring_base, uint32_t stage_ring_size);
    void allocate_sockets();
    // Serialize a Queue call's tensors into one or more socket pages, deduplicating
    // tensor layouts within each page and splitting when a page fills. Returns one entry per
    // logical page; each entry is a per-sender vector (size num_senders_) of materialized page
    // bytes — the header/entry/geometry bytes are identical across senders, while each sender's
    // page carries only that sender's slice of the per-receiver streaming rotation table.
    std::vector<std::vector<std::vector<uint8_t>>> serialize_request_pages(
        const experimental::GlobalCircularBuffer& gcb,
        const std::vector<experimental::TensorPrefetcherInput>& data_tensors) const;
    MeshCoordinateRangeSet full_mesh_subset() const;

    MeshDevice* mesh_device_;
    // Grabs the owning MeshDevice's api_mutex_ for the duration of an API call.
    std::function<std::lock_guard<std::mutex>()> lock_api_function_;
    bool active_ = false;
    uint32_t stage_ring_base_ = 0;
    uint32_t stage_ring_size_ = 0;
    uint32_t ring_half_ = 0;
    uint32_t stage_third_ = 0;
    // Per-DRAM-core L1 layout (uniform across all sender cores on all devices).
    // socket_config / socket_data are local L1 addresses; host writes add the
    // DRAM_L1_NOC_OFFSET (passed into H2DSocket's DRAM-recv ctor) before going
    // over NOC.
    uint32_t socket_config_l1_addr_ = 0;
    uint32_t socket_data_l1_addr_ = 0;
    // Base (local DRISC L1) of this prefetcher's per-CQ signal slots; uniform
    // across all sender cores. Carved at the front of the kernel working region.
    uint32_t cq_signal_l1_addr_ = 0;
    // Host-side monotonic signal counter per command queue. enqueue_cq_signal_and_wait
    // pre-increments cq_signal_counter_[cq_id] and uses it for both the dispatcher
    // write and the WAIT_CQ request value.
    std::array<uint32_t, kNumCqSignalSlots> cq_signal_counter_{};

    // sender_logical_cores_[s] = logical DRAM core for sender s. Picked at start
    // via pick_unused_dram_logical_core / dram_sender_logical_cores; GCBs queued
    // must use the same picks (deterministic on bank_id), which they do because the
    // GCB factory calls the same pickers with the same dual_senders_per_bank flag.
    std::vector<CoreCoord> sender_logical_cores_;
    uint32_t num_senders_ = 0;
    uint32_t num_banks_ = 0;
    // When true, each DRAM bank is driven by two sender cores (see TensorPrefetcherConfig).
    bool dual_senders_per_bank_ = false;

    // One program per IDevice in the mesh; programs_[d].
    std::vector<std::unique_ptr<Program>> programs_;
    std::vector<IDevice*> devices_;

    // sockets_[d * num_senders_ + s] = socket for (device d, sender s).
    std::vector<std::unique_ptr<H2DSocket>> sockets_;

    // MeshCoordinate -> index into devices_, populated once at start() so
    // worker_loop fan-out is O(targets) instead of O(targets * devices).
    std::unordered_map<MeshCoordinate, uint32_t> device_index_by_coord_;

    // Host worker thread + queue
    std::thread host_worker_;
    std::mutex queue_mu_;
    std::condition_variable queue_cv_;
    std::deque<Request> pending_;
    std::atomic<bool> stop_requested_{false};

    // Requests captured during trace capture, keyed by the recording trace's id. Populated by
    // queue() when its command queue is mid-capture; drained back onto pending_ by
    // replay_trace() on each trace execution; erased by release_trace(). Guarded by queue_mu_.
    std::unordered_map<MeshTraceId, std::vector<Request>> trace_requests_;
};

}  // namespace distributed
}  // namespace tt::tt_metal
