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

class MeshCommandQueue;
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

    void start();

    // Capture-vs-send contract, and the `trace_capture_cq` precondition: see
    // QueueTensorPrefetcherRequest. Captured pages live in trace_requests_ keyed by the
    // recording trace's MeshTraceId, and are re-queued by replay_trace().
    void queue(
        const experimental::GlobalCircularBuffer& gcb,
        const std::optional<MeshCoordinateRangeSet>& device_subset,
        const std::vector<experimental::TensorPrefetcherInput>& tensors,
        MeshCommandQueue* trace_capture_cq);

    // Re-queue every request captured under `trace_id` for immediate fan-out. No-op if no
    // prefetcher requests were captured during that trace's capture. Called from the trace
    // replay path so a captured request is re-sent on each trace execution.
    void replay_trace(const MeshTraceId& trace_id);

    // Drop the requests captured under `trace_id`. Called when the trace is released.
    void release_trace(const MeshTraceId& trace_id);

    // Make the prefetcher wait until all work currently enqueued on command queue
    // `cq` has landed before it reads DRAM. Bumps a host-side per-CQ counter,
    // has the dispatcher write the new value into every DRAM core's signal slot
    // (ordered after prior CQ work), and queues a WAIT_CQ request so each kernel
    // blocks until that value is observed. Must be called synchronously on the
    // host thread that enqueues the data writes (after them, before the dependent
    // prefetch request). `cq` must belong to this manager's mesh device, and its id
    // must be within [0, kNumCqSignalSlots).
    void enqueue_cq_signal_and_wait(MeshCommandQueue& cq, const std::optional<MeshCoordinateRangeSet>& device_subset);

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
        // One logical socket page. For PREFETCH this is either one shared page or one page
        // per entry in target_sender_indices (streaming rotations differ by sender).
        // STOP / WAIT_CQ carry one shared page and leave target_sender_indices empty to
        // broadcast to every provisioned sender.
        std::vector<std::vector<uint8_t>> sender_pages;
        std::vector<MeshCoordinate> target_devices;
        std::vector<uint32_t> target_sender_indices;
    };

    void worker_loop();
    void enumerate_dram_senders();
    std::vector<uint32_t> sender_indices_for_gcb(const experimental::GlobalCircularBuffer& gcb) const;
    void build_and_launch_programs(uint32_t stage_ring_base, uint32_t stage_ring_size);
    void allocate_sockets();
    // Serialize a Queue call's tensors into one or more socket pages, deduplicating
    // tensor layouts within each page and splitting when a page fills. Returns one entry per
    // logical page; each entry is either one shared page or a vector in GCB sender-mapping
    // order. The header/entry/geometry bytes are identical across senders, while a streaming
    // page carries only that GCB sender's slice of the per-receiver rotation table.
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
    // Distance between consecutive signal slots. The slots hold one uint32 each but are
    // spaced a full L1 alignment apart: each is the destination of its own dispatcher
    // write, and a dispatch write only lands on an L1-aligned address. Packed 4 bytes
    // apart, every slot but the first would be misaligned and its write would go nowhere,
    // leaving the kernel spinning on a WAIT_CQ that is never satisfied.
    uint32_t cq_signal_slot_stride_ = 0;
    // Host-side monotonic signal counter per command queue. enqueue_cq_signal_and_wait
    // pre-increments cq_signal_counter_[cq.id()] and uses it for both the dispatcher
    // write and the WAIT_CQ request value.
    std::array<uint32_t, kNumCqSignalSlots> cq_signal_counter_{};

    // sender_logical_cores_[s] is the logical DRAM core for sender slot s, a (bank,
    // primary/secondary role) pair. Both sender cores per bank are provisioned at start; each
    // queued GCB may map the primary only or both, and PREFETCH requests target that subset.
    // One list covers the whole mesh (see metal_SocDescriptor::dram_bank_endpoint_coords);
    // enumerate_dram_senders TT_FATALs if a device disagrees.
    std::vector<CoreCoord> sender_logical_cores_;
    uint32_t num_senders_ = 0;
    uint32_t num_banks_ = 0;

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
