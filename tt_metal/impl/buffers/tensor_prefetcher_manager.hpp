// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_trace_id.hpp>

#include "tt_metal/common/scoped_timer.hpp"
#include "impl/buffers/tensor_prefetcher_request.hpp"

namespace tt::tt_metal {

class DriscL1Allocation;
class IDevice;
class MeshTensor;
class Program;

namespace distributed {

class H2DSocketTransport;
class MeshCommandQueue;
class MeshDevice;
class RequestTransport;

struct TensorPrefetcherPreconditions {
    bool ok = false;
    // Empty when ok; otherwise the first precondition that failed.
    std::string reason;
};

// What StartTensorPrefetcher requires of `mesh_device`. IsTensorPrefetcherSupported is this. On Quasar
// the firmware term requires every sender CCE's GO_MSG to hold RUN_MSG_DONE or RUN_MSG_GO: it checks
// that firmware is present, not that the CCE is idle, so it holds while receiver Programs and
// prefetcher sessions run.
TensorPrefetcherPreconditions tensor_prefetcher_preconditions(const MeshDevice& mesh_device);

// TL1 offset of every Quasar CCE sender's control block (tensor_prefetcher_cce_abi.hpp). It depends only on
// the mesh's DRISC L1 arena, so it holds before Start and after Stop.
uint32_t cce_sender_control_base(const MeshDevice& mesh_device);

// Long-lived Tensor prefetcher (DRISC) for a single MeshDevice. Holds the
// per-device Programs, the request transport (an H2D socket per (device, sender) on
// Blackhole, a TL1 request ring per CCE sender on Quasar), and the host worker
// thread that drains the request queue. It does NOT own or hold the queued tensors
// or GCBs alive — queue() serializes each request into request pages and keeps only
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
//     fixed-size request pages (splitting when the tensor list overflows a page)
//     and pushes them onto the internal queue in order. The host worker thread
//     fans each page out to every socket whose mesh coord is in `subset` via
//     non-blocking try_write so one slow socket can't starve the others. The
//     caller is responsible for keeping tensors and the GCB alive until stop()
//     (see the public tensor_prefetcher.hpp note).
//   * stop() pushes DRAIN requests naming every target still in memory, then an
//     all-zero STOP page to the full mesh, joins the worker thread, WaitProgramDone
//     on each device, and releases per-cycle resources.
//   * Destructor calls stop().
class TensorPrefetcherManager {
public:
    // `lock_api_function` grabs the owning MeshDevice's api_mutex_ (bound from
    // MeshDeviceImpl::lock_api). start()/queue()/stop() take it for the duration
    // of the call so prefetcher operations serialize against the rest of the
    // device API, mirroring MeshCommandQueueBase. enqueue_cq_signal_and_wait() also
    // holds it for the whole call, across the dispatcher write, which does not re-lock
    // it, so the counter bump, that write and the WAIT_CQ are queued atomically.
    TensorPrefetcherManager(MeshDevice* mesh_device, std::function<std::lock_guard<std::mutex>()> lock_api_function);
    ~TensorPrefetcherManager();

    TensorPrefetcherManager(const TensorPrefetcherManager&) = delete;
    TensorPrefetcherManager& operator=(const TensorPrefetcherManager&) = delete;
    TensorPrefetcherManager(TensorPrefetcherManager&&) = delete;
    TensorPrefetcherManager& operator=(TensorPrefetcherManager&&) = delete;

    // FIFO depth — how many in-flight requests a single socket can hold before
    // back-pressuring. kSocketFifoPages × align_up(kRequestPageBytes, pcie) per socket.
    // Scaled inversely with kRequestPageBytes (128 × 128 B == the previous 16 × 1024 B) so
    // shrinking the page to one-matmul granularity keeps the per-socket DRISC L1 FIFO at the
    // same footprint while allowing 8× more small in-flight requests.
    static constexpr uint32_t kSocketFifoPages = 128;

    void start(const experimental::TensorPrefetcherConfig& config);

    // Capture-vs-send contract, and the `trace_capture_cq` precondition: see
    // QueueTensorPrefetcherRequest. Captured pages live in trace_requests_ keyed by the
    // recording trace's MeshTraceId, and are re-queued by replay_trace().
    void queue(
        const experimental::GlobalCircularBuffer& gcb,
        ttsl::optional_reference<const MeshCoordinateRangeSet> device_subset,
        const std::vector<experimental::TensorPrefetcherInput>& tensors,
        MeshCommandQueue* trace_capture_cq);

    // PrefetcherPipe delivery. Receiver-contiguous tensors support batched delivery and streaming
    // rotation. Block sizes may vary per tensor and need not match the initial applied size or divide
    // the fixed ring; the ring must hold at least one block (consumers may require more for lookahead).
    // `prefetcher_pipes` must be every pipe of one CreatePrefetcherPipesForTensorPrefetcher result, in
    // any order.
    void queue(
        const std::vector<std::reference_wrapper<const experimental::PrefetcherPipe>>& prefetcher_pipes,
        ttsl::optional_reference<const MeshCoordinateRangeSet> device_subset,
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
    void enqueue_cq_signal_and_wait(
        MeshCommandQueue& cq, ttsl::optional_reference<const MeshCoordinateRangeSet> device_subset);

    // Stops the senders and releases this session. If waiting for the senders throws, it logs their
    // statuses, releases everything but the DRISC L1 ranges a DRAIN was reading, and rethrows. If the
    // worker gave up on a sender whose request ring took no page, it logs the statuses, keeps those ranges
    // too, and throws once everything else is released. If a Quasar CCE sender ended Failed, or its status
    // cannot be read, it logs the statuses and throws once everything is released, naming each such sender
    // and how many request pages the worker dropped after it exited.
    void stop();

    // stop() for the destructor and device close: a failure is logged, not thrown, and the worker thread is
    // joined on every path.
    void stop_without_throwing();

    // Records `gcb` on its senders for Stop's DRAIN, as queueing a request to it would, without sending a
    // request. For tests that drain a live GlobalCircularBuffer without the PREFETCH path.
    void register_drain_target(const experimental::GlobalCircularBuffer& gcb);

    // A WAIT_CQ the dispatcher never satisfies: the local DRISC L1 address of the CQ signal slot, the same on every
    // provisioned sender, and the value the senders wait for there.
    struct UnsignaledCqWait {
        uint32_t slot_address = 0;
        uint32_t value = 0;
    };

    // enqueue_cq_signal_and_wait for command queue 0 on every device, without the dispatcher write: it bumps the
    // counter and queues the WAIT_CQ, so the senders wait until the caller writes `value` into the slot. For tests of
    // the fence, which slow dispatch otherwise satisfies before the WAIT_CQ is queued.
    UnsignaledCqWait queue_cq_wait_without_signal();

    bool is_active() const { return active_; }

private:
    // ---- Constants shared with the kernel side ----
    // The request page wire format (header + entry table + deduplicated layout table)
    // and the fixed payload size kRequestPageBytes live in
    // impl/buffers/tensor_prefetcher_request.hpp so the host and the kernel agree on
    // both the byte layout and the payload size. A Queue call whose tensors overflow one
    // page is split across multiple pages.

    struct Request {
        // One logical request page. PREFETCH carries one page per entry in target_sender_indices:
        // the header names that sender's target state, and its layout slots that sender's slab
        // base. STOP / WAIT_CQ carry one
        // shared page and leave target_sender_indices empty to broadcast to every provisioned
        // sender.
        std::vector<std::vector<uint8_t>> sender_pages;
        std::vector<MeshCoordinate> target_devices;
        std::vector<uint32_t> target_sender_indices;
    };

    struct MpfeWeights {
        uint32_t free_sender;
        uint32_t noc1_sender;
        uint32_t ordinary;
    };

    struct MpfePolicy {
        MpfeWeights active;
        bool dynamic;
    };

    // Everything the request path needs to know about a delivery target, so serialization does not
    // have to name the target's type. Built by target_for() from either a DRAM-sender
    // GlobalCircularBuffer or the per-bank groups of DRAM-sender PrefetcherPipes. Owns its mapping
    // by value: the pipes' mapping is assembled at queue time and has no home on the pipe objects.
    //
    // It holds no reference to the target itself, and must not start to. The queue call only borrows
    // the target, and the worker thread that later sends the pages needs nothing but addresses and a
    // mapping.
    struct RequestTarget {
        // Sender core -> receivers, in the order that fixes bank-local slab numbering.
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
        // DRISC L1 base of each sender's target state (header field target_state_addr), in mapping
        // order. A GCB plants every sender's block at one uniform offset and so repeats it; the
        // pipes hold one address each.
        std::vector<uint32_t> state_addr_per_sender;
        // The DRISC L1 range each of those addresses lies in, in mapping order. Weak, like the rest of
        // this struct: it only tells stop whether that range still belongs to the target.
        std::vector<std::weak_ptr<DriscL1Allocation>> state_alloc_per_sender;
        // Bank-local slab index of each sender's first receiver, in mapping order: local receiver
        // r of sender s reads slab recv_index_base_per_sender[s] + r. Stamped into every layout
        // slot of that sender's page.
        std::vector<uint32_t> recv_index_base_per_sender;
        // Transport for every tensor in the request.
        TensorPrefetcherTransport transport = TENSOR_PREFETCHER_TRANSPORT_GLOBAL_CB;
        // Per-receiver ring capacity in bytes; a tensor's page_bytes_per_recv must fit.
        uint32_t per_recv_capacity_bytes = 0;
    };

    void worker_loop();
    // Add `word` to sender slot `sender`'s drain targets, dropping the ones whose memory is gone.
    void record_drain_target(uint32_t sender, uint32_t word, const std::weak_ptr<DriscL1Allocation>& state);
    // record_drain_target for each sender `target` maps to.
    void record_drain_targets(const RequestTarget& target, const std::vector<uint32_t>& target_sender_indices);
    // One DRAIN request per page's worth of every sender's live drain targets, for stop to send ahead
    // of STOP.
    std::vector<Request> build_drain_requests();
    // Queues one WAIT_CQ page for every provisioned sender of `target_devices`: wait until CQ signal slot `cq_id`
    // reaches `signal_value`.
    void queue_wait_cq_request(uint32_t cq_id, uint32_t signal_value, std::vector<MeshCoordinate> target_devices);
    // Provisions the senders of `sender_banks` (empty = every bank).
    void enumerate_dram_senders(const std::vector<uint32_t>& sender_banks);
    RequestTarget target_for(const experimental::GlobalCircularBuffer& gcb) const;
    RequestTarget target_for(
        const std::vector<std::reference_wrapper<const experimental::PrefetcherPipe>>& prefetcher_pipes) const;
    std::vector<uint32_t> sender_indices_for_target(const RequestTarget& target) const;
    // A CCE runs one launch at a time, so a receiver on a provisioned sender CCE could never run.
    void require_receivers_off_sender_cces(const RequestTarget& target) const;
    void build_and_launch_programs(
        const H2DSocketTransport& sockets,
        uint32_t stage_ring_base,
        uint32_t stage_ring_size,
        const std::optional<MpfePolicy>& mpfe_policy);
    // A launch onto a CCE still running a kernel is lost, so every provisioned sender CCE must be idle.
    void require_idle_sender_cces() const;
    // Zeroes each sender's CQ signal slots, ring meta and status, which the CCE kernel never does.
    void zero_cce_sender_control_blocks() const;
    void build_cce_sender_programs(uint64_t wait_budget_cycles);
    void log_cce_sender_statuses() const;
    // One line per sender that ended Failed or whose status cannot be read, after log_cce_sender_statuses when
    // there is any. Empty off Quasar. Never throws.
    std::vector<std::string> failed_cce_senders() const;
    // Tells the worker to finish and joins it, if it is still running.
    void join_worker();
    // Serialize a Queue call's tensors into one or more request pages, deduplicating
    // tensor layouts within each page and splitting when a page fills. Returns one entry per
    // logical page, each a vector in target sender-mapping order. The entry and geometry bytes
    // are identical across senders, while the header carries that sender's state address and slab
    // base, and a streaming page carries only that sender's slice of the per-receiver rotation
    // table.
    std::vector<std::vector<std::vector<uint8_t>>> serialize_request_pages(
        const RequestTarget& target, const std::vector<experimental::TensorPrefetcherInput>& data_tensors) const;
    // Shared body of the two public queue() overloads.
    void queue_to_target(
        const RequestTarget& target,
        ttsl::optional_reference<const MeshCoordinateRangeSet> device_subset,
        const std::vector<experimental::TensorPrefetcherInput>& tensors,
        MeshCommandQueue* trace_capture_cq);
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
    // TL1 base of each Quasar CCE sender's control block; uniform across sender cores.
    uint32_t cce_control_base_ = 0;
    // Base (local DRISC L1) of this prefetcher's per-CQ signal slots; uniform
    // across all sender cores. Carved at the front of the kernel working region.
    uint32_t cq_signal_l1_addr_ = 0;
    // Distance between consecutive signal slots. The slots hold one uint32 each but are
    // spaced a full L1 alignment apart: each is the destination of its own dispatcher
    // write, and a dispatch write only lands on an L1-aligned address. Packed 4 bytes
    // apart, every slot but the first would be misaligned and its write would go nowhere,
    // leaving the kernel spinning on a WAIT_CQ that is never satisfied.
    uint32_t cq_signal_slot_stride_ = 0;
    // A target stop must drain on one sender: its DRAIN word (state address | kDrainTargetPipeBit for a
    // pipe) and the DRISC L1 range that address lies in. The range expiring means the target is gone
    // and its memory may already hold something else, so stop skips it rather than read that memory
    // as counters.
    struct DrainTarget {
        uint32_t word = 0;
        std::weak_ptr<DriscL1Allocation> state;
    };
    // drain_targets_per_sender_[s]: every target sender slot s has been queued for since start.
    std::vector<std::vector<DrainTarget>> drain_targets_per_sender_;
    // The ranges stop is draining, held until the kernels have exited so none can be reused while a
    // DRAIN reads it.
    std::vector<std::shared_ptr<DriscL1Allocation>> drain_holds_;
    // drain_holds_ of every Stop whose senders did not finish: a sender or receiver may still write those
    // ranges, so they are never released while this manager lives.
    std::vector<std::shared_ptr<DriscL1Allocation>> drain_holds_after_failed_stop_;
    // Host-side monotonic signal counter per command queue. enqueue_cq_signal_and_wait
    // pre-increments cq_signal_counter_[cq.id()] and uses it for both the dispatcher
    // write and the WAIT_CQ request value.
    std::array<uint32_t, kNumCqSignalSlots> cq_signal_counter_{};

    // sender_logical_cores_[s] is the logical DRAM core for sender slot s, a (bank,
    // primary/secondary role) pair. Every sender core of each provisioned bank is provisioned at
    // start (two on Blackhole, one on Quasar); each queued GCB may map the primary only or both,
    // and PREFETCH requests target that subset.
    // One list covers the whole mesh (see metal_SocDescriptor::dram_bank_endpoint_coords);
    // enumerate_dram_senders TT_FATALs if a device disagrees.
    std::vector<CoreCoord> sender_logical_cores_;
    uint32_t num_senders_ = 0;
    uint32_t num_banks_ = 0;

    // One program per IDevice in the mesh; programs_[d].
    std::vector<std::unique_ptr<Program>> programs_;
    std::vector<IDevice*> devices_;

    // Delivers request pages; endpoint d * num_senders_ + s is (device d, sender s).
    std::unique_ptr<RequestTransport> transport_;

    // MeshCoordinate -> index into devices_, populated once at start() so
    // worker_loop fan-out is O(targets) instead of O(targets * devices).
    std::unordered_map<MeshCoordinate, uint32_t> device_index_by_coord_;

    // How long the worker waits on a TL1 request ring that takes no page before it gives up on that sender. Set by
    // start() before the worker runs.
    std::chrono::steady_clock::duration request_ring_no_progress_limit_{};
    // Whether the worker also gives up on a STOP page at that limit while its sender still runs. Only when
    // TT_METAL_OPERATION_TIMEOUT_SECONDS is set, so stop() reaches WaitProgramDone and that times out.
    bool request_ring_stop_bounded_ = false;
    // Set by the worker when it gives up on a sender, so stop() reports the requests it dropped.
    std::atomic<bool> worker_gave_up_on_sender_{false};
    // Request pages the worker dropped because their sender had exited, so stop() can report them.
    std::atomic<uint32_t> dropped_request_pages_{0};

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

    // In benchmark mode, created only after kernels and the worker are live and
    // destroyed immediately after shutdown/drain so setup and cleanup are excluded.
    std::optional<tt::ScopedTimer<std::chrono::microseconds>> active_lifetime_timer_;
};

}  // namespace distributed
}  // namespace tt::tt_metal
