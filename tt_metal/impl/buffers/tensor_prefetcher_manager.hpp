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
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_trace_id.hpp>

#include "tt_metal/common/scoped_timer.hpp"
#include "impl/buffers/tensor_prefetcher_fifo.hpp"
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
class PrefetcherSenderBackend;
class RequestTransport;
class TensorPrefetcherManagerTestAccess;

struct TensorPrefetcherPreconditions {
    bool ok = false;
    // Empty when ok; otherwise the first precondition that failed.
    std::string reason;
};

// Long-lived Tensor prefetcher (DRISC) for a single MeshDevice. Holds the
// per-device Programs, the request transport (an H2D socket per (device, sender) on
// Blackhole, a TL1 request ring per CCE sender on Quasar), and the host worker
// thread that drains the request queue. It does NOT own or hold the queued tensors
// or pipes alive — queue() serializes each request into request pages and keeps only
// those bytes, so the caller must keep the tensors and pipes alive until stop() (see
// the public tensor_prefetcher.hpp note).
//
// Single-prefetcher-at-a-time invariant: start() asserts is_active() is false.
//
// Lifecycle:
//   * start(config) builds one Program per IDevice in the mesh, with a sender
//     kernel on every provisioned DRAM sender core, and the request transport.
//     Launches the programs (non-blocking — kernels park waiting for their first
//     request page) and spawns the host worker thread.
//   * queue(pipes, subset, tensors) serializes the request into one or more
//     fixed-size request pages (splitting when the tensor list overflows a page)
//     and pushes them onto the internal queue in order. The host worker thread
//     fans each page out to every sender whose mesh coord is in `subset` via
//     non-blocking writes so one slow sender can't starve the others. The caller
//     is responsible for keeping tensors and the pipes alive until stop() (see the
//     public tensor_prefetcher.hpp note).
//   * stop() pushes DRAIN requests naming every target still in memory, then an
//     all-zero STOP page to the full mesh, joins the worker thread, WaitProgramDone
//     on each device, and releases per-cycle resources.
//   * Destructor calls stop().
class TensorPrefetcherManager {
public:
    // `lock_api_function` grabs the owning MeshDevice's api_mutex_ (bound from
    // MeshDeviceImpl::lock_api). start()/queue()/stop() take it for the duration
    // of the call so prefetcher operations serialize against the rest of the
    // device API, mirroring MeshCommandQueueBase. enqueue_cq_signal_and_wait() and
    // enqueue_cq_signal_and_wait_if_active() hold it across their dispatcher write too, which
    // goes through a queue method that does not re-lock the (non-recursive) api_mutex_.
    TensorPrefetcherManager(MeshDevice* mesh_device, std::function<std::lock_guard<std::mutex>()> lock_api_function);
    ~TensorPrefetcherManager();

    TensorPrefetcherManager(const TensorPrefetcherManager&) = delete;
    TensorPrefetcherManager& operator=(const TensorPrefetcherManager&) = delete;
    TensorPrefetcherManager(TensorPrefetcherManager&&) = delete;
    TensorPrefetcherManager& operator=(TensorPrefetcherManager&&) = delete;

    // FIFO depth — how many in-flight requests a single Blackhole socket can hold before
    // back-pressuring. kSocketFifoPages × tt::align(kRequestPageBytes, pcie) per socket: 128 pages
    // of 128 B is a 16 KiB DRISC L1 FIFO that holds many small one-matmul requests in flight.
    static constexpr uint32_t kSocketFifoPages = 128;

    void start(const experimental::TensorPrefetcherConfig& config);

    // Receiver-contiguous tensors support batched delivery and streaming rotation; K-row-major
    // tensors need one pipe per bank. Block sizes may vary per tensor and need not match the initial
    // applied size or divide the fixed ring; the ring must hold at least one block (consumers may
    // require more for lookahead). `prefetcher_pipes` must be every pipe of one
    // CreatePrefetcherPipesForTensorPrefetcher result, in any order.
    //
    // Capture-vs-send contract, and the `trace_capture_cq` precondition: see
    // QueueTensorPrefetcherRequest. Captured pages live in trace_requests_ keyed by the
    // recording trace's MeshTraceId, and are re-queued by replay_trace().
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

    struct Request {
        // One logical request page. PREFETCH carries one page per entry in target_sender_indices:
        // the header names that sender's target state, and its layout slots that sender's slab
        // base. STOP / WAIT_SIGNAL carry one
        // shared page and leave target_sender_indices empty to broadcast to every provisioned
        // sender.
        std::vector<std::vector<uint8_t>> sender_pages;
        std::vector<MeshCoordinate> target_devices;
        std::vector<uint32_t> target_sender_indices;
        // Set on a wait on op signal s, which carries no page until worker_loop builds its WAIT_SIGNAL.
        std::optional<uint32_t> op_signal;
    };

    // Requests recorded into a CommandListBuilder and handed to the CommandLists it builds. Shared and
    // immutable, so every replay re-queues the same pages. Valid only while the prefetcher run that
    // serialized them (start() to stop()) is active.
    struct CapturedRequests {
        uint64_t run_id = 0;
        std::vector<std::shared_ptr<const Request>> requests;
    };

    // Serialize a request as queue() does and append it to `captured` instead of sending it. Same
    // preconditions as QueueTensorPrefetcherRequest.
    void capture(
        CapturedRequests& captured,
        const std::vector<std::reference_wrapper<const experimental::PrefetcherPipe>>& prefetcher_pipes,
        ttsl::optional_reference<const MeshCoordinateRangeSet> device_subset,
        const std::vector<experimental::TensorPrefetcherInput>& tensors);

    // Re-queue `captured` for immediate fan-out, in capture order. No-op if it is empty. TT_FATALs if the
    // prefetcher run that captured it has stopped. Caller holds the API lock, which keeps a concurrent stop()
    // from slipping in between the run check and the insert.
    void replay_captured_nolock(const CapturedRequests& captured);

    // Make the prefetcher wait until all work currently enqueued on command queue
    // `cq` has landed before it reads DRAM. Bumps a host-side per-CQ counter,
    // has the dispatcher write the new value into every DRAM core's CQ fence slot
    // (ordered after prior CQ work), and queues a WAIT_SIGNAL request so each kernel
    // blocks until that value is observed. Must be called synchronously on the
    // host thread that enqueues the data writes (after them, before the dependent
    // prefetch request). `cq` must belong to this manager's mesh device, and its id
    // must be within [0, kNumCqSignalSlots).
    void enqueue_cq_signal_and_wait(
        MeshCommandQueue& cq, ttsl::optional_reference<const MeshCoordinateRangeSet> device_subset);
    // enqueue_cq_signal_and_wait over the full mesh if a prefetcher is running, checked under the
    // same lock so a stop() on another thread cannot land in between. Returns false, having done
    // nothing, if none is running.
    bool enqueue_cq_signal_and_wait_if_active(MeshCommandQueue& cq);

    // DRISC L1 address of op signal `signal_id`'s slot, uniform across every sender core of every device.
    uint32_t signal_address(uint32_t signal_id);

    // Queue a WAIT_SIGNAL on op signal `signal_id`'s slot to every sender of every device. The count it
    // waits for is assigned when worker_loop takes the request, so a wait captured into a trace waits for
    // a new count on each replay. Capture-vs-send contract as queue().
    void queue_wait_for_signal(uint32_t signal_id, MeshCommandQueue* trace_capture_cq);

    // Stops the senders and releases this session. If waiting for the senders throws, it logs their
    // statuses, releases everything but the DRISC L1 ranges a DRAIN was reading and the DRAM receiver
    // zones their senders write, and rethrows. If the worker gave up on a sender whose request ring took
    // no page, it logs the statuses, keeps those ranges too, and throws once everything else is released.
    // If a Quasar CCE sender ended Failed, or its status cannot be read, it logs the statuses and throws
    // once everything is released, naming each such sender with the request pages left unread in its ring,
    // and how many more the worker dropped after it exited.
    void stop();

    // stop() for the destructor and device close: a failure is logged, not thrown, and the worker thread is
    // joined on every path.
    void stop_without_throwing();

    bool is_active() const { return active_; }

private:
    friend class TensorPrefetcherManagerTestAccess;

    // Everything the request path needs to know about a delivery target, built by target_for() from
    // the DRAM-sender PrefetcherPipes of one factory call. Owns its mapping by value: the pipes'
    // mapping is assembled at queue time and has no home on the pipe objects.
    //
    // It holds no reference to the target itself, and must not start to. The queue call only borrows
    // the target, and the worker thread that later sends the pages needs nothing but addresses and a
    // mapping.
    struct RequestTarget {
        // Sender core -> receivers, in the order that fixes bank-local slab numbering.
        std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
        // DRISC L1 address of each sender's pipe config page (header field target_config_page_addr), in
        // mapping order.
        std::vector<uint32_t> config_page_addr_per_sender;
        // The DRISC L1 range each of those addresses lies in, in mapping order. Weak, like the rest of
        // this struct: it only tells stop whether that range still belongs to the target.
        std::vector<std::weak_ptr<DriscL1Allocation>> config_page_alloc_per_sender;
        // The DRAM receivers' receiver-zone range the senders store data and credits into, held with the state
        // ranges once stop drains them. Empty for worker receivers.
        std::weak_ptr<DriscL1Allocation> receiver_alloc;
        // Bank-local slab index of each sender's first receiver, in mapping order: local receiver
        // r of sender s reads slab recv_index_base_per_sender[s] + r. Stamped into every layout
        // slot of that sender's page.
        std::vector<uint32_t> recv_index_base_per_sender;
        // Per-receiver ring capacity in bytes; a tensor's page_bytes_per_recv must fit.
        uint32_t per_recv_capacity_bytes = 0;
        // Receivers of every sender together (the ring size).
        uint32_t total_receivers = 0;
    };

    // Validate and serialize one Queue call into shared requests, one per logical page, and record its
    // drain targets. Caller holds the API lock and has checked active_.
    std::vector<std::shared_ptr<const Request>> build_requests(
        const RequestTarget& target,
        ttsl::optional_reference<const MeshCoordinateRangeSet> device_subset,
        const std::vector<experimental::TensorPrefetcherInput>& tensors);
    // The host worker for H2D-socket senders: batches each socket's pages and publishes them together.
    void worker_loop(H2DSocketTransport& sockets);
    // The host worker for CCE senders: one request at a time into each sender's TL1 request ring through
    // transport_, dropping pages to a sender that has exited or that the transport gave up on.
    void worker_loop_request_ring();
    // Whether the senders are Quasar CCEs on their own DRAM cores. Quasar's dispatch-engine senders are not CCEs.
    bool uses_cce_senders() const;
    // The body of enqueue_cq_signal_and_wait. The caller holds the API lock and has checked active_.
    void enqueue_cq_signal_and_wait_locked(
        MeshCommandQueue& cq, ttsl::optional_reference<const MeshCoordinateRangeSet> device_subset);
    // One WAIT_SIGNAL page on signal slot `slot_index` for `wait_value`.
    std::vector<uint8_t> wait_signal_page(uint32_t slot_index, uint32_t wait_value) const;
    // If `request` is a wait on an op signal, replaces it with a copy carrying its WAIT_SIGNAL page, which waits
    // for the next count of that signal. The worker thread calls it as it takes each request.
    void add_op_signal_wait_page(std::shared_ptr<const Request>& request);
    // TT_FATAL unless op signal `signal_id`, which `api_name`'s caller passed in, is in range.
    static void validate_signal_id(uint32_t signal_id, std::string_view api_name);
    // L1 address of signal slot `slot_index` on every core that keeps a signal slot table: each DRAM-core sender,
    // or the dispatch engine that keeps the one table for all of its senders.
    uint32_t signal_slot_addr(uint32_t slot_index) const;
    // Add the pipe config page at `config_page_addr` to sender slot `sender`'s drain targets, dropping
    // the ones whose memory is gone.
    void record_drain_target(
        uint32_t sender,
        uint32_t config_page_addr,
        const std::weak_ptr<DriscL1Allocation>& allocation,
        const std::weak_ptr<DriscL1Allocation>& receivers);
    void record_drain_targets(const RequestTarget& target, const std::vector<uint32_t>& target_sender_indices);
    // One DRAIN request per page's worth of every sender's live drain targets, for stop to send ahead
    // of STOP.
    std::vector<Request> build_drain_requests();
    // TT_FATALs unless every page is one socket page, which is what worker_loop sends from each. Runs on
    // the caller's thread before a request is queued, since a TT_FATAL on the worker thread would end
    // the process.
    void validate_request_pages(const std::vector<std::vector<uint8_t>>& sender_pages) const;
    // Queues one WAIT_SIGNAL page on CQ fence slot `cq_id` for every provisioned sender of `target_devices`: wait
    // until that slot reaches `signal_value`.
    void queue_wait_cq_request(uint32_t cq_id, uint32_t signal_value, std::vector<MeshCoordinate> target_devices);
    // Provisions the senders of `sender_banks` (empty = every bank), `roles_per_bank` on each, on every device of
    // the mesh.
    void enumerate_dram_senders(const std::vector<uint32_t>& sender_banks, size_t roles_per_bank);
    RequestTarget target_for(
        const std::vector<std::reference_wrapper<const experimental::PrefetcherPipe>>& prefetcher_pipes) const;
    std::vector<uint32_t> sender_indices_for_target(const RequestTarget& target) const;
    // Throws on a device without DRAM-core senders.
    PrefetcherSenderBackend& backend() const;
    // Tells the worker to finish and joins it, if it is still running.
    void join_worker();
    // Serialize a Queue call's tensors into one or more request pages, deduplicating
    // tensor layouts within each page and splitting when a page fills. Returns one entry per
    // logical page, each a vector in target sender-mapping order. The entry and geometry bytes
    // are identical across senders, while the header carries that sender's config page address and slab
    // base, and a streaming page carries only that sender's slice of the per-receiver rotation
    // table.
    std::vector<std::vector<std::vector<uint8_t>>> serialize_request_pages(
        const RequestTarget& target, const std::vector<experimental::TensorPrefetcherInput>& data_tensors) const;
    // TT_FATAL unless `cq`, which `api_name`'s caller passed in, is null or of this mesh device.
    void validate_own_cq(const MeshCommandQueue* cq, std::string_view api_name) const;
    // Capture `requests` into the trace `trace_capture_cq` is recording, if it is, else hand them to the
    // worker, in order and without interleaving with any other call's requests.
    void send_or_capture(
        const std::vector<std::shared_ptr<const Request>>& requests, MeshCommandQueue* trace_capture_cq);
    MeshCoordinateRangeSet full_mesh_subset() const;

    // Sets stage_ring_size_ from `raw_bytes`, with the halves and thirds the request serializer sizes
    // layouts from.
    void set_stage_ring_size(uint32_t raw_bytes, uint32_t l1_alignment);

    // DRAM-sender endpoints on the dispatch engine (DramSenderPlacement::DispatchEngine). Fast dispatch
    // already runs the kernels that serve them, and requests go into a FIFO in the engine's L1
    // instead of one H2D socket per sender. Defined in tensor_prefetcher_manager_dispatch_engine.cpp.
    bool on_dispatch_engine() const;
    // The rest of start() for this placement: stage-ring geometry, FIFO state, the worker.
    void start_on_dispatch_engine();
    // Rejects, before a request records any state, what the dispatch-engine kernels do not serve.
    void validate_dispatch_engine_pages(
        const std::vector<std::vector<uint8_t>>& sender_pages, const RequestTarget& target) const;
    void worker_loop_dispatch_engine();
    // Writes one FIFO slot to device `d`, waiting for a free slot first.
    void write_fifo_slot(uint32_t d, std::span<const uint8_t> slot);
    // Slots every kernel on device `d` has retired.
    uint32_t fifo_slots_retired(uint32_t d) const;
    // Waits until every kernel on device `d` has retired `target` slots. With an operation timeout set
    // (TT_METAL_OPERATION_TIMEOUT_SECONDS), no slot retiring for that long is a hang: it runs the dispatch
    // timeout hook and throws, which on the worker thread ends the process.
    void wait_for_fifo_retired(uint32_t d, uint32_t target) const;
    // Waits until every kernel on every device has retired every slot written to it.
    void wait_for_fifo_idle() const;

    MeshDevice* mesh_device_;
    // Grabs the owning MeshDevice's api_mutex_ for the duration of an API call.
    std::function<std::lock_guard<std::mutex>()> lock_api_function_;
    // What differs between the kinds of DRAM-core sender, chosen by start(); none on the dispatch engine, or on a
    // device without DRAM-sender endpoints.
    std::unique_ptr<PrefetcherSenderBackend> backend_;
    bool active_ = false;
    // Incremented by each start(), so requests captured for a CommandList can tell which run they belong to.
    uint64_t run_id_ = 0;
    uint32_t stage_ring_size_ = 0;
    uint32_t ring_half_ = 0;
    uint32_t stage_third_ = 0;
    // Base (local L1) of this prefetcher's signal slot table: kNumCqSignalSlots CQ fence slots, then
    // kNumOpSignalSlots op signal slots. Uniform across all sender cores of every device, carved at the front of
    // the DRISC kernel working region, or the table in the dispatch engine's region.
    uint32_t signal_slots_l1_addr_ = 0;
    // Distance between consecutive signal slots. The slots hold one uint32 each but are
    // spaced a full L1 alignment apart: each CQ fence slot is the destination of its own dispatcher
    // write, and a dispatch write only lands on an L1-aligned address. Packed 4 bytes
    // apart, every slot but the first would be misaligned and its write would go nowhere,
    // leaving the kernel spinning on a WAIT_SIGNAL that is never satisfied.
    uint32_t signal_slot_stride_ = 0;
    // A pipe stop must drain on one sender: the address of its config page on that sender, which the
    // DRAIN request carries, and the DRISC L1 range that address lies in. The range expiring means the
    // pipe is gone and its memory may already hold something else, so stop skips it rather than read
    // that memory as counters. `receivers` is the pipe's DRAM receiver zone, if it has one.
    struct DrainTarget {
        uint32_t config_page_addr = 0;
        std::weak_ptr<DriscL1Allocation> allocation;
        std::weak_ptr<DriscL1Allocation> receivers;
    };
    // drain_targets_per_sender_[s]: every target sender slot s has been queued for since start.
    std::vector<std::vector<DrainTarget>> drain_targets_per_sender_;
    // The ranges stop is draining, and the DRAM receiver zones their senders write, held until the kernels have
    // exited so none can be reused while a DRAIN reads it or a sender writes it.
    std::vector<std::shared_ptr<DriscL1Allocation>> drain_holds_;
    // drain_holds_ of every Stop whose senders did not finish: a sender or receiver may still write those
    // ranges, so they are never released while this manager lives.
    std::vector<std::shared_ptr<DriscL1Allocation>> drain_holds_after_failed_stop_;
    // Host-side monotonic signal counter per command queue. enqueue_cq_signal_and_wait
    // pre-increments cq_signal_counter_[cq.id()] and uses it for both the dispatcher
    // write and the WAIT_SIGNAL request value.
    std::array<uint32_t, kNumCqSignalSlots> cq_signal_counter_{};
    // Waits on each op signal handed to the worker since start, i.e. the count the latest of them waits
    // for. Zeroed by start, as the slots are, before the worker starts; only worker_loop touches it after.
    std::array<uint32_t, experimental::kTensorPrefetcherNumSignals> op_signal_waits_{};
    // How much one op signal raises a slot the kernels wait on: one per DRAM bank whose counter that slot is. A
    // DRAM-core sender keeps its own bank's counter; a dispatch engine's one table holds every bank's.
    uint32_t op_signal_increments_ = 1;

    // sender_logical_cores_[s] is the logical DRAM core for sender slot s, a (bank,
    // primary/secondary role) pair. Every sender core of each provisioned bank is provisioned at
    // start (two on Blackhole, one on Quasar); each queued pipe set may map the primary only or
    // both, and PREFETCH requests target that subset.
    // One list covers the whole mesh (see metal_SocDescriptor::dram_bank_endpoint_coords);
    // enumerate_dram_senders TT_FATALs if a device disagrees.
    std::vector<CoreCoord> sender_logical_cores_;
    uint32_t num_senders_ = 0;
    uint32_t num_banks_ = 0;

    // One program per IDevice in the mesh; programs_[d].
    std::vector<std::unique_ptr<Program>> programs_;
    std::vector<IDevice*> devices_;

    // Delivers request pages to DRAM-core senders; endpoint d * num_senders_ + s is (device d, sender s).
    // Null on the dispatch engine, whose requests go into its FIFO.
    std::unique_ptr<RequestTransport> transport_;
    // DispatchEngine placement: the engine's L1 region base, and per device the engine holding its
    // FIFO and the slots written to it.
    struct DeviceFifo {
        CoreCoord dispatch_engine;
        uint32_t written = 0;
    };
    uint32_t fifo_region_base_ = 0;
    tensor_prefetcher_fifo::RegionLayout fifo_layout_;
    std::vector<DeviceFifo> fifos_;

    // MeshCoordinate -> index into devices_, populated once at start() so
    // worker_loop fan-out is O(targets) instead of O(targets * devices).
    std::unordered_map<MeshCoordinate, uint32_t> device_index_by_coord_;

    // Request pages the worker dropped because their sender had exited, so stop() can report them.
    std::atomic<uint32_t> dropped_request_pages_{0};

    // Host worker thread + queue
    std::thread host_worker_;
    std::mutex queue_mu_;
    std::condition_variable queue_cv_;
    // Shared and immutable: a trace replay queues its captured requests again without copying their pages.
    std::deque<std::shared_ptr<const Request>> pending_;
    // Whether pending_ holds requests the worker has not taken; only written under queue_mu_. Lets
    // worker_loop skip the lock while it waits for room in the socket FIFOs and nothing new has come.
    std::atomic<bool> has_pending_{false};
    std::atomic<bool> stop_requested_{false};

    // Requests captured during trace capture, keyed by the recording trace's id. Populated by
    // queue() when its command queue is mid-capture; drained back onto pending_ by
    // replay_trace() on each trace execution; erased by release_trace(). Guarded by queue_mu_.
    std::unordered_map<MeshTraceId, std::vector<std::shared_ptr<const Request>>> trace_requests_;

    // In benchmark mode, created only after kernels and the worker are live and
    // destroyed immediately after shutdown/drain so setup and cleanup are excluded.
    std::optional<tt::ScopedTimer<std::chrono::microseconds>> active_lifetime_timer_;
};

}  // namespace distributed
}  // namespace tt::tt_metal
