// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
#include <tt-metalium/experimental/dram_core_prefetcher.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "impl/buffers/dram_core_prefetcher_request.hpp"

namespace tt::tt_metal {

class IDevice;
class MeshTensor;
class Program;

namespace distributed {

class MeshDevice;

// Long-lived DRAM-core (DRISC) prefetcher for a single MeshDevice. Holds the
// per-device Programs, the per-(device, sender) H2D sockets, and the host worker
// thread that drains the request queue. It does NOT own or hold the queued tensors
// or GCBs alive — queue() serializes each request into socket pages and keeps only
// those bytes, so the caller must keep the tensors and GCB alive until stop() (see
// the public dram_core_prefetcher.hpp note).
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
//     (see the public dram_core_prefetcher.hpp note).
//   * stop() pushes a zero-tensor request targeting the full mesh, joins the
//     worker thread (the kernel exits on `num_entries == 0`), WaitProgramDone
//     on each device, releases per-cycle resources.
//   * Destructor calls stop().
class DramCorePrefetcherManager {
public:
    // `lock_api_function` grabs the owning MeshDevice's api_mutex_ (bound from
    // MeshDeviceImpl::lock_api). start()/queue()/stop() take it for the duration
    // of the call so prefetcher operations serialize against the rest of the
    // device API, mirroring MeshCommandQueueBase.
    DramCorePrefetcherManager(MeshDevice* mesh_device, std::function<std::lock_guard<std::mutex>()> lock_api_function);
    ~DramCorePrefetcherManager();

    DramCorePrefetcherManager(const DramCorePrefetcherManager&) = delete;
    DramCorePrefetcherManager& operator=(const DramCorePrefetcherManager&) = delete;
    DramCorePrefetcherManager(DramCorePrefetcherManager&&) = delete;
    DramCorePrefetcherManager& operator=(DramCorePrefetcherManager&&) = delete;

    void start(const experimental::DramCorePrefetcherConfig& config);

    void queue(
        const experimental::GlobalCircularBuffer& gcb,
        const std::optional<MeshCoordinateRangeSet>& device_subset,
        const std::vector<experimental::DramCorePrefetcherInput>& tensors);

    void stop();

    bool is_active() const { return active_; }

private:
    // ---- Constants shared with the kernel side ----
    // The request page wire format (header + entry table + deduplicated layout table)
    // and the fixed payload size kRequestPageBytes live in
    // impl/buffers/dram_core_prefetcher_request.hpp so the host and the kernel agree on
    // both the byte layout and the payload size. A Queue call whose tensors overflow one
    // page is split across multiple pages.

    // FIFO depth — how many in-flight requests a single socket can hold before
    // back-pressuring. kSocketFifoPages × align_up(kRequestPageBytes, pcie) per socket.
    static constexpr uint32_t kSocketFifoPages = 16;

    struct Request {
        std::vector<uint8_t> page;  // one socket page; identical bytes for every target
        std::vector<MeshCoordinate> target_devices;
    };

    void worker_loop();
    void enumerate_dram_senders();
    void build_and_launch_programs(uint32_t stage_ring_base, uint32_t stage_ring_size);
    void allocate_sockets();
    // Serialize a Queue call's tensors into one or more socket pages, deduplicating
    // tensor layouts within each page and splitting when a page fills.
    std::vector<std::vector<uint8_t>> serialize_request_pages(
        const experimental::GlobalCircularBuffer& gcb,
        const std::vector<experimental::DramCorePrefetcherInput>& data_tensors) const;
    MeshCoordinateRangeSet full_mesh_subset() const;

    MeshDevice* mesh_device_;
    // Grabs the owning MeshDevice's api_mutex_ for the duration of an API call.
    std::function<std::lock_guard<std::mutex>()> lock_api_function_;
    bool active_ = false;
    uint32_t stage_ring_base_ = 0;
    uint32_t stage_ring_size_ = 0;
    uint32_t ring_half_ = 0;
    // Per-DRAM-core L1 layout (uniform across all sender cores on all devices).
    // socket_config / socket_data are local L1 addresses; host writes add the
    // DRAM_L1_NOC_OFFSET (passed into H2DSocket's DRAM-recv ctor) before going
    // over NOC.
    uint32_t socket_config_l1_addr_ = 0;
    uint32_t socket_data_l1_addr_ = 0;

    // sender_logical_cores_[s] = logical DRAM core for bank s. Picked at start
    // via pick_unused_dram_logical_core(s); GCBs queued must use the same
    // picks (deterministic on bank_id), which they do because the GCB factory
    // calls the same picker.
    std::vector<CoreCoord> sender_logical_cores_;
    uint32_t num_senders_ = 0;

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
};

}  // namespace distributed
}  // namespace tt::tt_metal
