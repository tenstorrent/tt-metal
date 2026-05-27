// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/dram_core_prefetcher.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal {

class IDevice;
class MeshTensor;
class Program;

namespace distributed {

class MeshDevice;

// Long-lived DRAM-core (DRISC) prefetcher for a single MeshDevice. Holds the
// per-device Programs, the per-(device, sender) H2D sockets, the host worker
// thread that drains the request queue, and the tensor references held alive
// for the lifetime of the Start/Stop cycle.
//
// Single-prefetcher-at-a-time invariant: start() asserts is_active() is false.
//
// Lifecycle:
//   * start(config) builds one Program per IDevice in the mesh, with a DRISC
//     kernel on every DRAM sender core. Allocates one H2D socket per (device,
//     sender). Launches the programs (non-blocking — kernels park on
//     socket_wait_for_pages) and spawns the host worker thread.
//   * queue(gcb, subset, tensors, num_layers) serializes one request into a
//     fixed-size socket page and pushes it onto the internal queue. The host
//     worker thread fans it out to every socket whose mesh coord is in
//     `subset` via non-blocking try_write so one slow socket can't starve the
//     others. Tensor shared_ptrs are held until stop().
//   * stop() pushes the stop sentinel onto the queue, joins the worker thread
//     (which broadcasts the sentinel to every socket), WaitProgramDone on
//     each device, releases held resources.
//   * Destructor calls stop().
class DramCorePrefetcherManager {
public:
    explicit DramCorePrefetcherManager(MeshDevice* mesh_device);
    ~DramCorePrefetcherManager();

    DramCorePrefetcherManager(const DramCorePrefetcherManager&) = delete;
    DramCorePrefetcherManager& operator=(const DramCorePrefetcherManager&) = delete;
    DramCorePrefetcherManager(DramCorePrefetcherManager&&) = delete;
    DramCorePrefetcherManager& operator=(DramCorePrefetcherManager&&) = delete;

    void start(const experimental::DramCorePrefetcherConfig& config);

    void queue(
        const experimental::GlobalCircularBuffer& gcb,
        const std::optional<MeshCoordinateRangeSet>& device_subset,
        const std::vector<const MeshTensor*>& tensors,
        uint32_t num_layers);

    void stop();

    bool is_active() const { return active_; }

private:
    // ---- Constants shared with the kernel side ----
    // Max tensors per request. Matches the kernel's per-tensor stride in the
    // socket payload (10 uint32 fields per tensor). 16 is enough for Llama
    // production shapes (typical is 5–10 tensors per matmul layer).
    static constexpr uint32_t kMaxTensorsPerRequest = 16;
    static constexpr uint32_t kRequestPageHeaderWords = 4;   // num_tensors, num_layers, num_blocks, gcb_state_addr
    static constexpr uint32_t kRequestPageTensorWords = 10;  // see kernel doc
    static constexpr uint32_t kRequestPageBytes =
        sizeof(uint32_t) * (kRequestPageHeaderWords + kMaxTensorsPerRequest * kRequestPageTensorWords);

    // FIFO depth — how many in-flight requests a single socket can hold before
    // back-pressuring. 16 pages × ~656 B per page ≈ 11 KB per socket.
    static constexpr uint32_t kSocketFifoPages = 16;

    struct Request {
        // is_stop_sentinel == true means a Stop broadcast: send to every socket.
        bool is_stop_sentinel = false;
        std::vector<uint8_t> page;  // size = kRequestPageBytes; identical bytes for every target
        std::vector<MeshCoordinate> target_devices;
        std::vector<std::shared_ptr<const MeshTensor>> tensor_refs;
    };

    void worker_loop();
    void enumerate_dram_senders();
    void build_and_launch_programs(uint32_t stage_ring_base, uint32_t stage_ring_size);
    void allocate_sockets();
    // Serialize a Queue call's bytes into one socket page.
    std::vector<uint8_t> serialize_request_page(
        const experimental::GlobalCircularBuffer& gcb,
        const std::vector<const MeshTensor*>& data_tensors,
        uint32_t num_layers) const;
    MeshCoordinateRangeSet full_mesh_subset() const;

    MeshDevice* mesh_device_;
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

    // Host worker thread + queue
    std::thread host_worker_;
    std::mutex queue_mu_;
    std::condition_variable queue_cv_;
    std::deque<Request> pending_;
    std::atomic<bool> stop_requested_{false};

    // Tensor refs of completed requests, held alive until stop().
    std::vector<Request> held_;
};

}  // namespace distributed
}  // namespace tt::tt_metal
