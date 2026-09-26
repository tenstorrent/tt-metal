// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pinned_upload.hpp"

#include <unistd.h>

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/memory_pin_access.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/cleanup.hpp>
#include <tt_stl/span.hpp>

#include "common/memory_pin_impl.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/buffers/dispatch.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/distributed/mesh_device_view_impl.hpp"
#include "tt_metal/distributed/pinned_memory_cache.hpp"

namespace tt::tt_metal::pinned_upload {

namespace {

using PinnedMemoryPtr = std::shared_ptr<experimental::PinnedMemory>;

// Chunk indices whose pins stay alive after their writes are enqueued. Releasing an older chunk's pins waits for its
// writes, so this is how far enqueueing may run ahead of the device.
constexpr size_t k_in_flight_chunks = 2;

// Chunk pins each worker may have queued or finished ahead of the enqueueing thread; bounds pinned memory to about
// (k_in_flight_chunks + threads * k_pins_ahead_per_thread) chunks instead of the whole tensor.
constexpr size_t k_pins_ahead_per_thread = 2;

// Distinct (address, size) pairs remembered for promoting re-uploaded host buffers to the pin cache.
constexpr size_t k_seen_uploads_capacity = 64;

// One contiguous host range and the device coordinates it is written to. Shards that share memory (replicated
// storage) are one source, so the range is pinned once per MMIO device.
struct Source {
    HostBuffer host_buffer;  // Keeps the storage alive for the duration of the upload.
    std::byte* base = nullptr;
    size_t size = 0;
    size_t chunk_bytes = 0;  // Set by chunked_upload_applies; see chunk_bytes_for.
    distributed::MeshCoordinateRangeSet coord_range;
    std::vector<distributed::MeshCoordinate> coords;
};

// Process-wide pool that runs chunk pins. Separate from the command queues' dispatch pools, whose workers are pinned
// to devices and whose wait() covers the whole pool.
class PinWorkerPool {
public:
    explicit PinWorkerPool(size_t num_threads) {
        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; i++) {
            workers_.emplace_back([this] { run(); });
        }
    }

    ~PinWorkerPool() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }

    PinWorkerPool(const PinWorkerPool&) = delete;
    PinWorkerPool& operator=(const PinWorkerPool&) = delete;

    size_t num_threads() const { return workers_.size(); }

    std::future<PinnedMemoryPtr> submit(std::function<PinnedMemoryPtr()> fn) {
        std::packaged_task<PinnedMemoryPtr()> task(std::move(fn));
        auto future = task.get_future();
        {
            std::lock_guard lock(mutex_);
            tasks_.push_back(std::move(task));
        }
        cv_.notify_one();
        return future;
    }

private:
    void run() {
        while (true) {
            std::packaged_task<PinnedMemoryPtr()> task;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::packaged_task<PinnedMemoryPtr()>> tasks_;
    bool stop_ = false;
    std::vector<std::thread> workers_;
};

// Created on first use with the thread count configured at that time.
PinWorkerPool& pin_worker_pool(size_t num_threads) {
    static PinWorkerPool pool(num_threads);
    return pool;
}

// Pins of device-immutable uploads that returned before their writes completed.
struct PendingUpload {
    uint32_t cq_id = 0;
    distributed::MeshEvent completion;  // Recorded after the upload's last write.
    std::vector<PinnedMemoryPtr> pins;
    std::vector<MemoryPin> storage;  // Keeps the uploaded memory alive until the pins are released.
};

// Host ranges already uploaded once through chunk pins. A second upload of the same range while its storage is alive
// pins it whole through PinnedMemoryCache instead, so a buffer uploaded repeatedly is pinned once. A range is
// forgotten when its storage is released, because a new mapping of the same size often lands at the same address.
class SeenUploads {
public:
    // Never destroyed: storage released at any point during exit may still call forget().
    static SeenUploads& instance() {
        static auto* seen = new SeenUploads();
        return *seen;
    }

    // Returns whether every source was recorded before; records the ones that were not. Storage without a
    // MemoryPin has no release to forget it on, so it is never recorded.
    bool check_and_record(const std::vector<Source>& sources) {
        std::lock_guard lock(mutex_);
        bool all_seen = true;
        for (const auto& source : sources) {
            const Key key{source.base, source.size};
            if (std::find(keys_.begin(), keys_.end(), key) != keys_.end()) {
                continue;
            }
            all_seen = false;
            MemoryPin pin = source.host_buffer.pin();
            if (pin == nullptr) {
                continue;
            }
            pin.impl().add_final_release_callback([key] { SeenUploads::instance().forget(key); });
            keys_.push_back(key);
            if (keys_.size() > k_seen_uploads_capacity) {
                keys_.pop_front();
            }
        }
        return all_seen;
    }

private:
    using Key = std::pair<const std::byte*, size_t>;

    void forget(const Key& key) {
        std::lock_guard lock(mutex_);
        auto it = std::find(keys_.begin(), keys_.end(), key);
        if (it != keys_.end()) {
            keys_.erase(it);
        }
    }

    std::mutex mutex_;
    std::deque<Key> keys_;
};

// Drops storage references on a background thread. Unmapping a file mapping costs about as much as transferring it
// (page table teardown for every page the pins faulted in), so it stays off the uploading thread.
class StorageReleaser {
public:
    ~StorageReleaser() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void release(std::vector<MemoryPin> storage) {
        if (storage.empty()) {
            return;
        }
        {
            std::lock_guard lock(mutex_);
            if (!thread_.joinable()) {
                thread_ = std::thread([this] { run(); });
            }
            queue_.push_back(std::move(storage));
            num_queued_++;
        }
        cv_.notify_all();
    }

    void wait() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return num_released_ == num_queued_; });
    }

private:
    void run() {
        std::unique_lock lock(mutex_);
        while (true) {
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) {
                return;
            }
            auto storage = std::move(queue_.front());
            queue_.pop_front();
            lock.unlock();
            storage.clear();
            lock.lock();
            num_released_++;
            cv_.notify_all();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::vector<MemoryPin>> queue_;
    size_t num_queued_ = 0;
    size_t num_released_ = 0;
    bool stop_ = false;
    std::thread thread_;
};

class PendingUploads {
public:
    static PendingUploads& instance() {
        static PendingUploads pending;
        return pending;
    }

    PendingUploads() {
        // Releasing storage runs its MemoryPin callbacks, which call into the pin cache; constructing it first
        // destroys it after this one at exit.
        experimental::PinnedMemoryCache::instance();
    }

    void add(const distributed::MeshDevice& mesh_device, PendingUpload upload) {
        std::lock_guard lock(mutex_);
        by_device_[&mesh_device].push_back(std::move(upload));
    }

    // Releases uploads whose writes have completed, without waiting.
    void release_completed(const distributed::MeshDevice& mesh_device) {
        std::vector<PendingUpload> completed;
        {
            std::lock_guard lock(mutex_);
            auto it = by_device_.find(&mesh_device);
            if (it == by_device_.end()) {
                return;
            }
            auto& uploads = it->second;
            // Events on one command queue complete in order, so stop at the first incomplete upload per queue.
            std::set<uint32_t> blocked_cqs;
            for (auto upload = uploads.begin(); upload != uploads.end();) {
                if (!blocked_cqs.contains(upload->cq_id) && distributed::EventQuery(upload->completion)) {
                    completed.push_back(std::move(*upload));
                    upload = uploads.erase(upload);
                } else {
                    blocked_cqs.insert(upload->cq_id);
                    ++upload;
                }
            }
        }
        release(std::move(completed));
    }

    void drain(const distributed::MeshDevice& mesh_device, std::optional<uint32_t> cq_id) {
        std::vector<PendingUpload> drained;
        {
            std::lock_guard lock(mutex_);
            auto it = by_device_.find(&mesh_device);
            if (it == by_device_.end()) {
                return;
            }
            auto& uploads = it->second;
            for (auto upload = uploads.begin(); upload != uploads.end();) {
                if (!cq_id.has_value() || upload->cq_id == *cq_id) {
                    drained.push_back(std::move(*upload));
                    upload = uploads.erase(upload);
                } else {
                    ++upload;
                }
            }
            if (uploads.empty()) {
                by_device_.erase(it);
            }
        }
        release(std::move(drained));
    }

    size_t num_pending(const distributed::MeshDevice& mesh_device) const {
        std::lock_guard lock(mutex_);
        auto it = by_device_.find(&mesh_device);
        return it == by_device_.end() ? 0 : it->second.size();
    }

    void wait_for_storage_release() { storage_releaser_.wait(); }

private:
    // Unpins here, outside the lock: each pin first waits for its own writes. The storage the pins read from is
    // released afterwards, in the background.
    void release(std::vector<PendingUpload> uploads) {
        std::vector<MemoryPin> storage;
        for (auto& upload : uploads) {
            upload.pins.clear();
            for (auto& pin : upload.storage) {
                storage.push_back(std::move(pin));
            }
        }
        uploads.clear();
        storage_releaser_.release(std::move(storage));
    }

    mutable std::mutex mutex_;
    std::unordered_map<const distributed::MeshDevice*, std::deque<PendingUpload>> by_device_;
    StorageReleaser storage_releaser_;
};

size_t os_page_size() {
    const long page_size = sysconf(_SC_PAGESIZE);
    return page_size > 0 ? static_cast<size_t>(page_size) : 4096;
}

std::vector<Source> collect_local_sources(
    const distributed::MeshDevice& mesh_device, const DistributedHostBuffer& host_buffer) {
    const auto& view = mesh_device.get_view();
    std::vector<Source> sources;
    for (const auto& coord : host_buffer.shard_coords()) {
        // get_shard yields a buffer only for shards owned by this host, so remote chips are never pinned or written.
        auto shard = host_buffer.get_shard(coord);
        if (!shard.has_value()) {
            continue;
        }
        // The host buffer's distribution must agree with the device's: host memory can only be pinned to MMIO
        // devices local to this process, so a populated shard for a coord the device owns on another host must never
        // reach a pin (which would fault while resolving the remote device).
        TT_FATAL(
            view.impl().is_local(coord),
            "Host buffer holds a shard for device coordinate {}, but that device is not local to this host; host "
            "memory can only be pinned to MMIO devices owned by this process.",
            coord);
        auto bytes = shard->view_bytes();
        auto existing = std::find_if(sources.begin(), sources.end(), [&](const Source& source) {
            return source.base == bytes.data() && source.size == bytes.size();
        });
        if (existing == sources.end()) {
            sources.push_back(Source{.host_buffer = *shard, .base = bytes.data(), .size = bytes.size()});
            existing = std::prev(sources.end());
        }
        existing->coord_range.merge(distributed::MeshCoordinateRange(coord, coord));
        existing->coords.push_back(coord);
    }
    return sources;
}

// Today's path: each shard pinned whole through the cache (a hit reuses an existing pin), then one blocking write.
bool write_shards_with_cached_pins(
    distributed::MeshCommandQueue& cq,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    std::vector<Source>& sources) {
    auto& mesh_device = *mesh_buffer->device();
    std::vector<distributed::ShardDataTransfer> transfers;
    bool any_pinned = false;
    for (auto& source : sources) {
        HostBuffer pin_buffer(source.host_buffer);
        auto pinned_memory = experimental::PinnedMemoryCache::instance().try_pin(
            mesh_device,
            source.coord_range,
            pin_buffer,
            /*map_to_noc=*/true,
            experimental::PinnedMemoryDeviceAccess::ReadOnly);
        any_pinned = any_pinned || pinned_memory != nullptr;
        for (const auto& coord : source.coords) {
            auto transfer =
                distributed::ShardDataTransfer{coord}.host_data(source.base).region(BufferRegion(0, source.size));
            experimental::ShardDataTransferSetPinnedMemory(transfer, pinned_memory);
            transfers.push_back(std::move(transfer));
        }
    }
    if (!any_pinned) {
        return false;
    }
    cq.enqueue_write_shards(mesh_buffer, transfers, /*blocking=*/true);
    return true;
}

// Whether the chunked pipeline can write every chunk of every source through the pinned branch of
// write_to_device_buffer. It must decide up front: the pinned branch reads a transfer's host_data as the base of the
// whole buffer, the copy branch as the start of the region, so the pipeline cannot find out per chunk.
bool chunked_upload_applies(
    distributed::MeshDevice& mesh_device, const distributed::MeshBuffer& mesh_buffer, std::vector<Source>& sources) {
    auto& metal_env = mesh_device.impl().metal_env();
    const auto& rtoptions = metal_env.get_rtoptions();
    if (rtoptions.get_pinned_upload_threads() == 0 || !rtoptions.get_fast_dispatch() ||
        rtoptions.get_pinned_memory_cache_limit_bytes() == 0) {
        return false;
    }
    // Wormhole reaches host memory through a handful of iATU windows, too few for one pin per chunk.
    if (!metal_env.get_hal().get_supports_64_bit_pcie_addressing()) {
        return false;
    }
    if (!experimental::GetMemoryPinningParameters(mesh_device).supports_read_only) {
        return false;
    }
    if (sources.empty()) {
        return false;
    }
    const size_t page_size = mesh_buffer.page_size();
    for (auto& source : sources) {
        source.chunk_bytes = chunk_bytes_for(source.size, page_size);
        if (source.chunk_bytes == 0 || source.size % page_size != 0 || source.size > mesh_buffer.device_local_size()) {
            return false;
        }
        for (const auto& coord : source.coords) {
            const Buffer* device_buffer = mesh_buffer.get_device_buffer(coord);
            if (!buffer_dispatch::pinned_interleaved_write_layout_supported(*device_buffer) ||
                experimental::per_core_allocation::is_per_core_allocation(*device_buffer)) {
                return false;
            }
            for (size_t offset = 0; offset < source.size; offset += source.chunk_bytes) {
                if (!buffer_dispatch::pinned_write_source_aligned(*device_buffer, source.base + offset)) {
                    return false;
                }
            }
        }
        // A cached pin of this range may still be in use, and its presence means the range is being re-uploaded.
        if (experimental::PinnedMemoryCache::instance().contains(source.base)) {
            return false;
        }
    }
    return true;
}

void write_shards_chunked(
    distributed::MeshCommandQueue& cq,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const std::vector<Source>& sources) {
    auto& mesh_device = *mesh_buffer->device();
    auto& metal_env = mesh_device.impl().metal_env();
    auto& pool = pin_worker_pool(metal_env.get_rtoptions().get_pinned_upload_threads());

    // The driver serializes pins per device handle. Pins for different MMIO devices already go through different
    // handles, so each device needs only enough extra handles for its share of the pool threads.
    std::set<ChipId> mmio_device_ids;
    for (const auto& source : sources) {
        for (const auto& coord : source.coords) {
            mmio_device_ids.insert(
                metal_env.get_cluster().get_associated_mmio_device(mesh_device.impl().get_device(coord)->id()));
        }
    }
    const size_t extra_handles_per_device =
        mmio_device_ids.size() >= pool.num_threads()
            ? 0
            : (pool.num_threads() + mmio_device_ids.size() - 1) / mmio_device_ids.size();
    for (ChipId mmio_device_id : mmio_device_ids) {
        metal_env.get_cluster().set_pin_handle_count(mmio_device_id, extra_handles_per_device);
    }

    size_t num_chunks = 0;
    for (const auto& source : sources) {
        num_chunks = std::max(num_chunks, (source.size + source.chunk_bytes - 1) / source.chunk_bytes);
    }
    const bool device_immutable = std::all_of(sources.begin(), sources.end(), [](const Source& source) {
        return experimental::MemoryPinIsDeviceImmutable(source.host_buffer.pin());
    });
    log_debug(
        tt::LogMetal,
        "Pinned upload: {} source(s) of up to {} chunk(s), {} pin thread(s), {} extra pin handle(s) per MMIO device, "
        "device-immutable: {}",
        sources.size(),
        num_chunks,
        pool.num_threads(),
        extra_handles_per_device,
        device_immutable);

    // pin_futures[chunk][source]; an invalid future means the chunk is past the end of that source.
    std::vector<std::vector<std::future<PinnedMemoryPtr>>> pin_futures(num_chunks);
    size_t num_chunks_submitted = 0;
    const size_t chunks_ahead =
        std::max<size_t>(1, (pool.num_threads() * k_pins_ahead_per_thread + sources.size() - 1) / sources.size());
    auto submit_through = [&](size_t chunk_end) {
        for (; num_chunks_submitted < std::min(chunk_end, num_chunks); num_chunks_submitted++) {
            auto& futures = pin_futures[num_chunks_submitted];
            futures.resize(sources.size());
            // Consecutive tasks pin different sources, which are on different devices unless replicated.
            for (size_t s = 0; s < sources.size(); s++) {
                const auto& source = sources[s];
                const size_t offset = num_chunks_submitted * source.chunk_bytes;
                if (offset >= source.size) {
                    continue;
                }
                const size_t length = std::min(source.chunk_bytes, source.size - offset);
                futures[s] = pool.submit([&mesh_device, &source, offset, length]() {
                    // The chunk HostBuffer only names the range for Create; Source::host_buffer keeps the storage
                    // alive, so the chunk carries no MemoryPin (which would add a release callback per chunk).
                    HostBuffer chunk(ttsl::Span<std::byte>(source.base + offset, length), MemoryPin());
                    auto pinned = experimental::PinnedMemory::Create(
                        mesh_device,
                        source.coord_range,
                        chunk,
                        /*map_to_noc=*/true,
                        experimental::PinnedMemoryDeviceAccess::ReadOnly);
                    experimental::HostBufferSetPinnedMemory(chunk, nullptr);
                    return pinned;
                });
            }
        }
    };
    // Pins still being created reference the sources and the device, so they must finish before this returns,
    // including when it throws.
    auto wait_for_pins = ttsl::make_cleanup([&] {
        for (auto& futures : pin_futures) {
            for (auto& future : futures) {
                if (future.valid()) {
                    future.wait();
                }
            }
        }
    });

    std::deque<std::vector<PinnedMemoryPtr>> in_flight;
    for (size_t c = 0; c < num_chunks; c++) {
        submit_through(c + 1 + chunks_ahead);
        std::vector<PinnedMemoryPtr> chunk_pins;
        std::vector<distributed::ShardDataTransfer> transfers;
        for (size_t s = 0; s < sources.size(); s++) {
            const auto& source = sources[s];
            const size_t offset = c * source.chunk_bytes;
            if (offset >= source.size) {
                continue;
            }
            const size_t length = std::min(source.chunk_bytes, source.size - offset);
            PinnedMemoryPtr pinned;
            try {
                pinned = pin_futures[c][s].get();
            } catch (const std::exception& e) {
                // Pinning can fail when the driver runs out of pin resources. Copy this chunk through the command
                // queue instead; the pinned chunks around it are unaffected.
                static std::once_flag pin_failure_warned;
                std::call_once(pin_failure_warned, [&] {
                    log_warning(
                        tt::LogMetal,
                        "Pinning a {} B chunk of a tensor upload failed; copying that chunk through the command queue "
                        "instead. This message is emitted once per process. Error: {}",
                        length,
                        e.what());
                });
            }
            for (const auto& coord : source.coords) {
                auto transfer = distributed::ShardDataTransfer{coord}.region(BufferRegion(offset, length));
                if (pinned) {
                    const auto pinned_source = buffer_dispatch::resolve_pinned_interleaved_write_source(
                        *mesh_buffer->get_device_buffer(coord), source.base + offset, length, *pinned);
                    TT_FATAL(
                        pinned_source.status == buffer_dispatch::PinnedInterleavedWriteSource::Status::Pinned,
                        "Chunk at offset {} ({} B) of the upload to device coordinate {} cannot be read from its pin "
                        "(status {}); the copy path would read the wrong bytes because it treats host_data as the "
                        "chunk start. chunked_upload_applies must reject this upload.",
                        offset,
                        length,
                        coord,
                        static_cast<int>(pinned_source.status));
                    // The pinned branch of write_to_device_buffer reads host_data + region offset.
                    transfer.host_data(source.base);
                    experimental::ShardDataTransferSetPinnedMemory(transfer, pinned);
                } else {
                    // The copy branch reads the region's bytes starting at host_data.
                    transfer.host_data(source.base + offset);
                }
                transfers.push_back(std::move(transfer));
            }
            if (pinned) {
                chunk_pins.push_back(std::move(pinned));
            }
        }
        cq.enqueue_write_shards(mesh_buffer, transfers, /*blocking=*/false);
        in_flight.push_back(std::move(chunk_pins));
        if (in_flight.size() > k_in_flight_chunks) {
            // Releasing the last reference waits for the chunk's writes, then unpins.
            in_flight.pop_front();
        }
    }

    if (!device_immutable) {
        // The caller may reuse its memory once this returns.
        in_flight.clear();
        return;
    }
    PendingUpload pending{.cq_id = cq.id(), .completion = cq.enqueue_record_event_to_host()};
    for (auto& chunk_pins : in_flight) {
        for (auto& pin : chunk_pins) {
            pending.pins.push_back(std::move(pin));
        }
    }
    for (const auto& source : sources) {
        pending.storage.push_back(source.host_buffer.pin());
    }
    PendingUploads::instance().add(mesh_device, std::move(pending));
}

}  // namespace

size_t chunk_bytes_for(size_t shard_bytes, size_t page_bytes) {
    const size_t granule = std::lcm(page_bytes, os_page_size());
    if (shard_bytes == 0 || granule > k_max_chunk_bytes) {
        return 0;
    }
    const auto round_up = [granule](size_t bytes) { return (bytes + granule - 1) / granule * granule; };
    size_t num_chunks = (shard_bytes + k_max_chunk_bytes - 1) / k_max_chunk_bytes;
    while (round_up((shard_bytes + num_chunks - 1) / num_chunks) > k_max_chunk_bytes) {
        num_chunks++;
    }
    return round_up((shard_bytes + num_chunks - 1) / num_chunks);
}

bool should_use_pinned_write_path(distributed::MeshDevice& mesh_device, size_t size_bytes) {
    if (size_bytes <= k_pin_write_threshold_bytes) {
        return false;
    }
    const auto params = experimental::GetMemoryPinningParameters(mesh_device);
    return params.max_pins > 0 && params.can_map_to_noc;
}

bool write_shards(
    distributed::MeshCommandQueue& cq,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const DistributedHostBuffer& host_buffer) {
    auto& mesh_device = *mesh_buffer->device();
    auto sources = collect_local_sources(mesh_device, host_buffer);
    size_t total_size = 0;
    for (const auto& source : sources) {
        total_size += source.size * source.coords.size();
    }
    if (sources.empty() || !should_use_pinned_write_path(mesh_device, total_size)) {
        return false;
    }

    PendingUploads::instance().release_completed(mesh_device);

    if (chunked_upload_applies(mesh_device, *mesh_buffer, sources)) {
        if (!SeenUploads::instance().check_and_record(sources)) {
            write_shards_chunked(cq, mesh_buffer, sources);
            return true;
        }
        // A pin cache entry must not overlap chunk pins still held by an earlier upload of this range.
        drain(mesh_device);
    }
    return write_shards_with_cached_pins(cq, mesh_buffer, sources);
}

void drain(const distributed::MeshDevice& mesh_device, std::optional<uint32_t> cq_id) {
    PendingUploads::instance().drain(mesh_device, cq_id);
}

size_t num_pending(const distributed::MeshDevice& mesh_device) {
    return PendingUploads::instance().num_pending(mesh_device);
}

void wait_for_storage_release() { PendingUploads::instance().wait_for_storage_release(); }

}  // namespace tt::tt_metal::pinned_upload
