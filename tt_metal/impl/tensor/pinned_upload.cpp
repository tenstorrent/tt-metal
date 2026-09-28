// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pinned_upload.hpp"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/memory_pin_access.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/cleanup.hpp>
#include <tt_stl/indestructible.hpp>
#include <tt_stl/span.hpp>

#include "common/memory_pin_impl.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_event_impl.hpp"
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
    size_t chunk_bytes = 0;  // Size of this source's chunks in a chunked upload; see chunk_bytes_for.
    distributed::MeshCoordinateRangeSet coord_range;
    std::vector<distributed::MeshCoordinate> coords;

    size_t num_chunks() const { return tt::div_up(size, chunk_bytes); }

    // The byte range of chunk `c`, the last one possibly short; nullopt past the end of the source.
    std::optional<BufferRegion> chunk(size_t c) const {
        const size_t offset = c * chunk_bytes;
        if (offset >= size) {
            return std::nullopt;
        }
        return BufferRegion(offset, std::min(chunk_bytes, size - offset));
    }
};

// Threads that run submitted tasks in submission order. Destroying the queue runs the tasks already submitted, then
// joins the threads.
class TaskQueue {
public:
    explicit TaskQueue(size_t num_threads) {
        threads_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; i++) {
            threads_.emplace_back([this] { run(); });
        }
    }

    ~TaskQueue() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& thread : threads_) {
            thread.join();
        }
    }

    TaskQueue(const TaskQueue&) = delete;
    TaskQueue& operator=(const TaskQueue&) = delete;

    size_t num_threads() const { return threads_.size(); }

    template <typename Fn>
    std::future<std::invoke_result_t<Fn>> submit(Fn&& fn) {
        std::packaged_task<std::invoke_result_t<Fn>()> task(std::forward<Fn>(fn));
        auto future = task.get_future();
        {
            std::lock_guard lock(mutex_);
            tasks_.emplace_back(std::move(task));
        }
        cv_.notify_one();
        return future;
    }

private:
    void run() {
        while (true) {
            std::packaged_task<void()> task;
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
    std::deque<std::packaged_task<void()>> tasks_;  // Each wraps the task whose future submit() returned.
    bool stop_ = false;
    std::vector<std::thread> threads_;
};

// Threads that run chunk pins. Separate from the command queues' dispatch pools, whose workers are pinned to devices
// and whose wait() covers the whole pool. Uploads configured with the same thread count share a pool. An upload
// configured with a different count (a changed TT_METAL_PINNED_UPLOAD_THREADS, or another MetalEnv) replaces it; the
// old pool exits once the uploads using it finish.
std::shared_ptr<TaskQueue> pin_worker_pool(size_t num_threads) {
    static std::mutex mutex;
    static std::shared_ptr<TaskQueue> pool;
    std::lock_guard lock(mutex);
    if (pool == nullptr || pool->num_threads() != num_threads) {
        pool = std::make_shared<TaskQueue>(num_threads);
    }
    return pool;
}

// Chunk pins created so far; see num_chunk_pins_created.
std::atomic<size_t> chunk_pins_created{0};

// Falls back to 4 KiB, as PinnedMemory does, if sysconf cannot report it.
size_t os_page_size() {
    static const size_t page_size = [] {
        const long reported = sysconf(_SC_PAGESIZE);
        return reported > 0 ? static_cast<size_t>(reported) : size_t{4096};
    }();
    return page_size;
}

// A host range widened to whole OS pages, which is what a pin of the range covers.
struct PinnedSpan {
    uintptr_t begin = 0;
    uintptr_t end = 0;

    PinnedSpan(const std::byte* base, size_t size) :
        begin(tt::round_down(reinterpret_cast<uintptr_t>(base), os_page_size())),
        end(tt::round_up(reinterpret_cast<uintptr_t>(base) + size, os_page_size())) {}

    bool overlaps(const PinnedSpan& other) const { return begin < other.end && other.begin < end; }
};

// Pins of device-immutable uploads that returned before their writes completed.
struct PendingUpload {
    distributed::MeshEvent completion;  // Recorded after the upload's last write.
    std::vector<PinnedMemoryPtr> pins;
    std::vector<PinnedSpan> spans;   // The host ranges of the upload's sources.
    std::vector<MemoryPin> storage;  // Keeps the uploaded memory alive until the pins are released.

    const distributed::MeshDevice* device() const { return completion.device(); }
    uint32_t cq_id() const { return completion.impl().mesh_cq_id(); }
};

// Host ranges already uploaded once through chunk pins. A second upload of the same range while its storage is alive
// pins it whole through PinnedMemoryCache instead, so a buffer uploaded repeatedly is pinned once. A range is
// forgotten when its storage is released, because a new mapping of the same size often lands at the same address.
class SeenUploads {
public:
    // Never destroyed: storage released at any point during exit may still call forget().
    static SeenUploads& instance() {
        static ttsl::Indestructible<SeenUploads> seen;
        return seen.get();
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
            // A key pushed out of keys_ keeps the callback it registered; registering another each time the key is
            // recorded again would grow the storage's callback list by one per upload.
            if (!keys_with_release_callback_.contains(key)) {
                MemoryPin pin = source.host_buffer.pin();
                if (pin == nullptr) {
                    continue;
                }
                pin.impl().add_final_release_callback([key] { SeenUploads::instance().forget(key); });
                keys_with_release_callback_.insert(key);
            }
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
        keys_with_release_callback_.erase(key);
    }

    std::mutex mutex_;
    std::deque<Key> keys_;
    // Keys whose storage calls forget() when released, including keys since pushed out of keys_.
    std::set<Key> keys_with_release_callback_;
};

// Drops storage references on a background thread. Unmapping a file mapping costs about as much as transferring it
// (page table teardown for every page the pins faulted in), so it stays off the uploading thread.
class StorageReleaser {
public:
    void release(std::vector<MemoryPin> storage) {
        if (!storage.empty()) {
            queue().submit([storage = std::move(storage)]() mutable { storage.clear(); });
        }
    }

    // Waits until the storage of every earlier release() has been dropped.
    void wait() {
        queue().submit([] {}).wait();
    }

private:
    // Started on first use: every process whose command queues finish() constructs this.
    TaskQueue& queue() {
        std::call_once(started_, [this] { queue_.emplace(1); });
        return *queue_;
    }

    std::once_flag started_;
    std::optional<TaskQueue> queue_;
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

    void add(PendingUpload upload) {
        std::lock_guard lock(mutex_);
        uploads_.push_back(std::move(upload));
    }

    // Releases uploads to `mesh_device` whose writes have completed, without waiting.
    void release_completed(const distributed::MeshDevice& mesh_device) {
        // Events on one command queue complete in order, so stop at the first incomplete upload per queue.
        std::set<uint32_t> blocked_cqs;
        release(take([&](const PendingUpload& upload) {
            if (upload.device() != &mesh_device || blocked_cqs.contains(upload.cq_id())) {
                return false;
            }
            if (distributed::EventQuery(upload.completion)) {
                return true;
            }
            blocked_cqs.insert(upload.cq_id());
            return false;
        }));
    }

    void drain(const distributed::MeshDevice& mesh_device, std::optional<uint32_t> cq_id) {
        release(take([&](const PendingUpload& upload) {
            return upload.device() == &mesh_device && (!cq_id.has_value() || upload.cq_id() == *cq_id);
        }));
    }

    // Waits for, then releases, the pending uploads (to any device, through any command queue) whose sources overlap
    // the host ranges of `sources`.
    void drain_overlapping(const std::vector<Source>& sources) {
        std::vector<PinnedSpan> spans;
        spans.reserve(sources.size());
        for (const auto& source : sources) {
            spans.emplace_back(source.base, source.size);
        }
        release(take([&](const PendingUpload& upload) {
            return std::any_of(upload.spans.begin(), upload.spans.end(), [&](const auto& held) {
                return std::any_of(spans.begin(), spans.end(), [&](const auto& span) { return held.overlaps(span); });
            });
        }));
    }

    size_t num_pending(const distributed::MeshDevice& mesh_device) const {
        std::lock_guard lock(mutex_);
        return static_cast<size_t>(std::count_if(uploads_.begin(), uploads_.end(), [&](const PendingUpload& upload) {
            return upload.device() == &mesh_device;
        }));
    }

    void wait_for_storage_release() { storage_releaser_.wait(); }

private:
    // Removes and returns the uploads for which `pred` holds; `pred` sees them in the order they were added.
    template <typename Pred>
    std::vector<PendingUpload> take(Pred&& pred) {
        std::vector<PendingUpload> taken;
        std::lock_guard lock(mutex_);
        for (auto upload = uploads_.begin(); upload != uploads_.end();) {
            if (pred(*upload)) {
                taken.push_back(std::move(*upload));
                upload = uploads_.erase(upload);
            } else {
                ++upload;
            }
        }
        return taken;
    }

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
    std::deque<PendingUpload> uploads_;  // In the order they were added.
    StorageReleaser storage_releaser_;
};

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
            sources.push_back(Source{.host_buffer = std::move(*shard), .base = bytes.data(), .size = bytes.size()});
            existing = std::prev(sources.end());
        }
        existing->coord_range.merge(distributed::MeshCoordinateRange(coord, coord));
        existing->coords.push_back(coord);
    }
    return sources;
}

// The one source of a host buffer replicated to every device of the mesh that this host owns; none if it owns no
// device.
std::vector<Source> replicated_local_sources(
    const distributed::MeshDevice& mesh_device, const HostBuffer& host_buffer) {
    const auto& view = mesh_device.get_view();
    Source source{.host_buffer = host_buffer};
    auto bytes = source.host_buffer.view_bytes();
    source.base = bytes.data();
    source.size = bytes.size();
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device.shape())) {
        // Only chips owned by this host can be pinned or written.
        if (view.impl().is_local(coord)) {
            source.coord_range.merge(distributed::MeshCoordinateRange(coord, coord));
            source.coords.push_back(coord);
        }
    }
    std::vector<Source> sources;
    if (!source.coords.empty()) {
        sources.push_back(std::move(source));
    }
    return sources;
}

// Appends the writes of `region` of `source` to each of its devices, read from `pinned` when set and copied through the
// command queue otherwise. Either way host_data is the region's first byte.
void append_transfers(
    const Source& source,
    const BufferRegion& region,
    const PinnedMemoryPtr& pinned,
    std::vector<distributed::ShardDataTransfer>& transfers) {
    for (const auto& coord : source.coords) {
        auto transfer = distributed::ShardDataTransfer{coord}.host_data(source.base + region.offset).region(region);
        if (pinned) {
            experimental::ShardDataTransferSetPinnedMemory(transfer, pinned);
        }
        transfers.push_back(std::move(transfer));
    }
}

// Whole-shard path: each shard pinned whole through the cache (a hit reuses an existing pin), then one blocking write.
bool write_shards_with_cached_pins(
    distributed::MeshCommandQueue& cq,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const std::vector<Source>& sources) {
    auto& mesh_device = *mesh_buffer->device();
    std::vector<distributed::ShardDataTransfer> transfers;
    bool any_pinned = false;
    for (const auto& source : sources) {
        HostBuffer pin_buffer(source.host_buffer);
        auto pinned_memory = experimental::PinnedMemoryCache::instance().try_pin(
            mesh_device,
            source.coord_range,
            pin_buffer,
            /*map_to_noc=*/true,
            experimental::PinnedMemoryDeviceAccess::ReadOnly);
        any_pinned = any_pinned || pinned_memory != nullptr;
        append_transfers(source, BufferRegion(0, source.size), pinned_memory, transfers);
    }
    if (!any_pinned) {
        return false;
    }
    cq.enqueue_write_shards(mesh_buffer, transfers, /*blocking=*/true);
    return true;
}

// Whether the chunks of every source can go through the pinned branch of write_to_device_buffer. A chunk that cannot
// is copied instead, which makes pinning it wasted work, so an upload that would copy its chunks is not chunked.
bool chunked_upload_applies(
    distributed::MeshDevice& mesh_device,
    const distributed::MeshBuffer& mesh_buffer,
    const std::vector<Source>& sources) {
    if (!chunked_uploads_supported(mesh_device)) {
        return false;
    }
    const size_t page_size = mesh_buffer.page_size();
    for (const auto& source : sources) {
        if (source.chunk_bytes == 0 || source.size % page_size != 0 || source.size > mesh_buffer.device_local_size()) {
            return false;
        }
        for (const auto& coord : source.coords) {
            const Buffer* device_buffer = mesh_buffer.get_device_buffer(coord);
            if (!buffer_dispatch::pinned_interleaved_write_layout_supported(*device_buffer) ||
                experimental::per_core_allocation::is_per_core_allocation(*device_buffer)) {
                return false;
            }
        }
        // Chunk offsets are multiples of the OS page, which meets any L1 read alignment, so every chunk is as aligned
        // as the base.
        if (!buffer_dispatch::pinned_write_source_aligned(
                *mesh_buffer.get_device_buffer(source.coords.front()), source.base)) {
            return false;
        }
        // A cached pin of this range may still be in use, and its presence means the range is being re-uploaded.
        if (experimental::PinnedMemoryCache::instance().contains(source.base)) {
            return false;
        }
    }
    return true;
}

// Waits until the writes enqueued before `written`, on its command queue, have completed. A queue that stops on a
// device error never completes the event, so this throws then instead of waiting forever; the device error itself
// surfaces from the queue's next finish().
void wait_for_writes(const distributed::MeshEvent& written) {
    if (!written.device()->impl().wait_for_event_unless_queue_failed(written)) {
        TT_THROW(
            "A tensor upload through command queue {} did not complete: the queue stopped after a device error.",
            written.impl().mesh_cq_id());
    }
}

void write_shards_chunked(
    distributed::MeshCommandQueue& cq,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const std::vector<Source>& sources) {
    auto& mesh_device = *mesh_buffer->device();
    auto& metal_env = mesh_device.impl().metal_env();
    // Held for the whole upload: another upload may replace the shared pool meanwhile.
    const auto shared_pool = pin_worker_pool(metal_env.get_rtoptions().get_pinned_upload_threads());
    auto& pool = *shared_pool;

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
        mmio_device_ids.size() >= pool.num_threads() ? 0 : tt::div_up(pool.num_threads(), mmio_device_ids.size());
    for (ChipId mmio_device_id : mmio_device_ids) {
        try {
            metal_env.get_cluster().set_pin_handle_count(mmio_device_id, extra_handles_per_device);
        } catch (const std::exception& e) {
            // Opening a handle fails when the process is out of file descriptors, for example. Pins then share the
            // handles already open, and run less concurrently.
            static std::once_flag handle_failure_warned;
            std::call_once(handle_failure_warned, [&] {
                log_warning(
                    tt::LogMetal,
                    "Opening {} extra device handle(s) for pinning tensor uploads to device {} failed; pinning through "
                    "the handles already open. This message is emitted once per process. Error: {}",
                    extra_handles_per_device,
                    mmio_device_id,
                    e.what());
            });
        }
    }

    size_t num_chunks = 0;
    for (const auto& source : sources) {
        num_chunks = std::max(num_chunks, source.num_chunks());
    }
    // One copy of each source's pin: it carries the device-immutable mark, and a pending upload keeps it.
    std::vector<MemoryPin> storage;
    storage.reserve(sources.size());
    for (const auto& source : sources) {
        storage.push_back(source.host_buffer.pin());
    }
    const bool device_immutable = std::all_of(storage.begin(), storage.end(), [](const MemoryPin& pin) {
        return experimental::MemoryPinIsDeviceImmutable(pin);
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

    // pin_futures[chunk * sources.size() + source]; an invalid future means the chunk is past the end of that source.
    std::vector<std::future<PinnedMemoryPtr>> pin_futures(num_chunks * sources.size());
    size_t num_chunks_submitted = 0;
    const size_t chunks_ahead =
        std::max<size_t>(1, tt::div_up(pool.num_threads() * k_pins_ahead_per_thread, sources.size()));
    auto submit_through = [&](size_t chunk_end) {
        for (; num_chunks_submitted < std::min(chunk_end, num_chunks); num_chunks_submitted++) {
            // Consecutive tasks pin different sources, which are on different devices unless replicated.
            for (size_t s = 0; s < sources.size(); s++) {
                const auto& source = sources[s];
                const auto region = source.chunk(num_chunks_submitted);
                if (!region.has_value()) {
                    continue;
                }
                pin_futures[num_chunks_submitted * sources.size() + s] =
                    pool.submit([&mesh_device, &source, bytes = *region]() {
                        // The chunk HostBuffer only names the range for Create; Source::host_buffer keeps the storage
                        // alive, so the chunk carries no MemoryPin (which would add a release callback per chunk).
                        HostBuffer chunk(ttsl::Span<std::byte>(source.base + bytes.offset, bytes.size), MemoryPin());
                        auto pinned = experimental::PinnedMemory::Create(
                            mesh_device,
                            source.coord_range,
                            chunk,
                            /*map_to_noc=*/true,
                            experimental::PinnedMemoryDeviceAccess::ReadOnly);
                        experimental::HostBufferSetPinnedMemory(chunk, nullptr);
                        chunk_pins_created.fetch_add(1, std::memory_order_relaxed);
                        return pinned;
                    });
            }
        }
    };
    // Pins still being created reference the sources and the device, so they must finish before this returns,
    // including when it throws.
    auto wait_for_pins = ttsl::make_cleanup([&] {
        for (auto& future : pin_futures) {
            if (future.valid()) {
                future.wait();
            }
        }
    });

    struct InFlightChunk {
        distributed::MeshEvent written;  // Recorded after the chunk's writes.
        std::vector<PinnedMemoryPtr> pins;
    };
    std::deque<InFlightChunk> in_flight;
    std::vector<distributed::ShardDataTransfer> transfers;
    for (size_t c = 0; c < num_chunks; c++) {
        submit_through(c + 1 + chunks_ahead);
        std::vector<PinnedMemoryPtr> chunk_pins;
        for (size_t s = 0; s < sources.size(); s++) {
            const auto& source = sources[s];
            const auto region = source.chunk(c);
            if (!region.has_value()) {
                continue;
            }
            PinnedMemoryPtr pinned;
            try {
                pinned = pin_futures[c * sources.size() + s].get();
            } catch (const std::exception& e) {
                // Pinning can fail when the driver runs out of pin resources. Copy this chunk through the command
                // queue instead; the pinned chunks around it are unaffected.
                static std::once_flag pin_failure_warned;
                std::call_once(pin_failure_warned, [&] {
                    log_warning(
                        tt::LogMetal,
                        "Pinning a {} B chunk of a tensor upload failed; copying that chunk through the command queue "
                        "instead. This message is emitted once per process. Error: {}",
                        region->size,
                        e.what());
                });
            }
            append_transfers(source, *region, pinned, transfers);
            if (pinned) {
                chunk_pins.push_back(std::move(pinned));
            }
        }
        cq.enqueue_write_shards(mesh_buffer, transfers, /*blocking=*/false);
        transfers.clear();  // Its pin references go too; in_flight holds the chunk's pins.
        in_flight.push_back({.written = cq.enqueue_record_event_to_host(), .pins = std::move(chunk_pins)});
        if (in_flight.size() > k_in_flight_chunks) {
            wait_for_writes(in_flight.front().written);
            in_flight.pop_front();
        }
    }

    if (!device_immutable) {
        // The caller may reuse its memory once this returns. Events on one queue complete in order.
        wait_for_writes(in_flight.back().written);
        return;
    }
    PendingUpload pending{.completion = in_flight.back().written, .storage = std::move(storage)};
    for (auto& chunk : in_flight) {
        for (auto& pin : chunk.pins) {
            pending.pins.push_back(std::move(pin));
        }
    }
    for (const auto& source : sources) {
        pending.spans.emplace_back(source.base, source.size);
    }
    PendingUploads::instance().add(std::move(pending));
}

bool should_use_pinned_write_path(distributed::MeshDevice& mesh_device, size_t size_bytes) {
    if (size_bytes <= k_pin_write_threshold_bytes) {
        return false;
    }
    const auto params = experimental::GetMemoryPinningParameters(mesh_device);
    return params.max_pins > 0 && params.can_map_to_noc;
}

// Writes `sources`, which together pass should_use_pinned_write_path, from pinned host memory.
bool write_sources(
    distributed::MeshCommandQueue& cq,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    std::vector<Source> sources) {
    auto& mesh_device = *mesh_buffer->device();
    PendingUploads::instance().release_completed(mesh_device);
    // A pin of a range that an earlier upload's chunk pins still cover may duplicate one of them exactly, which fails
    // the pin (and a failing cached whole-range pin first evicts the cache's other entries for the device). An upload
    // of the range still enqueueing on another thread is not pending yet; a duplicate there fails the same way and is
    // copied.
    PendingUploads::instance().drain_overlapping(sources);

    const size_t page_size = mesh_buffer->page_size();
    for (auto& source : sources) {
        source.chunk_bytes = chunk_bytes_for(source.size, page_size);
    }
    if (chunked_upload_applies(mesh_device, *mesh_buffer, sources) &&
        !SeenUploads::instance().check_and_record(sources)) {
        write_shards_chunked(cq, mesh_buffer, sources);
        return true;
    }
    return write_shards_with_cached_pins(cq, mesh_buffer, sources);
}

}  // namespace

size_t chunk_bytes_for(size_t shard_bytes, size_t page_bytes) {
    const size_t granule = std::lcm(page_bytes, os_page_size());
    if (shard_bytes == 0 || granule > k_max_chunk_bytes) {
        return 0;
    }
    size_t num_chunks = tt::div_up(shard_bytes, k_max_chunk_bytes);
    while (tt::round_up(tt::div_up(shard_bytes, num_chunks), granule) > k_max_chunk_bytes) {
        num_chunks++;
    }
    return tt::round_up(tt::div_up(shard_bytes, num_chunks), granule);
}

bool chunked_uploads_supported(distributed::MeshDevice& mesh_device) {
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
    const auto params = experimental::GetMemoryPinningParameters(mesh_device);
    return params.max_pins > 0 && params.can_map_to_noc && params.supports_read_only;
}

bool write_shards(
    distributed::MeshCommandQueue& cq,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const DistributedHostBuffer& host_buffer) {
    auto& mesh_device = *mesh_buffer->device();
    // Summed before the sources are collected, which an upload too small to pin would build only to discard.
    size_t total_size = 0;
    host_buffer.apply([&total_size](const HostBuffer& shard) { total_size += shard.view_bytes().size(); });
    if (!should_use_pinned_write_path(mesh_device, total_size)) {
        return false;
    }
    return write_sources(cq, mesh_buffer, collect_local_sources(mesh_device, host_buffer));
}

bool write_replicated(
    distributed::MeshCommandQueue& cq,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const HostBuffer& host_buffer) {
    auto& mesh_device = *mesh_buffer->device();
    auto sources = replicated_local_sources(mesh_device, host_buffer);
    if (sources.empty() ||
        !should_use_pinned_write_path(mesh_device, sources.front().size * sources.front().coords.size())) {
        return false;
    }
    return write_sources(cq, mesh_buffer, std::move(sources));
}

size_t num_chunk_pins_created() { return chunk_pins_created.load(std::memory_order_relaxed); }

void drain(const distributed::MeshDevice& mesh_device, std::optional<uint32_t> cq_id) {
    PendingUploads::instance().drain(mesh_device, cq_id);
}

size_t num_pending(const distributed::MeshDevice& mesh_device) {
    return PendingUploads::instance().num_pending(mesh_device);
}

void wait_for_storage_release() { PendingUploads::instance().wait_for_storage_release(); }

}  // namespace tt::tt_metal::pinned_upload
