// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// LayerCompletionQueue: POSIX-SHM-backed bounded MPSC ring of layer-completion messages. create() owns the
// segment (creates, initialises, unlinks on shutdown); connect() attaches to an existing one. Either side may
// push or pop: the router owns the host-local ring and pops while runners push; the master router owns the
// scheduler-facing ring and pushes while the scheduler pops.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <internal/disaggregation/layer_completion_message.hpp>

namespace tt::tt_metal::distributed {
class NamedShm;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::internal {

using tt::tt_metal::distributed::NamedShm;

struct LayerCompletionRingHeader;
struct LayerCompletionCell;

class LayerCompletionQueue {
public:
    // Owner: shm_open(O_CREAT|O_EXCL), initialise header and cells, mmap. Throws if the segment exists.
    // shm_name: leading '/', no other slashes. completions_per_chunk goes into the header; must be non-zero.
    static std::unique_ptr<LayerCompletionQueue> create(const std::string& shm_name, uint32_t completions_per_chunk);

    // Connector: poll for the segment up to connect_timeout_ms, mmap, validate magic, capacity and
    // completions_per_chunk. Throws on timeout or mismatch.
    static std::unique_ptr<LayerCompletionQueue> connect(
        const std::string& shm_name, uint32_t connect_timeout_ms = 30'000);

    ~LayerCompletionQueue();
    LayerCompletionQueue(const LayerCompletionQueue&) = delete;
    LayerCompletionQueue& operator=(const LayerCompletionQueue&) = delete;
    LayerCompletionQueue(LayerCompletionQueue&&) = delete;
    LayerCompletionQueue& operator=(LayerCompletionQueue&&) = delete;

    // False when the ring is full.
    bool try_push(const LayerCompletionMessage& msg);

    // False (out untouched) when the ring is empty.
    bool try_pop(LayerCompletionMessage& out);

    // Idempotent. The owner also unlinks the segment.
    void shutdown();

    const std::string& shm_name() const noexcept { return shm_name_; }
    static constexpr uint32_t capacity() noexcept { return kLayerCompletionRingCapacity; }
    uint32_t completions_per_chunk() const noexcept;

private:
    enum class Role : uint8_t { Owner, Connector };
    LayerCompletionQueue(std::unique_ptr<NamedShm> shm, std::string shm_name, Role role);

    LayerCompletionRingHeader* header() const noexcept;
    LayerCompletionCell* cells() const noexcept;

    std::unique_ptr<NamedShm> shm_;
    std::string shm_name_;
    Role role_;
    std::atomic<bool> shutdown_called_{false};
};

}  // namespace tt::tt_metal::internal
