// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// LayerCompletionQueue — POSIX-SHM-backed bounded MPSC ring carrying
// LayerCompletionMessages. Dual role, modelled on
// InterProcessCounterChannel:
//   * create(name)  → owner. Creates /dev/shm/<name>, initialises the
//                     ring, owns its lifetime, unlinks on shutdown.
//   * connect(name) → producer/consumer. Attaches to an owner-created
//                     segment by name (polls until present or timeout).
//
// The ring itself is symmetric (any attached process may push and/or
// pop). In the prefill topology the router owns it and is the sole
// consumer; the prefill runner(s) connect and push.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <internal/disaggregation/layer_completion_message.hpp>

namespace tt::tt_metal::distributed {
class NamedShm;  // fwd — defined in tt_metal/distributed/named_shm.hpp
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::internal {

using tt::tt_metal::distributed::NamedShm;  // tt_metal/distributed/named_shm.hpp

struct LayerCompletionRingHeader;  // fwd — defined in layer_completion_ring_layout.hpp
struct LayerCompletionCell;        // fwd — defined in layer_completion_ring_layout.hpp

class LayerCompletionQueue {
public:
    // Owner: shm_open(O_CREAT|O_EXCL) the segment at /dev/shm/<shm_name>,
    // initialise the ring header + cell sequences, mmap. Throws
    // std::runtime_error if the segment already exists (caller unlinks a
    // stale segment first). shm_name: leading '/', no other slashes.
    static std::unique_ptr<LayerCompletionQueue> create(const std::string& shm_name);

    // Connector: poll for /dev/shm/<shm_name> up to connect_timeout_ms,
    // mmap, validate magic + capacity. Throws on timeout / mismatch.
    static std::unique_ptr<LayerCompletionQueue> connect(
        const std::string& shm_name, uint32_t connect_timeout_ms = 30'000);

    ~LayerCompletionQueue();
    LayerCompletionQueue(const LayerCompletionQueue&) = delete;
    LayerCompletionQueue& operator=(const LayerCompletionQueue&) = delete;
    LayerCompletionQueue(LayerCompletionQueue&&) = delete;
    LayerCompletionQueue& operator=(LayerCompletionQueue&&) = delete;

    // Producer. Returns false (no write) when the ring is full.
    bool try_push(const LayerCompletionMessage& msg);

    // Consumer. Returns false (out untouched) when the ring is empty.
    bool try_pop(LayerCompletionMessage& out);

    // Idempotent. Owner: munmap + shm_unlink. Connector: munmap only.
    void shutdown();

    const std::string& shm_name() const noexcept { return shm_name_; }
    static constexpr uint32_t capacity() noexcept { return kLayerCompletionRingCapacity; }

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
