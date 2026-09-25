// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// LayerCompletionQueueT — POSIX-SHM-backed bounded MPSC ring carrying
// layer-completion messages. Dual role, modelled on
// InterProcessCounterChannel:
//   * create(name)  → owner. Creates /dev/shm/<name>, initialises the
//                     ring, owns its lifetime, unlinks on shutdown.
//   * connect(name) → producer/consumer. Attaches to an owner-created
//                     segment by name (polls until present or timeout).
//
// The ring itself is symmetric (any attached process may push and/or
// pop). In the prefill topology the router owns the host-local ring and
// is the sole consumer; the prefill runner(s) connect and push. The v2
// scheduler-facing ring inverts the roles: the master router owns and
// pushes, the scheduler connects and pops.
//
// Templated on the message version (see layer_completion_message.hpp):
//   LayerCompletionQueue   — v1 (24B messages, magic 'LCQ1'), the frozen
//                            count-protocol format.
//   LayerCompletionQueueV2 — v2 (40B self-describing messages, magic
//                            'LCQ2'), the structured protocol.
// Only these two instantiations exist (extern template below); the magic
// check in connect() rejects a cross-version attach.

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

struct LayerCompletionRingHeader;           // fwd — defined in layer_completion_ring_layout.hpp
template <typename MsgT>
struct LayerCompletionCellT;                // fwd — defined in layer_completion_ring_layout.hpp

// Protocol-agnostic base for the two typed rings below, so an owner that is
// protocol-polymorphic (the router) holds EITHER message version through one
// member. Only the protocol-neutral ops are virtual; typed use is recovered by
// static_cast at sites where the protocol (hence the dynamic type) is known.
class LayerCompletionQueueBase {
public:
    virtual ~LayerCompletionQueueBase() = default;
    virtual void shutdown() = 0;
    virtual const std::string& shm_name() const = 0;
};

template <typename MsgT>
class LayerCompletionQueueT : public LayerCompletionQueueBase {
public:
    // Owner: shm_open(O_CREAT|O_EXCL) the segment at /dev/shm/<shm_name>,
    // initialise the ring header + cell sequences, mmap. Throws
    // std::runtime_error if the segment already exists (caller unlinks a
    // stale segment first). shm_name: leading '/', no other slashes.
    static std::unique_ptr<LayerCompletionQueueT> create(const std::string& shm_name);

    // Connector: poll for /dev/shm/<shm_name> up to connect_timeout_ms,
    // mmap, validate magic + capacity. Throws on timeout / mismatch.
    static std::unique_ptr<LayerCompletionQueueT> connect(
        const std::string& shm_name, uint32_t connect_timeout_ms = 30'000);

    ~LayerCompletionQueueT() override;
    LayerCompletionQueueT(const LayerCompletionQueueT&) = delete;
    LayerCompletionQueueT& operator=(const LayerCompletionQueueT&) = delete;
    LayerCompletionQueueT(LayerCompletionQueueT&&) = delete;
    LayerCompletionQueueT& operator=(LayerCompletionQueueT&&) = delete;

    // Producer. Returns false (no write) when the ring is full.
    bool try_push(const MsgT& msg);

    // Consumer. Returns false (out untouched) when the ring is empty.
    bool try_pop(MsgT& out);

    // Idempotent. Owner: munmap + shm_unlink. Connector: munmap only.
    void shutdown() override;

    const std::string& shm_name() const noexcept override { return shm_name_; }
    static constexpr uint32_t capacity() noexcept { return kLayerCompletionRingCapacity; }

private:
    enum class Role : uint8_t { Owner, Connector };
    LayerCompletionQueueT(std::unique_ptr<NamedShm> shm, std::string shm_name, Role role);

    LayerCompletionRingHeader* header() const noexcept;
    LayerCompletionCellT<MsgT>* cells() const noexcept;

    std::unique_ptr<NamedShm> shm_;
    std::string shm_name_;
    Role role_;
    std::atomic<bool> shutdown_called_{false};
};

using LayerCompletionQueue = LayerCompletionQueueT<LayerCompletionMessage>;      // v1
using LayerCompletionQueueV2 = LayerCompletionQueueT<LayerCompletionMessageV2>;  // v2

// The only two instantiations — defined in layer_completion_queue.cpp.
extern template class LayerCompletionQueueT<LayerCompletionMessage>;
extern template class LayerCompletionQueueT<LayerCompletionMessageV2>;

}  // namespace tt::tt_metal::internal
