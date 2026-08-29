// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/layer_completion_queue.hpp>

#include <chrono>
#include <cstring>
#include <new>
#include <stdexcept>
#include <thread>

#include <fmt/format.h>
#include <sys/stat.h>

#include "layer_completion_ring_layout.hpp"
#include "tt_metal/distributed/named_shm.hpp"

namespace tt::tt_metal::internal {

using tt::tt_metal::distributed::NamedShm;

namespace {
bool shm_path_exists(const std::string& shm_name) {
    struct stat st{};
    return ::stat(("/dev/shm" + shm_name).c_str(), &st) == 0;
}

// The cells region must land cache-line-aligned in every attached process.
// mmap is page-aligned so this cannot fail today; the check converts
// "impossible" into "loudly impossible" if the mapping behaviour or the
// layout math ever changes (v2 cells are one cache line each).
template <typename MsgT>
void check_mapping_alignment(const void* base, const std::string& shm_name, const char* op) {
    const auto addr = reinterpret_cast<std::uintptr_t>(base);
    if (addr % alignof(LayerCompletionCellT<MsgT>) != 0 ||
        (addr + layer_completion_cells_offset<MsgT>()) % alignof(LayerCompletionCellT<MsgT>) != 0) {
        throw std::runtime_error(fmt::format(
            "LayerCompletionQueue::{}: {} mapped at {:#x}, not aligned to {} as the ring layout requires",
            op,
            shm_name,
            addr,
            alignof(LayerCompletionCellT<MsgT>)));
    }
}
}  // namespace

template <typename MsgT>
LayerCompletionQueueT<MsgT>::LayerCompletionQueueT(std::unique_ptr<NamedShm> shm, std::string shm_name, Role role) :
    shm_(std::move(shm)), shm_name_(std::move(shm_name)), role_(role) {}

template <typename MsgT>
LayerCompletionQueueT<MsgT>::~LayerCompletionQueueT() {
    shutdown();
}

template <typename MsgT>
LayerCompletionRingHeader* LayerCompletionQueueT<MsgT>::header() const noexcept {
    return static_cast<LayerCompletionRingHeader*>(shm_->ptr());
}

template <typename MsgT>
LayerCompletionCellT<MsgT>* LayerCompletionQueueT<MsgT>::cells() const noexcept {
    return reinterpret_cast<LayerCompletionCellT<MsgT>*>(
        static_cast<std::byte*>(shm_->ptr()) + layer_completion_cells_offset<MsgT>());
}

template <typename MsgT>
std::unique_ptr<LayerCompletionQueueT<MsgT>> LayerCompletionQueueT<MsgT>::create(const std::string& shm_name) {
    auto shm = std::make_unique<NamedShm>(NamedShm::create(shm_name, kLayerCompletionRingBytes<MsgT>));
    check_mapping_alignment<MsgT>(shm->ptr(), shm_name, "create");
    auto* base = static_cast<std::byte*>(shm->ptr());

    // NamedShm zero-inits the region. Placement-construct the atomics so
    // their lifetime is well-defined, then publish `magic` last.
    auto* hdr = reinterpret_cast<LayerCompletionRingHeader*>(base);
    new (&hdr->enqueue_pos) std::atomic<uint64_t>(0);
    new (&hdr->dequeue_pos) std::atomic<uint64_t>(0);
    hdr->capacity = kLayerCompletionRingCapacity;
    auto* cell_arr = reinterpret_cast<LayerCompletionCellT<MsgT>*>(base + layer_completion_cells_offset<MsgT>());
    for (uint32_t i = 0; i < kLayerCompletionRingCapacity; ++i) {
        new (&cell_arr[i].sequence) std::atomic<uint64_t>(i);
    }
    hdr->magic = LayerCompletionRingTraits<MsgT>::magic;

    return std::unique_ptr<LayerCompletionQueueT>(new LayerCompletionQueueT(std::move(shm), shm_name, Role::Owner));
}

template <typename MsgT>
std::unique_ptr<LayerCompletionQueueT<MsgT>> LayerCompletionQueueT<MsgT>::connect(
    const std::string& shm_name, uint32_t connect_timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(connect_timeout_ms);
    while (!shm_path_exists(shm_name)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error(fmt::format(
                "LayerCompletionQueue::connect timed out waiting for {} after {} ms", shm_name, connect_timeout_ms));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    auto shm = std::make_unique<NamedShm>(NamedShm::open(shm_name, kLayerCompletionRingBytes<MsgT>));
    check_mapping_alignment<MsgT>(shm->ptr(), shm_name, "connect");
    auto* hdr = static_cast<LayerCompletionRingHeader*>(shm->ptr());
    if (hdr->magic != LayerCompletionRingTraits<MsgT>::magic || hdr->capacity != kLayerCompletionRingCapacity) {
        throw std::runtime_error(fmt::format(
            "LayerCompletionQueue::connect: {} is not a valid ring for this protocol version "
            "(magic={:#x} capacity={})",
            shm_name,
            hdr->magic,
            hdr->capacity));
    }
    return std::unique_ptr<LayerCompletionQueueT>(
        new LayerCompletionQueueT(std::move(shm), shm_name, Role::Connector));
}

template <typename MsgT>
bool LayerCompletionQueueT<MsgT>::try_push(const MsgT& msg) {
    auto* hdr = header();
    auto* cell_arr = cells();
    uint64_t pos = hdr->enqueue_pos.load(std::memory_order_relaxed);
    for (;;) {
        LayerCompletionCellT<MsgT>& cell = cell_arr[pos & kLayerCompletionRingMask];
        const uint64_t seq = cell.sequence.load(std::memory_order_acquire);
        const int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos);
        if (diff == 0) {
            if (hdr->enqueue_pos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                cell.msg = msg;
                cell.sequence.store(pos + 1, std::memory_order_release);
                return true;
            }
        } else if (diff < 0) {
            return false;  // full
        } else {
            pos = hdr->enqueue_pos.load(std::memory_order_relaxed);
        }
    }
}

template <typename MsgT>
bool LayerCompletionQueueT<MsgT>::try_pop(MsgT& out) {
    auto* hdr = header();
    auto* cell_arr = cells();
    uint64_t pos = hdr->dequeue_pos.load(std::memory_order_relaxed);
    for (;;) {
        LayerCompletionCellT<MsgT>& cell = cell_arr[pos & kLayerCompletionRingMask];
        const uint64_t seq = cell.sequence.load(std::memory_order_acquire);
        const int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos + 1);
        if (diff == 0) {
            if (hdr->dequeue_pos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                out = cell.msg;
                cell.sequence.store(pos + kLayerCompletionRingMask + 1, std::memory_order_release);
                return true;
            }
        } else if (diff < 0) {
            return false;  // empty
        } else {
            pos = hdr->dequeue_pos.load(std::memory_order_relaxed);
        }
    }
}

template <typename MsgT>
void LayerCompletionQueueT<MsgT>::shutdown() {
    if (shutdown_called_.exchange(true)) {
        return;
    }
    if (!shm_) {
        return;
    }
    if (role_ == Role::Owner) {
        shm_->unlink();
    }
    shm_->close();
}

// The only two instantiations (extern template declarations in the public header).
template class LayerCompletionQueueT<LayerCompletionMessage>;
template class LayerCompletionQueueT<LayerCompletionMessageV2>;

}  // namespace tt::tt_metal::internal
