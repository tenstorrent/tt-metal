// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/layer_completion_queue.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <new>
#include <stdexcept>
#include <thread>

#include <fmt/format.h>
#include <sys/stat.h>

#include "layer_completion_ring_layout.hpp"
#include <tt-metalium/experimental/sockets/named_shm.hpp>

namespace tt::tt_metal::internal {

using tt::tt_metal::distributed::NamedShm;

namespace {
constexpr std::chrono::milliseconds kConnectInitGrace{2000};

bool shm_path_exists(const std::string& shm_name) {
    struct stat st{};
    return ::stat(("/dev/shm" + shm_name).c_str(), &st) == 0;
}

}  // namespace

LayerCompletionQueue::LayerCompletionQueue(std::unique_ptr<NamedShm> shm, std::string shm_name, Role role) :
    shm_(std::move(shm)), shm_name_(std::move(shm_name)), role_(role) {}

LayerCompletionQueue::~LayerCompletionQueue() { shutdown(); }

LayerCompletionRingHeader* LayerCompletionQueue::header() const noexcept {
    return static_cast<LayerCompletionRingHeader*>(shm_->ptr());
}

LayerCompletionCell* LayerCompletionQueue::cells() const noexcept {
    return reinterpret_cast<LayerCompletionCell*>(static_cast<std::byte*>(shm_->ptr()) + kLayerCompletionCellsOffset);
}

std::unique_ptr<LayerCompletionQueue> LayerCompletionQueue::create(
    const std::string& shm_name, uint32_t completions_per_chunk) {
    if (completions_per_chunk == 0) {
        throw std::invalid_argument(
            fmt::format("LayerCompletionQueue::create: {} needs a non-zero completions_per_chunk", shm_name));
    }
    auto shm = std::make_unique<NamedShm>(NamedShm::create(shm_name, kLayerCompletionRingBytes));
    auto* base = static_cast<std::byte*>(shm->ptr());

    // Placement-construct the atomics; publish `magic` last.
    auto* hdr = reinterpret_cast<LayerCompletionRingHeader*>(base);
    new (&hdr->enqueue_pos) std::atomic<uint64_t>(0);
    new (&hdr->dequeue_pos) std::atomic<uint64_t>(0);
    hdr->capacity = kLayerCompletionRingCapacity;
    hdr->completions_per_chunk = completions_per_chunk;
    auto* cell_arr = reinterpret_cast<LayerCompletionCell*>(base + kLayerCompletionCellsOffset);
    for (uint32_t i = 0; i < kLayerCompletionRingCapacity; ++i) {
        new (&cell_arr[i].sequence) std::atomic<uint64_t>(i);
    }
    hdr->magic = kLayerCompletionRingMagic;

    return std::unique_ptr<LayerCompletionQueue>(new LayerCompletionQueue(std::move(shm), shm_name, Role::Owner));
}

std::unique_ptr<LayerCompletionQueue> LayerCompletionQueue::connect(
    const std::string& shm_name, uint32_t connect_timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(connect_timeout_ms);
    while (!shm_path_exists(shm_name)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error(fmt::format(
                "LayerCompletionQueue::connect timed out waiting for {} after {} ms", shm_name, connect_timeout_ms));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // The owner writes `magic` last, so a connector can see a short or unmagicked segment for microseconds.
    // Past the grace it is some other segment at this name: fail now, not after the whole budget.
    const auto init_grace = std::min(deadline, std::chrono::steady_clock::now() + kConnectInitGrace);
    for (;;) {
        std::unique_ptr<NamedShm> shm;
        try {
            shm = std::make_unique<NamedShm>(NamedShm::open(shm_name, kLayerCompletionRingBytes));
        } catch (const std::exception& e) {
            if (std::chrono::steady_clock::now() >= init_grace) {
                throw std::runtime_error(fmt::format(
                    "LayerCompletionQueue::connect: {} is not a valid layer-completion ring ({})", shm_name, e.what()));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        auto* hdr = static_cast<LayerCompletionRingHeader*>(shm->ptr());
        if (hdr->magic == 0 && std::chrono::steady_clock::now() < init_grace) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        if (hdr->magic != kLayerCompletionRingMagic || hdr->capacity != kLayerCompletionRingCapacity ||
            hdr->completions_per_chunk == 0) {
            throw std::runtime_error(fmt::format(
                "LayerCompletionQueue::connect: {} is not a valid layer-completion ring "
                "(magic={:#x} capacity={} completions_per_chunk={})",
                shm_name,
                hdr->magic,
                hdr->capacity,
                hdr->completions_per_chunk));
        }
        return std::unique_ptr<LayerCompletionQueue>(
            new LayerCompletionQueue(std::move(shm), shm_name, Role::Connector));
    }
}

uint32_t LayerCompletionQueue::completions_per_chunk() const noexcept { return header()->completions_per_chunk; }

bool LayerCompletionQueue::try_push(const LayerCompletionMessage& msg) {
    auto* hdr = header();
    auto* cell_arr = cells();
    uint64_t pos = hdr->enqueue_pos.load(std::memory_order_relaxed);
    for (;;) {
        LayerCompletionCell& cell = cell_arr[pos & kLayerCompletionRingMask];
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

bool LayerCompletionQueue::try_pop(LayerCompletionMessage& out) {
    auto* hdr = header();
    auto* cell_arr = cells();
    uint64_t pos = hdr->dequeue_pos.load(std::memory_order_relaxed);
    for (;;) {
        LayerCompletionCell& cell = cell_arr[pos & kLayerCompletionRingMask];
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

void LayerCompletionQueue::shutdown() {
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

}  // namespace tt::tt_metal::internal
