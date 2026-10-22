// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/host_transport/loopback_transport.hpp"

#include <tt_stl/assert.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>

namespace tt::tt_metal::distributed::host_transport {

namespace {

// tag_base -> {sender, receiver}. Serialized against construction and destruction only; the data
// path reads the peer pointer atomically.
struct Registry {
    std::mutex mutex;
    std::map<int, std::pair<LoopbackTransport*, LoopbackTransport*>> pairs;
};

Registry& registry() {
    static Registry r;
    return r;
}

}  // namespace

LoopbackTransport::LoopbackTransport(const TransportParams& params) :
    geom_(params.geometry),
    is_sender_(params.is_sender),
    ring_(static_cast<std::byte*>(params.ring)),
    tag_base_(params.tag_base) {
    TT_FATAL(ring_ != nullptr, "LoopbackTransport needs a ring");
    TT_FATAL(geom_.page_size != 0 && geom_.num_pages != 0, "LoopbackTransport needs a ring geometry");
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto& pair = reg.pairs[tag_base_];
    auto*& mine = is_sender_ ? pair.first : pair.second;
    TT_FATAL(
        mine == nullptr, "LoopbackTransport: tag {} already has a {}", tag_base_, is_sender_ ? "sender" : "receiver");
    mine = this;
    LoopbackTransport* other = is_sender_ ? pair.second : pair.first;
    if (other != nullptr) {
        TT_FATAL(
            other->geom_.page_size == geom_.page_size && other->geom_.num_pages == geom_.num_pages,
            "LoopbackTransport: tag {} ends disagree on ring geometry",
            tag_base_);
        peer_.store(other, std::memory_order_release);
        other->peer_.store(this, std::memory_order_release);
    }
}

LoopbackTransport::~LoopbackTransport() {
    auto& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.pairs.find(tag_base_);
    if (it == reg.pairs.end()) {
        return;
    }
    auto*& mine = is_sender_ ? it->second.first : it->second.second;
    mine = nullptr;
    if (LoopbackTransport* other = peer_.exchange(nullptr, std::memory_order_acq_rel)) {
        other->peer_.store(nullptr, std::memory_order_release);
    }
    if (it->second.first == nullptr && it->second.second == nullptr) {
        reg.pairs.erase(it);
    }
}

bool LoopbackTransport::can_send(uint32_t /*pages*/) const { return peer_.load(std::memory_order_acquire) != nullptr; }

bool LoopbackTransport::send(uint64_t first_page, uint32_t pages) {
    TT_ASSERT(is_sender_, "send() on a receiving transport");
    TT_ASSERT(first_page == sent_, "send() expects the absolute page index: got {}, sent {}", first_page, sent_);
    LoopbackTransport* peer = peer_.load(std::memory_order_acquire);
    if (pages == 0 || peer == nullptr) {
        return false;
    }
    // Same slot on both rings; a run never crosses the wrap.
    uint64_t next = first_page;
    uint32_t left = pages;
    while (left > 0) {
        const uint32_t slot = static_cast<uint32_t>(next % geom_.num_pages);
        const uint32_t run = std::min(left, geom_.num_pages - slot);
        const size_t offset = static_cast<size_t>(slot) * geom_.page_size;
        std::memcpy(peer->ring_ + offset, ring_ + offset, static_cast<size_t>(run) * geom_.page_size);
        next += run;
        left -= run;
    }
    sent_ += pages;
    released_.store(sent_, std::memory_order_release);
    peer->delivered_.store(sent_, std::memory_order_release);
    return true;
}

bool LoopbackTransport::post_credit(uint64_t consumed_pages) {
    TT_ASSERT(!is_sender_, "post_credit() on a sending transport");
    LoopbackTransport* peer = peer_.load(std::memory_order_acquire);
    if (peer == nullptr) {
        return false;
    }
    peer->peer_consumed_.store(consumed_pages, std::memory_order_release);
    return true;
}

std::string LoopbackTransport::describe() const {
    return is_sender_ ? fmt::format(
                            "loopback sender sent={} peer_consumed={} paired={}",
                            sent_,
                            peer_consumed_.load(std::memory_order_acquire),
                            peer_.load(std::memory_order_acquire) != nullptr)
                      : fmt::format(
                            "loopback receiver delivered={} consumed={} paired={}",
                            delivered_.load(std::memory_order_acquire),
                            consumed_,
                            peer_.load(std::memory_order_acquire) != nullptr);
}

}  // namespace tt::tt_metal::distributed::host_transport
