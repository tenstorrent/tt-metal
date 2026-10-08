// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/erisc_e2h_leg.hpp"

#include <algorithm>
#include <atomic>
#include <vector>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>

#include "hostdevcommon/erisc_bridge_layout.h"
#include "hostdevcommon/erisc_host_bridge.h"

namespace tt::tt_metal::experimental {

using namespace tt::tt_fabric;
using namespace tt::tt_metal::distributed;

struct E2HLeg::Impl {
    struct Chan {
        std::unique_ptr<D2HSocket> sock;
        uint8_t* arena = nullptr;  // the aliased Tx arena: the pages the ERISC's DMA landed in
        uint32_t arena_idx = 0;
        uint32_t in_flight = 0;  // handed to the sink, not yet retired
        uint32_t next_slot = 0;
        uint32_t expect = 1;  // next ordering_cntr
        uint64_t drained = 0;
    };
    Config cfg{};
    uint32_t page = 0;
    std::vector<Chan> chans;
    uint64_t drained_total = 0;
    uint64_t unarmed = 0;
    uint64_t out_of_order = 0;

    Chan* by_arena(uint32_t arena_idx) {
        auto it = std::find_if(chans.begin(), chans.end(), [&](const Chan& c) { return c.arena_idx == arena_idx; });
        return it == chans.end() ? nullptr : &*it;
    }
};

E2HLeg::E2HLeg() : impl_(std::make_unique<Impl>()) {}
E2HLeg::~E2HLeg() = default;

std::unique_ptr<E2HLeg> E2HLeg::create(
    std::vector<std::unique_ptr<D2HSocket>> socks, const Config& cfg, std::string& err) {
    err.clear();
    if (socks.empty() || cfg.packet_capacity == 0 || cfg.ring_pages == 0 || cfg.arenas == 0 ||
        cfg.chans_per_link == 0 || cfg.alias_region_base == nullptr) {
        err = "E2HLeg::create: sockets, packet_capacity, ring_pages, arenas, chans_per_link and alias base required";
        return nullptr;
    }
    std::unique_ptr<E2HLeg> leg(new E2HLeg());
    auto& im = *leg->impl_;
    im.cfg = cfg;
    im.page = bridge_socket_page_bytes(cfg.packet_capacity);  // a socket page is a bridge slot
    for (size_t i = 0; i < socks.size(); ++i) {
        Impl::Chan c;
        c.sock = std::move(socks[i]);
        if (c.sock == nullptr || c.sock->get_page_size() != im.page) {
            err = "E2HLeg::create: socket " + std::to_string(i) + " is null or its page is not a bridge slot";
            return nullptr;
        }
        // The ordinal within the link, not the channel number: arena indices count from zero.
        c.arena_idx = bridge_arena_index(cfg.link_idx, static_cast<uint32_t>(i), cfg.chans_per_link);
        if (c.arena_idx >= cfg.arenas) {
            err = "E2HLeg::create: arena " + std::to_string(c.arena_idx) + " is outside the region";
            return nullptr;
        }
        c.arena = cfg.alias_region_base +
                  bridge_segment_offset(c.arena_idx, BridgeArena::Tx, cfg.arenas, cfg.ring_pages, cfg.packet_capacity);
        im.chans.push_back(std::move(c));
    }
    return leg;
}

uint32_t E2HLeg::poll(const Sink& sink) {
    Impl& im = *impl_;
    uint32_t total = 0;
    for (auto& c : im.chans) {
        // pages_available() still counts pages handed out but not yet retired.
        const uint32_t avail = c.sock->pages_available() - c.in_flight;
        uint32_t n = 0;
        for (; n < avail; ++n) {
            const uint32_t slot = (c.next_slot + n) % im.cfg.ring_pages;
            const uint8_t* page = c.arena + static_cast<size_t>(slot) * im.page;
            // Volatile: the device wrote these bytes, so the poll must not be hoisted.
            const auto* d = reinterpret_cast<const volatile BridgeDescriptor*>(
                page + bridge_desc_offset_in_slot(im.cfg.packet_capacity));
            if (!bridge_guard_armed(d->guard)) {  // the guard is written last; a gap ends what has landed
                ++im.unarmed;
                break;
            }
            std::atomic_thread_fence(std::memory_order_acquire);  // payload reads stay below the guard
            BridgeSendTask t;
            t.arena = c.arena_idx;
            t.page_offset = static_cast<uint64_t>(page - im.cfg.alias_region_base);  // region-relative
            t.page_bytes = im.page;
            t.ordering_cntr = d->ordering_cntr;
            t.length = d->length;
            t.origin = c.arena_idx;
            t.elapsed = d->elapsed;
            t.release_cyc = d->release_cyc;
            if (!sink(t)) {
                break;
            }
            // Counted, not refused: ordering_cntr and the lap tag disagreeing means upstream loss.
            if (t.ordering_cntr != c.expect || bridge_guard_seq(d->guard) != t.ordering_cntr) {
                ++im.out_of_order;
            }
            c.expect = t.ordering_cntr + 1;
        }
        c.next_slot = (c.next_slot + n) % im.cfg.ring_pages;
        c.in_flight += n;
        c.drained += n;
        im.drained_total += n;
        total += n;
    }
    return total;
}

void E2HLeg::retire(uint32_t arena, uint32_t pages) {
    auto* c = impl_->by_arena(arena);
    if (c == nullptr || pages == 0) {
        return;
    }
    pages = std::min(pages, c->in_flight);
    c->sock->pop(pages, /*notify_sender=*/true);  // pop, not read: read() would copy
    c->in_flight -= pages;
}

uint32_t E2HLeg::page_size() const { return impl_->page; }
uint64_t E2HLeg::unarmed() const { return impl_->unarmed; }
uint64_t E2HLeg::out_of_order() const { return impl_->out_of_order; }
uint64_t E2HLeg::drained_total() const { return impl_->drained_total; }

std::string E2HLeg::describe() const {
    const Impl& im = *impl_;
    std::string s = "E2H link " + std::to_string(im.cfg.link_idx) + ": " + std::to_string(im.cfg.ring_pages) +
                    " slots x " + std::to_string(im.page) + " B, drained " + std::to_string(im.drained_total) +
                    ", unarmed " + std::to_string(im.unarmed);
    for (const auto& c : im.chans) {
        s += "\n  arena " + std::to_string(c.arena_idx) + ": drained " + std::to_string(c.drained) + ", in flight " +
             std::to_string(c.in_flight);
    }
    return s;
}

}  // namespace tt::tt_metal::experimental
