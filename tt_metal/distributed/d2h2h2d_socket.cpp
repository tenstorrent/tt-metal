// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/d2h2h2d_socket.hpp>

#include <fmt/format.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/sockets/internal/host_rdma_window.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal::experimental {

struct D2H2H2DSocket::Impl {
    Config cfg{};
    L1MapNew l1{};
    Counters counters{};
    HostRegion* region = nullptr;

    // Declaration order is teardown order and is load-bearing: the window must go before
    // the region it covers, and the legs' overlays before the sockets they alias.
    std::unique_ptr<D2HLeg> d2h;
    std::unique_ptr<H2DLeg> h2d;
    std::unique_ptr<H2HSocket> h2h;

    std::string err;
};

D2H2H2DSocket::D2H2H2DSocket() : impl_(std::make_unique<Impl>()) {}

// Three-step, because the window and the overlays want opposite sides of the unpin: the RMA
// window names pinned pages and must go first, the legs' anonymous re-maps must come after.
D2H2H2DSocket::~D2H2H2DSocket() {
    impl_->h2h.reset();
    if (impl_->region != nullptr) {
        HostRegion::release();
    }
}

std::unique_ptr<D2H2H2DSocket> D2H2H2DSocket::create(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh,
    tt::tt_metal::IDevice* device,
    const Config& cfg,
    std::string& err) {
    err.clear();
    std::unique_ptr<D2H2H2DSocket> s(new D2H2H2DSocket());
    Impl& im = *s->impl_;
    im.cfg = cfg;
    const uint32_t page = tt_uva_frame_page_size(cfg.payload_bytes);

    // Steps 1 and 2 are local and can fail on ONE host (pin limits, shm, a device throw), so
    // they run to a verdict rather than returning: every rank must reach the agreement below.
    const bool local_ok = [&]() -> bool {
        if (mesh == nullptr || device == nullptr || cfg.cores == 0 || cfg.payload_bytes == 0) {
            err = "D2H2H2DSocket::create: mesh, device, cores and payload_bytes are all required";
            return false;
        }
        if (cfg.cores > cfg.grid_width * cfg.grid_height) {
            err = fmt::format(
                "D2H2H2DSocket::create: cores {} exceeds the {} on a {}x{} grid",
                cfg.cores,
                cfg.grid_width * cfg.grid_height,
                cfg.grid_width,
                cfg.grid_height);
            return false;
        }

        const uint32_t l1_base = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
        im.l1 = L1MapNew::compute(
            l1_base, static_cast<uint32_t>(device->l1_size_per_core()), cfg.payload_bytes, cfg.bidirectional);
        if (const std::string e = im.l1.fits(cfg.payload_bytes); !e.empty()) {
            err = e;
            return false;
        }

        uint8_t* const base = HostRegion::reserved_base();

        // 1. Both legs first: they allocate their sockets and MAP_FIXED the rings over the
        //    arenas, which must happen before anything pins those pages.
        D2HLeg::Config dc;
        dc.cores = cfg.cores;
        dc.grid_width = cfg.grid_width;
        dc.payload_bytes = cfg.payload_bytes;
        dc.ring_pages = cfg.ring_pages;
        dc.consumed_addr = im.l1.consumed_addr;
        dc.alias_region_base = base;
        im.d2h = D2HLeg::create(mesh, dc, err);
        if (!im.d2h) {
            return false;
        }

        H2DLeg::Config hc;
        hc.cores = cfg.cores;
        hc.grid_width = cfg.grid_width;
        hc.page_bytes = page;
        hc.ring_pages = cfg.ring_pages;
        hc.alias_region_base = base;
        im.h2d = H2DLeg::create(mesh, hc, err);
        if (!im.h2d) {
            return false;
        }

        // 2. Now pin and publish.
        try {
            im.region = &HostRegion::provision(
                mesh, cfg.chip, cfg.cores, cfg.topo, HostRegion::Grid{cfg.grid_width, cfg.grid_height});
        } catch (const std::exception& ex) {
            err = std::string("host region unavailable: ") + ex.what();
            return false;
        }
        if (const std::string e = im.region->verify_header(); !e.empty()) {
            err = "region header check failed: " + e;
            return false;
        }
        if (im.region->base() != base) {
            err =
                "the region was provisioned somewhere other than the reserved base; the overlays "
                "were built against the wrong address";
            return false;
        }
        return true;
    }();

    if (!RdmaWindow::agree(local_ok, err)) {
        return nullptr;
    }

    // 3. The window last, over pages that are now pinned and will not move.
    H2HSocket::Config mc;
    mc.topo = cfg.topo;
    mc.chip = cfg.chip;
    mc.cores = cfg.cores;
    mc.page_bytes = page;
    mc.ring_pages = cfg.ring_pages;
    mc.region_base = im.region->base();
    mc.region_bytes = im.region->pinned_bytes();
    mc.send_window = cfg.send_window;
    im.h2h = H2HSocket::create(mc, err);
    if (!im.h2h) {
        return nullptr;
    }
    return s;
}

// The whole pipeline. Each leg is non-blocking and refuses rather than waits, so a full
// queue anywhere propagates back to the device as an unacked FIFO page.
uint32_t D2H2H2DSocket::poll() {
    Impl& im = *impl_;
    uint32_t progress = 0;

    progress += im.d2h->poll([&](const SendTask& t) {
        if (!im.h2h->submit(t)) {
            return false;
        }
        ++im.counters.sent;
        return true;
    });

    progress += im.h2h->poll(
        [&](uint32_t core, uint32_t pages) {
            im.d2h->retire(core, pages);
            im.counters.retired += pages;
        },
        [&](const DeliverTask& t) {
            if (!im.h2d->publish(t)) {
                return false;
            }
            ++im.counters.received;
            return true;
        });

    // A drained page is what frees the peer's slot, so the credit follows the device.
    for (uint32_t c = 0; c < im.cfg.cores; ++c) {
        if (const uint32_t pages = im.h2d->drained(c); pages != 0) {
            im.h2h->consumed(c, pages);
            im.counters.drained += pages;
            ++progress;
        }
        // Publish the inbound credit to our own sender, so tt_uva_sync() can see it.
        im.d2h->credit(c, im.h2h->credit_total(c));
    }
    return progress;
}

const L1MapNew& D2H2H2DSocket::l1() const { return impl_->l1; }
const D2H2H2DSocket::Counters& D2H2H2DSocket::counters() const { return impl_->counters; }
HostRegion& D2H2H2DSocket::region() const { return *impl_->region; }
D2HLeg& D2H2H2DSocket::d2h() const { return *impl_->d2h; }
H2HSocket& D2H2H2DSocket::h2h() const { return *impl_->h2h; }
H2DLeg& D2H2H2DSocket::h2d() const { return *impl_->h2d; }

std::string D2H2H2DSocket::barrier() { return impl_->h2h->barrier(); }

bool D2H2H2DSocket::failed() const { return !first_error().empty(); }

// First non-empty wins, in pipeline order, so the earliest failure is the one reported.
std::string D2H2H2DSocket::first_error() const {
    const Impl& im = *impl_;
    for (const std::string& e : {im.d2h->first_error(), im.h2h->first_error(), im.h2d->first_error(), im.err}) {
        if (!e.empty()) {
            return e;
        }
    }
    return {};
}

std::string D2H2H2DSocket::describe() const {
    return fmt::format("{}\n  {}\n  {}", impl_->d2h->describe(), impl_->h2h->describe(), impl_->h2d->describe());
}

}  // namespace tt::tt_metal::experimental
