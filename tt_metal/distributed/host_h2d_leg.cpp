// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/host_h2d_leg.hpp"

#include <cstddef>

#include <fmt/format.h>

#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include "tt_metal/distributed/host_ring_alias.hpp"
#include <tt-metalium/experimental/sockets/host_uva_layout.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <internal/cluster_noc_helpers.hpp>

#include "tt_metal/distributed/hd_socket_connector_state.hpp"
#include "tt_metal/distributed/hd_socket_descriptor.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"

namespace tt::tt_metal::experimental {

namespace {

namespace dist = tt::tt_metal::distributed;

ttsl::Span<const std::byte> byte_span(const void* p, std::size_t n) {
    return ttsl::Span<const std::byte>(static_cast<const std::byte*>(p), n);
}

}  // namespace

struct H2DLeg::Impl {
    Config cfg{};
    uint32_t device_id = 0;
    uint32_t fifo_bytes = 0;

    // Everything one core owns, in one place. `connector`/`bytes_acked` point into the alias.
    struct Core {
        std::unique_ptr<dist::H2DSocket> socket;
        CoreCoord virt{};
        uint32_t cfg_addr = 0;
        dist::HDSocketConnectorState* connector = nullptr;
        const volatile uint32_t* bytes_acked = nullptr;
        uint32_t sent = 0;            // bytes published to the device
        uint64_t consumed_bytes = 0;  // unwrapped total: page need not divide 2^32
        uint64_t drained_tot = 0;     // pages derived from it
        uint32_t last_acked = 0;      // the wire counter, which wraps
    };
    std::vector<Core> core;
    // AFTER core, so it unmaps before the shm it overlays is destroyed.
    std::unique_ptr<RingAlias> alias;

    std::string err;

    void fail(const std::string& what) {
        if (err.empty()) {
            err = what;
        }
    }
};

H2DLeg::H2DLeg() : impl_(std::make_unique<Impl>()) {}
H2DLeg::~H2DLeg() = default;

std::unique_ptr<H2DLeg> H2DLeg::create(
    const std::shared_ptr<dist::MeshDevice>& mesh, const Config& cfg, std::string& err) {
    err.clear();
    if (mesh == nullptr || cfg.cores == 0 || cfg.grid_width == 0 || cfg.page_bytes == 0) {
        err = "H2DLeg::create: mesh, cores, grid_width and page_bytes are all required";
        return nullptr;
    }

    std::unique_ptr<H2DLeg> leg(new H2DLeg());
    Impl& im = *leg->impl_;
    im.cfg = cfg;
    im.fifo_bytes = cfg.ring_pages * cfg.page_bytes;
    im.device_id = static_cast<uint32_t>(mesh->get_devices()[0]->id());

    const uint32_t n = cfg.cores;
    im.core.resize(n);

    try {
        for (uint32_t i = 0; i < n; ++i) {
            const CoreCoord logical{i % cfg.grid_width, i / cfg.grid_width};
            // Logical here: H2DSocket translates. The virtual coords are for the doorbell.
            im.core[i].virt = mesh->get_devices()[0]->virtual_core_from_logical_core(logical, tt::CoreType::WORKER);
            im.core[i].socket = std::make_unique<dist::H2DSocket>(
                mesh,
                dist::MeshCoreCoord{dist::MeshCoordinate(0, 0), logical},
                tt::tt_metal::BufferType::L1,
                im.fifo_bytes,
                dist::H2DMode::DEVICE_PULL);
            im.core[i].socket->set_page_size(cfg.page_bytes);
        }
    } catch (const std::exception& e) {
        err = std::string("H2DLeg::create: H2DSocket construction failed: ") + e.what();
        return nullptr;
    }

    std::vector<dist::HDSocketDescriptor> descs;
    std::vector<RingAlias::Slot> slots;
    descs.reserve(n);
    slots.reserve(n);
    for (uint32_t c = 0; c < n; ++c) {
        descs.push_back(im.core[c].socket->populate_descriptor());
        const auto& d = descs.back();
        slots.push_back(RingAlias::Slot{d.shm_name, d.shm_size, d.data_offset, d.fifo_size});
    }

    im.alias = RingAlias::map(cfg.alias_region_base, AliasArena::Rx, slots, err);
    if (!im.alias) {
        return nullptr;
    }

    for (uint32_t c = 0; c < n; ++c) {
        const auto& d = descs[c];
        uint8_t* const b = im.alias->base(c);
        im.core[c].cfg_addr = d.config_buffer_address;
        im.core[c].connector = reinterpret_cast<dist::HDSocketConnectorState*>(b + d.connector_state_offset);
        im.core[c].bytes_acked = reinterpret_cast<const volatile uint32_t*>(b + d.bytes_acked_offset);
    }
    return leg;
}

// bytes_sent is the first field of receiver_socket_md, so the offset is 0. Asserted rather
// than assumed: a field inserted ahead of it would advance the wrong word silently.
bool H2DLeg::publish(const DeliverTask& task) {
    // static_assert(offsetof(receiver_socket_md, bytes_sent) == 0, "bytes_sent moved within receiver_socket_md");
    Impl& im = *impl_;
    const uint32_t c = task.core;
    if (c >= im.cfg.cores) {
        im.fail("h2d: core index out of range");
        return false;
    }
    if (task.page_bytes != im.cfg.page_bytes) {
        im.fail("h2d: a frame's page size does not match the ring's");
        return false;
    }
    // The device must have drained a page before its slot can be refilled.
    if (im.core[c].sent - static_cast<uint32_t>(im.core[c].drained_tot * im.cfg.page_bytes) >= im.fifo_bytes) {
        return false;
    }

    im.core[c].sent += task.page_bytes;
    // Host copy first, device copy last: the device write is what releases the kernel.
    if (im.core[c].connector != nullptr) {
        im.core[c].connector->bytes_sent = im.core[c].sent;
    }
    const auto& v = im.core[c].virt;
    tt::tt_metal::internal::noc_write_immediate(
        im.device_id,
        static_cast<uint32_t>(v.x),
        static_cast<uint32_t>(v.y),
        im.core[c].cfg_addr + offsetof(receiver_socket_md, bytes_sent),
        byte_span(&im.core[c].sent, sizeof(uint32_t)));
    return true;
}

// bytes_acked wraps at 2^32 and page_bytes need not divide it, so the division to pages
// happens on an unwrapped total built from 32-bit deltas.
uint32_t H2DLeg::drained(uint32_t core) {
    Impl& im = *impl_;
    if (core >= im.cfg.cores || im.core[core].bytes_acked == nullptr) {
        return 0;
    }
    const uint32_t now = __atomic_load_n(const_cast<const uint32_t*>(im.core[core].bytes_acked), __ATOMIC_ACQUIRE);
    const uint32_t delta = now - im.core[core].last_acked;
    im.core[core].last_acked = now;
    if (delta == 0) {
        return 0;
    }
    im.core[core].consumed_bytes += delta;
    const uint64_t pages = im.core[core].consumed_bytes / im.cfg.page_bytes;
    const uint32_t fresh = static_cast<uint32_t>(pages - im.core[core].drained_tot);
    im.core[core].drained_tot = pages;
    return fresh;
}

std::vector<uint32_t> H2DLeg::config_addresses() const {
    std::vector<uint32_t> out;
    out.reserve(impl_->core.size());
    for (const auto& c : impl_->core) {
        out.push_back(c.cfg_addr);
    }
    return out;
}

std::string H2DLeg::describe() const {
    return fmt::format(
        "device {} ({} x H2DSocket DEVICE_PULL, fifo {} B, page {} B; {})",
        impl_->device_id,
        impl_->cfg.cores,
        impl_->fifo_bytes,
        impl_->cfg.page_bytes,
        impl_->alias->describe());
}

std::string H2DLeg::first_error() const { return impl_->err; }

}  // namespace tt::tt_metal::experimental
