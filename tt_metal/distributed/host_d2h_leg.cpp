// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/host_d2h_leg.hpp"

#include <fmt/format.h>

#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include "tt_metal/distributed/host_ring_alias.hpp"
#include "tt_metal/distributed/host_uva_layout.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <internal/cluster_noc_helpers.hpp>

#include "tt_metal/distributed/hd_socket_connector_state.hpp"
#include "tt_metal/distributed/hd_socket_descriptor.hpp"

namespace tt::tt_metal::experimental {

namespace {

namespace dist = tt::tt_metal::distributed;

ttsl::Span<const std::byte> byte_span(const void* p, std::size_t n) {
    return ttsl::Span<const std::byte>(static_cast<const std::byte*>(p), n);
}

}  // namespace

// Single-threaded by design: the caller drives poll(). No atomics, no locks.
struct D2HLeg::Impl {
    Config cfg{};
    uint32_t page_size = 0;
    uint32_t fifo_bytes = 0;
    uint32_t device_id = 0;

    // Everything one core owns, in one place. `fifo`/`bytes_sent`/`connector` point into
    // the alias; the counters are bytes, the watermarks are frames.
    struct Core {
        std::unique_ptr<dist::D2HSocket> socket;
        CoreCoord virt{};
        uint32_t cfg_addr = 0;
        uint32_t acked_dev_off = 0;
        uint32_t data_off = 0;  // the ring's offset inside the arena; 0 for today's layouts
        uint8_t* fifo = nullptr;
        const volatile uint32_t* bytes_sent = nullptr;
        dist::HDSocketConnectorState* connector = nullptr;
        uint32_t acked = 0;
        uint32_t read_ptr = 0;
        uint64_t forwarded = 0;
        uint64_t retired = 0;
        uint32_t credited = 0;  // last value poked to the device
    };
    std::vector<Core> core;
    // AFTER core, so it unmaps before the shm it overlays is destroyed. Members die in
    // reverse declaration order, which makes that ordering structural.
    std::unique_ptr<RingAlias> alias;
    uint32_t rr = 0;  // where the next sweep starts

    std::string err;

    void fail(const std::string& what) {
        if (err.empty()) {
            err = what;
        }
    }
};

D2HLeg::D2HLeg() : impl_(std::make_unique<Impl>()) {}
D2HLeg::~D2HLeg() = default;

std::unique_ptr<D2HLeg> D2HLeg::create(
    const std::shared_ptr<dist::MeshDevice>& mesh, const Config& cfg, std::string& err) {
    err.clear();
    if (mesh == nullptr || cfg.cores == 0 || cfg.grid_width == 0 || cfg.payload_bytes == 0 ||
        cfg.ring_pages == 0) {
        err = "D2HLeg::create: mesh, cores, grid_width, payload_bytes and ring_pages are all required";
        return nullptr;
    }
    // page_size and fifo_bytes end up as divisors in poll() and retire(), so a 32-bit wrap
    // to zero here is a SIGFPE later. Checked in 64 bits, where it cannot wrap.
    const uint64_t page64 = static_cast<uint64_t>(cfg.payload_bytes) + kFrameTrailerBytes;
    const uint64_t fifo64 = page64 * cfg.ring_pages;
    if (page64 > UINT32_MAX || fifo64 > UINT32_MAX) {
        err = fmt::format(
            "D2HLeg::create: {} B payload x {} ring pages overflows the 32-bit ring geometry",
            cfg.payload_bytes,
            cfg.ring_pages);
        return nullptr;
    }

    const uint32_t page = tt_uva_frame_page_size(cfg.payload_bytes);
    const uint32_t align = tt::tt_metal::hal::get_pcie_alignment();
    if (page % align != 0) {
        err = fmt::format(
            "D2HLeg::create: page {} B (payload {} + trailer {}) is not a multiple of the {} B PCIe alignment",
            page,
            cfg.payload_bytes,
            kFrameTrailerBytes,
            align);
        return nullptr;
    }

    std::unique_ptr<D2HLeg> leg(new D2HLeg());
    Impl& im = *leg->impl_;
    im.cfg = cfg;
    im.page_size = page;
    im.fifo_bytes = static_cast<uint32_t>(fifo64);
    im.device_id = static_cast<uint32_t>(mesh->get_devices()[0]->id());

    const uint32_t n = cfg.cores;
    im.core.resize(n);

    try {
        for (uint32_t i = 0; i < n; ++i) {
            const CoreCoord logical{i % cfg.grid_width, i / cfg.grid_width};
            // Logical here: D2HSocket does its own translation. The virtual coords below
            // are for the bytes_acked write only.
            im.core[i].virt = mesh->get_devices()[0]->virtual_core_from_logical_core(logical, tt::CoreType::WORKER);
            im.core[i].socket = std::make_unique<dist::D2HSocket>(
                mesh,
                dist::MeshCoreCoord{dist::MeshCoordinate(0, 0), logical},
                im.fifo_bytes,
                dist::D2HSocket::ProcessScope::CrossProcess);
            im.core[i].socket->set_page_size(page);
        }
    } catch (const std::exception& e) {
        err = std::string("D2HLeg::create: D2HSocket construction failed: ") + e.what();
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

    // Refuses if the region is already pinned; see RingAlias.
    im.alias = RingAlias::map(cfg.alias_region_base, AliasArena::Tx, slots, err);
    if (!im.alias) {
        return nullptr;
    }

    for (uint32_t c = 0; c < n; ++c) {
        const auto& d = descs[c];
        uint8_t* const b = im.alias->base(c);
        im.core[c].data_off = d.data_offset;
        im.core[c].fifo = b + d.data_offset;
        im.core[c].bytes_sent = reinterpret_cast<const volatile uint32_t*>(b + d.bytes_sent_offset);
        im.core[c].connector = reinterpret_cast<dist::HDSocketConnectorState*>(b + d.connector_state_offset);
        im.core[c].cfg_addr = d.config_buffer_address;
        im.core[c].acked_dev_off = d.bytes_acked_device_offset;
    }
    return leg;
}

// One pass over every core. Non-blocking; a refusal stops the sweep and the next one
// resumes there, so sustained backpressure cannot let one core starve the rest.
uint32_t D2HLeg::poll(const Sink& sink) {
    Impl& im = *impl_;
    uint32_t accepted = 0;

    for (uint32_t k = 0; k < im.cfg.cores; ++k) {
        const uint32_t c = (im.rr + k) % im.cfg.cores;
        if (im.core[c].fifo == nullptr) {
            continue;
        }
        // Device-written over PCIe. Acquire orders the trailer reads after it.
        const uint32_t sent = __atomic_load_n(const_cast<const uint32_t*>(im.core[c].bytes_sent), __ATOMIC_ACQUIRE);
        const uint64_t available = static_cast<uint64_t>(sent - im.core[c].acked) / im.page_size;
        const uint64_t outstanding = im.core[c].forwarded - im.core[c].retired;
        if (available <= outstanding) {
            continue;
        }

        uint64_t fresh = available - outstanding;
        uint32_t off = static_cast<uint32_t>((im.core[c].read_ptr + outstanding * im.page_size) % im.fifo_bytes);
        while (fresh-- > 0) {
            const uint8_t* const page = im.core[c].fifo + off;
            const FrameTrailer* const t =
                reinterpret_cast<const FrameTrailer*>(page + im.page_size - kFrameTrailerBytes);

            // bytes_sent advancing already implies the page landed: both are posted PCIe
            // writes from one core to one endpoint. An unarmed guard here is corruption.
            if (!tt_uva_frame_armed(t->guard)) {
                im.fail(fmt::format(
                    "d2h: core {} page at ring offset {} has guard {:#x}, expected an armed frame", c, off, t->guard));
                break;
            }
            // Subtract rather than add: t->length is device-written, and near UINT32_MAX
            // the addition wraps to a small number and passes. page_size > trailer always.
            if (t->length > im.page_size - kFrameTrailerBytes) {
                im.fail(fmt::format(
                    "d2h: core {} trailer claims {} payload bytes, which does not fit a {} B page",
                    c,
                    t->length,
                    im.page_size));
                break;
            }

            SendTask task;
            task.core = c;
            // data_off included: the H2H leg turns this back into an address off the region
            // base, and the ring does not have to start at the arena's first byte.
            task.page_offset = tx_arena_offset(c) + im.core[c].data_off + off;
            task.page_bytes = im.page_size;
            task.dst = static_cast<tt_uva_t>(t->dst);
            task.length = t->length;
            task.origin = t->origin;
            task.elapsed = t->elapsed;
            if (!sink(task)) {
                im.rr = c;  // resume here, so this core does not monopolise the sweep
                return accepted;
            }
            im.core[c].forwarded++;
            ++accepted;
            off = static_cast<uint32_t>((off + im.page_size) % im.fifo_bytes);
        }
    }
    im.rr = im.cfg.cores != 0 ? (im.rr + 1) % im.cfg.cores : 0;
    return accepted;
}

// Contiguous prefix only: bytes_acked is one cumulative counter and cannot free a page by
// name, so the caller must retire in the order poll() emitted.
void D2HLeg::retire(uint32_t core, uint32_t pages) {
    Impl& im = *impl_;
    if (core >= im.cfg.cores || pages == 0) {
        return;
    }
    // Over-retiring puts acked ahead of sent, which underflows poll()'s wrap-safe
    // subtraction and opens the device's send gate on pages this host never read.
    const uint64_t outstanding = im.core[core].forwarded - im.core[core].retired;
    if (pages > outstanding) {
        im.fail(fmt::format(
            "d2h: core {} retire of {} pages exceeds the {} forwarded and not yet retired",
            core,
            pages,
            outstanding));
        return;
    }

    // Disarm each page as it is freed -- this is the point the transport is done with it.
    // host_uva_frame.hpp makes a zero guard "not armed", so a reused slot cannot read fresh.
    uint32_t disarm_off = im.core[core].read_ptr;
    for (uint32_t i = 0; i < pages; ++i) {
        auto* const t = reinterpret_cast<FrameTrailer*>(
            im.core[core].fifo + disarm_off + im.page_size - kFrameTrailerBytes);
        __atomic_store_n(&t->guard, UINT64_C(0), __ATOMIC_RELEASE);
        disarm_off = static_cast<uint32_t>((disarm_off + im.page_size) % im.fifo_bytes);
    }

    const uint32_t bytes = static_cast<uint32_t>(static_cast<uint64_t>(pages) * im.page_size);
    im.core[core].acked += bytes;
    im.core[core].read_ptr = (im.core[core].read_ptr + bytes) % im.fifo_bytes;
    im.core[core].retired += pages;

    // Host copy first, device copy last: the device write is what unblocks the kernel's
    // socket_reserve_pages, so it stays the final store.
    if (im.core[core].connector != nullptr) {
        im.core[core].connector->bytes_acked = im.core[core].acked;
        im.core[core].connector->read_ptr = im.core[core].read_ptr;
    }
    const auto& v = im.core[core].virt;
    tt::tt_metal::internal::noc_write_immediate(
        im.device_id,
        static_cast<uint32_t>(v.x),
        static_cast<uint32_t>(v.y),
        im.core[core].cfg_addr + im.core[core].acked_dev_off,
        byte_span(&im.core[core].acked, sizeof(uint32_t)));
}

// Only on change: an unchanged counter is a PCIe write the kernel would not notice.
void D2HLeg::credit(uint32_t core, uint64_t pages) {
    Impl& im = *impl_;
    const uint32_t v = static_cast<uint32_t>(pages);
    if (core >= im.cfg.cores || im.cfg.consumed_addr == 0 || im.core[core].credited == v) {
        return;
    }
    im.core[core].credited = v;
    const auto& c = im.core[core].virt;
    tt::tt_metal::internal::noc_write_immediate(
        im.device_id,
        static_cast<uint32_t>(c.x),
        static_cast<uint32_t>(c.y),
        im.cfg.consumed_addr,
        byte_span(&im.core[core].credited, sizeof(uint32_t)));
}

uint32_t D2HLeg::page_size() const { return impl_->page_size; }
uint32_t D2HLeg::cores() const { return impl_->cfg.cores; }

std::vector<uint32_t> D2HLeg::config_addresses() const {
    std::vector<uint32_t> out;
    out.reserve(impl_->core.size());
    for (const auto& c : impl_->core) {
        out.push_back(c.cfg_addr);
    }
    return out;
}

std::string D2HLeg::describe() const {
    return fmt::format(
        "device {} ({} x D2HSocket, fifo {} B, page {} B; {})",
        impl_->device_id,
        impl_->cfg.cores,
        impl_->fifo_bytes,
        impl_->page_size,
        impl_->alias->describe());
}

std::string D2HLeg::first_error() const { return impl_->err; }

}  // namespace tt::tt_metal::experimental
