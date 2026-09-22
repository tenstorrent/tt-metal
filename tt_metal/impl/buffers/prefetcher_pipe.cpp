// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <buffer_types.hpp>
#include <core_coord.hpp>
#include <device.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/context_types.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/api/tt-metalium/hal_types.hpp"
#include <tt_align.hpp>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

#include "hostdev/remote_dfb_config_layout.h"
#include "mesh_device.hpp"
#include "tt_metal/api/tt-metalium/tt_metal.hpp"

namespace tt::tt_metal::experimental {

namespace {

void initialize_prefetcher_pipe(
    IDevice* device,
    CoreCoord sender_core,
    const CoreRangeSet& receiver_cores,
    CoreRangeSet& sender_cores_out,
    CoreRangeSet& all_cores_out) {
    TT_FATAL(device != nullptr, "Device cannot be null");
    TT_FATAL(receiver_cores.num_cores() > 0, "PrefetcherPipe requires at least one receiver");

    sender_cores_out = CoreRangeSet(CoreRange(sender_core));
    all_cores_out = sender_cores_out.merge(receiver_cores);
    TT_FATAL(
        all_cores_out.num_cores() == 1 + receiver_cores.num_cores(),
        "PrefetcherPipe sender {} and receiver cores {} must be disjoint",
        sender_core.str(),
        receiver_cores.str());
}

void validate_ring_geometry(IDevice* device, uint32_t ring_size, BufferType buffer_type) {
    const auto context_id = extract_context_id(device);
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    TT_FATAL(ring_size > 0, "ring_size must be > 0");
    TT_FATAL(
        ring_size % l1_alignment == 0, "ring_size {} must be a multiple of L1_ALIGNMENT {}", ring_size, l1_alignment);
    TT_FATAL(buffer_type == BufferType::L1, "PrefetcherPipe persistent-arena allocations require BufferType::L1");
}

struct PrefetcherPipeConfigPageLayout {
    uint32_t noc_xy_offset;
    uint32_t sent_offset;   // SENT block base: [sent | wr_cursor] slot per (receiver, lane)
    uint32_t acked_offset;  // ACKED block base: [acked] slot per (receiver, lane)
    uint32_t page_size;
};

// See remote_dfb_config_layout.h. SENT and ACKED live in separate cache lines so a core's
// cached stores to its own counters can never write back over the peer's NoC-written ones.
PrefetcherPipeConfigPageLayout compute_prefetcher_pipe_config_page_layout(
    uint32_t num_receivers, uint32_t num_credit_lanes, uint32_t l1_alignment) {
    TT_FATAL(num_credit_lanes >= 1, "num_credit_lanes must be >= 1");
    const uint32_t block_align = std::max(l1_alignment, PREFETCHER_PIPE_CREDIT_BLOCK_ALIGN);
    const uint32_t noc_xy_offset = prefetcher_pipe_noc_xy_byte_offset();
    const uint32_t block_bytes = num_receivers * num_credit_lanes * l1_alignment;
    const uint32_t sent_offset =
        tt::align(noc_xy_offset + 2 * num_receivers * static_cast<uint32_t>(sizeof(uint32_t)), block_align);
    const uint32_t acked_offset = tt::align(sent_offset + block_bytes, block_align);
    const uint32_t page_size = tt::align(acked_offset + block_bytes, l1_alignment);
    return PrefetcherPipeConfigPageLayout{
        .noc_xy_offset = noc_xy_offset,
        .sent_offset = sent_offset,
        .acked_offset = acked_offset,
        .page_size = page_size};
}

}  // namespace

PrefetcherPipeImpl::PrefetcherPipeImpl(
    distributed::MeshDevice* device,
    CoreCoord sender_core,
    const CoreRangeSet& receiver_cores,
    uint32_t ring_size,
    BufferType buffer_type) :
    device_(device),
    sender_core_(sender_core),
    receiver_cores_(receiver_cores),
    ring_size_(ring_size),
    // Quasar reserves lane slots up front; active count starts at 1 and is raised by the first
    // multi-thread consumer Attach / multi-producer relay. WH/BH stay single-lane.
    credit_lane_capacity_(
        (device != nullptr && device->arch() == tt::ARCH::QUASAR) ? PREFETCHER_PIPE_MAX_CREDIT_LANES : 1u) {
    initialize_prefetcher_pipe(device, sender_core, receiver_cores_, sender_cores_, all_cores_);
    try {
        setup_buffers(buffer_type);
    } catch (...) {
        release_allocations();
        throw;
    }
}

void PrefetcherPipeImpl::build_config_pages() {
    TT_FATAL(config_address_ != 0, "PrefetcherPipe config allocation must exist before building pages");
    TT_FATAL(data_address_ != 0, "PrefetcherPipe data address must be set before building config pages");

    const auto context_id = extract_context_id(device_);
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    const auto layout =
        compute_prefetcher_pipe_config_page_layout(receiver_cores_.num_cores(), credit_lane_capacity_, l1_alignment);
    config_page_size_ = layout.page_size;
    // Both credit blocks (and the pad between them) — zeroing this range resets every
    // counter and every write cursor together.
    credit_reset_offset_ = layout.sent_offset;
    credit_reset_size_ = config_page_size_ - credit_reset_offset_;

    const uint32_t data_base_addr = data_address_;
    const uint32_t words_per_page = config_page_size_ / sizeof(uint32_t);
    config_pages_.clear();

    const auto receiver_vec = corerange_to_cores(receiver_cores_);
    const uint32_t num_recv = static_cast<uint32_t>(receiver_vec.size());

    std::vector<uint32_t> sender_page(words_per_page, 0);
    uint32_t si = 0;
    sender_page[si++] = 1;
    sender_page[si++] = num_recv;
    sender_page[si++] = data_base_addr;
    sender_page[si++] = ring_size_;
    sender_page[si++] = data_base_addr;  // word[4]: initial fifo_ptr checkpoint
    sender_page[si++] = 0;               // word[5]: applied_entry_size; set by first Attach
    sender_page[si++] = layout.noc_xy_offset;
    sender_page[si++] = layout.sent_offset;   // word[7]: local sent/wr block; same offset is the remote sent base
    sender_page[si++] = layout.acked_offset;  // word[8]: local acked block (receivers' NoC atomics land here)
    sender_page[si++] = 0;                    // word[9]: reserved (P is per program, in the kernel-config slot)
    for (uint32_t ri = 0; ri < num_recv; ++ri) {
        auto phys = device_->worker_core_from_logical_core(receiver_vec[ri]);
        sender_page[si++] = static_cast<uint32_t>(phys.x);
        sender_page[si++] = static_cast<uint32_t>(phys.y);
    }
    config_pages_[sender_core_] = std::move(sender_page);

    const auto sender_phys = device_->worker_core_from_logical_core(sender_core_);
    for (uint32_t ri = 0; ri < num_recv; ++ri) {
        std::vector<uint32_t> receiver_page(words_per_page, 0);
        uint32_t rci = 0;
        receiver_page[rci++] = 0;
        receiver_page[rci++] = num_recv;
        receiver_page[rci++] = data_base_addr;
        receiver_page[rci++] = ring_size_;
        receiver_page[rci++] = data_base_addr;
        receiver_page[rci++] = 0;
        receiver_page[rci++] = layout.noc_xy_offset;
        // This receiver's lane-0 slot in each block; device adds tid * L1_ALIGNMENT for other
        // lanes. Offsets use credit_lane_capacity_ (allocated stride), not active count. The
        // same offsets address this receiver's mirror slots on the sender page.
        const uint32_t slot = ri * credit_lane_capacity_ * l1_alignment;
        receiver_page[rci++] = layout.sent_offset + slot;   // word[7]: sender's NoC atomics land here
        receiver_page[rci++] = layout.acked_offset + slot;  // word[8]: local acked (cached stores)
        receiver_page[rci++] = 0;                           // word[9]: reserved
        receiver_page[rci++] = static_cast<uint32_t>(sender_phys.x);
        receiver_page[rci++] = static_cast<uint32_t>(sender_phys.y);
        config_pages_[receiver_vec[ri]] = std::move(receiver_page);
    }
}

void PrefetcherPipeImpl::write_config_to_device() {
    TT_FATAL(device_ != nullptr, "PrefetcherPipe device cannot be null");
    for (const auto& [core, page] : config_pages_) {
        for (IDevice* target_device : device_->get_devices()) {
            auto page_copy = page;
            TT_FATAL(
                detail::WriteToDeviceL1(target_device, core, config_address_, page_copy),
                "Failed to write PrefetcherPipe config page to core {} on device {}",
                core.str(),
                target_device->id());
        }
    }
}

void PrefetcherPipeImpl::setup_buffers(BufferType buffer_type) {
    validate_ring_geometry(device_, ring_size_, buffer_type);

    const auto context_id = extract_context_id(device_);
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    config_page_size_ =
        compute_prefetcher_pipe_config_page_layout(receiver_cores_.num_cores(), credit_lane_capacity_, l1_alignment)
            .page_size;

    auto& arena = device_->allocator_impl()->persistent_l1();
    auto data_allocation = arena.allocate(all_cores_, ring_size_, l1_alignment);
    data_allocation_id_ = data_allocation.id;
    TT_FATAL(
        data_allocation.address <= std::numeric_limits<uint32_t>::max(),
        "PrefetcherPipe ring address {} exceeds device-address width",
        data_allocation.address);
    data_address_ = static_cast<uint32_t>(data_allocation.address);

    // Page-relative block offsets are only line-aligned if the page itself is.
    auto config_allocation =
        arena.allocate(all_cores_, config_page_size_, std::max(l1_alignment, PREFETCHER_PIPE_CREDIT_BLOCK_ALIGN));
    config_allocation_id_ = config_allocation.id;
    TT_FATAL(
        config_allocation.address <= std::numeric_limits<uint32_t>::max(),
        "PrefetcherPipe config address {} exceeds device-address width",
        config_allocation.address);
    config_address_ = static_cast<uint32_t>(config_allocation.address);

    const DeviceAddr persistent_begin = std::min(data_allocation.address, config_allocation.address);
    const DeviceAddr persistent_end = std::max(
        data_allocation.address + data_allocation.size, config_allocation.address + config_allocation.size);
    for (const CoreCoord& core : corerange_to_cores(all_cores_)) {
        const auto& bank_ids = device_->allocator_impl()->get_bank_ids_from_logical_core(BufferType::L1, core);
        TT_FATAL(bank_ids.size() == 1, "Expected one L1 bank for PrefetcherPipe core {}", core.str());
        const auto lowest_global_allocation =
            device_->allocator_impl()->get_lowest_occupied_l1_address(bank_ids.front());
        TT_FATAL(
            !lowest_global_allocation.has_value() || persistent_end <= *lowest_global_allocation,
            "PrefetcherPipe persistent L1 region [{}, {}) overlaps an existing L1 allocation at {} on core {}",
            persistent_begin,
            persistent_end,
            lowest_global_allocation.value_or(0),
            core.str());
    }
    build_config_pages();
    write_config_to_device();
}

PrefetcherPipeImpl::~PrefetcherPipeImpl() { release_allocations(); }

void PrefetcherPipeImpl::release_allocations() noexcept {
    if (device_ == nullptr || !device_->is_initialized()) {
        config_allocation_id_ = 0;
        data_allocation_id_ = 0;
        return;
    }
    try {
        auto& arena = device_->allocator_impl()->persistent_l1();
        arena.deallocate(config_allocation_id_);
        arena.deallocate(data_allocation_id_);
    } catch (...) {
        // Destructors must not throw. A missing allocation indicates an internal
        // lifetime error and will be caught by focused arena tests.
        log_warning(LogMetal, "PrefetcherPipe destructor: persistent L1 release failed with unknown exception");
    }
    config_allocation_id_ = 0;
    data_allocation_id_ = 0;
}

uint32_t PrefetcherPipeImpl::buffer_address() const { return data_address_; }

uint32_t PrefetcherPipeImpl::config_address() const { return config_address_; }

const std::vector<uint32_t>& PrefetcherPipeImpl::config_page(const CoreCoord& core) const {
    auto it = config_pages_.find(core);
    TT_FATAL(it != config_pages_.end(), "PrefetcherPipe has no host config page for core {}", core.str());
    return it->second;
}

const CoreRangeSet& PrefetcherPipeImpl::sender_cores() const { return sender_cores_; }

const CoreRangeSet& PrefetcherPipeImpl::receiver_cores() const { return receiver_cores_; }

const CoreRangeSet& PrefetcherPipeImpl::all_cores() const { return all_cores_; }

void PrefetcherPipeImpl::set_active_credit_lanes(uint32_t num_lanes) {
    TT_FATAL(num_lanes >= 1, "active credit lanes must be >= 1");
    TT_FATAL(
        num_lanes <= credit_lane_capacity_,
        "active credit lanes {} exceeds allocated capacity {} "
        "(Quasar CreatePrefetcherPipe reserves PREFETCHER_PIPE_MAX_CREDIT_LANES)",
        num_lanes,
        credit_lane_capacity_);
    if (num_lanes == active_credit_lanes_) {
        return;
    }
    TT_FATAL(
        active_credit_lanes_ == 1,
        "PrefetcherPipe num_pipe_consumer_threads already set to {}, cannot reprogram to {}",
        active_credit_lanes_,
        num_lanes);
    // Host state only. P reaches the device packed into each program's kernel-config slot
    // (build_prefetcher_pipe_config_payload), so it is ordered with the program that uses it;
    // nothing in the persistent config page is touched after Create. The one-shot 1 -> P
    // guard above stays because the persistent credit block is interpreted through P.
    active_credit_lanes_ = num_lanes;
}

void PrefetcherPipeImpl::validate_lane_geometry(uint32_t entry_size, uint32_t num_lanes) const {
    TT_FATAL(num_lanes >= 1, "PrefetcherPipe lane count must be >= 1");
    if (num_lanes == 1) {
        return;
    }
    // Lane mode stripes entry i to lane i % P and acknowledges whole entries only: the ring
    // must be an exact multiple of the entry, and the entry count a multiple of P, or the
    // trailing gap / a partial stripe is never credited and the sender stalls
    // (lane_capacity_units asserts the same on device).
    TT_FATAL(
        ring_size_ % entry_size == 0,
        "PrefetcherPipe with {} credit lanes requires entry_size {} to divide ring_size {}",
        num_lanes,
        entry_size,
        ring_size_);
    TT_FATAL(
        (ring_size_ / entry_size) % num_lanes == 0,
        "PrefetcherPipe ring holds {} entries of {} bytes, which is not a multiple of {} credit lanes",
        ring_size_ / entry_size,
        entry_size,
        num_lanes);
}

PrefetcherPipe::PrefetcherPipe(
    distributed::MeshDevice* device,
    CoreCoord sender_core,
    const CoreRangeSet& receiver_cores,
    uint32_t ring_size,
    BufferType buffer_type) :
    pimpl_(std::make_unique<PrefetcherPipeImpl>(device, sender_core, receiver_cores, ring_size, buffer_type)) {}

PrefetcherPipe::PrefetcherPipe(PrefetcherPipe&&) noexcept = default;

// Defined here rather than in the header because assigning over a pipe destroys the
// PrefetcherPipeImpl it held, which needs the complete type.
PrefetcherPipe& PrefetcherPipe::operator=(PrefetcherPipe&&) noexcept = default;

PrefetcherPipe::~PrefetcherPipe() = default;

// A Program records an Attach by pointing at the PrefetcherPipeImpl, so relocating a handle has to
// leave that object alone; holding it behind a pointer is what makes that true. Containers relocate
// on growth, and a throwing move there would strand the ring, so require the move to be noexcept.
static_assert(
    std::is_nothrow_move_constructible_v<PrefetcherPipe> && !std::is_copy_constructible_v<PrefetcherPipe>,
    "PrefetcherPipe must move without throwing and must not copy: two handles would free one ring");

uint32_t PrefetcherPipe::buffer_address() const { return pimpl_->buffer_address(); }

uint32_t PrefetcherPipe::config_address() const { return pimpl_->config_address(); }

uint32_t PrefetcherPipe::ring_size() const { return pimpl_->ring_size(); }

uint32_t PrefetcherPipe::config_page_size() const { return pimpl_->config_page_size(); }

uint32_t PrefetcherPipe::credit_reset_offset() const { return pimpl_->credit_reset_offset(); }

uint32_t PrefetcherPipe::credit_reset_size() const { return pimpl_->credit_reset_size(); }

CoreCoord PrefetcherPipe::sender_core() const { return pimpl_->sender_core(); }

const CoreRangeSet& PrefetcherPipe::sender_cores() const { return pimpl_->sender_cores(); }

const CoreRangeSet& PrefetcherPipe::receiver_cores() const { return pimpl_->receiver_cores(); }

const CoreRangeSet& PrefetcherPipe::all_cores() const { return pimpl_->all_cores(); }

distributed::MeshDevice* PrefetcherPipe::get_device() const { return pimpl_->get_device(); }

PrefetcherPipe CreatePrefetcherPipe(
    distributed::MeshDevice* device,
    CoreCoord sender_core,
    const CoreRangeSet& receiver_cores,
    uint32_t ring_size,
    BufferType buffer_type) {
    return PrefetcherPipe(device, sender_core, receiver_cores, ring_size, buffer_type);
}

uint8_t AttachPrefetcherPipe(
    Program& program,
    PrefetcherPipe& prefetcher_pipe,
    const CoreRangeSet& cores,
    uint32_t entry_size,
    uint32_t num_pipe_consumer_threads) {
    return program.impl().add_prefetcher_pipe_attachment(
        prefetcher_pipe.impl(), cores, entry_size, num_pipe_consumer_threads);
}

uint32_t CreatePrefetcherPipeRelayDataflowBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& receiver_core_spec,
    const dfb::DataflowBufferConfig& config,
    uint8_t prefetcher_pipe_id) {
    PrefetcherPipeImpl& pipe = program.impl().get_prefetcher_pipe_attachment(prefetcher_pipe_id);

    CoreRangeSet receiver_cores;
    if (std::holds_alternative<CoreCoord>(receiver_core_spec)) {
        receiver_cores = CoreRangeSet({CoreRange(std::get<CoreCoord>(receiver_core_spec))});
    } else if (std::holds_alternative<CoreRange>(receiver_core_spec)) {
        receiver_cores = CoreRangeSet({std::get<CoreRange>(receiver_core_spec)});
    } else {
        receiver_cores = std::get<CoreRangeSet>(receiver_core_spec);
    }

    TT_FATAL(
        pipe.receiver_cores().contains(receiver_cores),
        "CreatePrefetcherPipeRelayDataflowBuffer: relay cores {} must be a subset of receiver cores {}",
        receiver_cores.str(),
        pipe.receiver_cores().str());
    TT_FATAL(
        config.num_producers <= pipe.credit_lane_capacity(),
        "CreatePrefetcherPipeRelayDataflowBuffer: num_producers {} exceeds PrefetcherPipe credit lane "
        "capacity {} (Quasar sizes the config page for PREFETCHER_PIPE_MAX_CREDIT_LANES at pipe create)",
        config.num_producers,
        pipe.credit_lane_capacity());
    TT_FATAL(config.entry_size > 0, "CreatePrefetcherPipeRelayDataflowBuffer: entry_size must be > 0");
    TT_FATAL(
        pipe.ring_size() % config.entry_size == 0,
        "CreatePrefetcherPipeRelayDataflowBuffer: entry size {} must divide PrefetcherPipe ring size {}",
        config.entry_size,
        pipe.ring_size());
    TT_FATAL(
        config.num_entries == pipe.ring_size() / config.entry_size,
        "CreatePrefetcherPipeRelayDataflowBuffer: depth {} must equal ring_size/entry_size ({})",
        config.num_entries,
        pipe.ring_size() / config.entry_size);

    auto relay_config = config;
    relay_config.borrows_memory = true;
    relay_config.is_relay = true;
    const uint32_t relay_dfb_id = dfb::CreateDataflowBuffer(program, receiver_cores, relay_config);
    // register_prefetcher_pipe_relay_dfb programs active credit lanes from num_producers.
    program.impl().register_prefetcher_pipe_relay_dfb(receiver_cores, prefetcher_pipe_id, relay_dfb_id);
    return relay_dfb_id;
}

}  // namespace tt::tt_metal::experimental
