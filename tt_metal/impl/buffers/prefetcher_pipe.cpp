// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <buffer_types.hpp>
#include <core_coord.hpp>
#include <device.hpp>
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
    uint32_t counters_offset;
    uint32_t page_size;
};

PrefetcherPipeConfigPageLayout compute_prefetcher_pipe_config_page_layout(
    uint32_t num_receivers, uint32_t l1_alignment) {
    const uint32_t noc_xy_offset = prefetcher_pipe_noc_xy_byte_offset();
    const uint32_t counters_offset =
        tt::align(noc_xy_offset + 2 * num_receivers * static_cast<uint32_t>(sizeof(uint32_t)), l1_alignment);
    const uint32_t page_size = counters_offset + 2 * num_receivers * l1_alignment;
    return PrefetcherPipeConfigPageLayout{
        .noc_xy_offset = noc_xy_offset, .counters_offset = counters_offset, .page_size = page_size};
}

}  // namespace

PrefetcherPipe::PrefetcherPipe(
    distributed::MeshDevice* device,
    CoreCoord sender_core,
    const CoreRangeSet& receiver_cores,
    uint32_t ring_size,
    BufferType buffer_type) :
    device_(device), sender_core_(sender_core), receiver_cores_(receiver_cores), ring_size_(ring_size) {
    initialize_prefetcher_pipe(device, sender_core, receiver_cores_, sender_cores_, all_cores_);
    try {
        setup_buffers(buffer_type);
    } catch (...) {
        release_allocations();
        throw;
    }
}

void PrefetcherPipe::build_config_pages() {
    TT_FATAL(config_address_ != 0, "PrefetcherPipe config allocation must exist before building pages");
    TT_FATAL(data_address_ != 0, "PrefetcherPipe data address must be set before building config pages");

    const auto context_id = extract_context_id(device_);
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    const auto layout = compute_prefetcher_pipe_config_page_layout(receiver_cores_.num_cores(), l1_alignment);
    config_page_size_ = layout.page_size;
    credit_reset_offset_ = layout.counters_offset;
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
    sender_page[si++] = layout.counters_offset;
    sender_page[si++] = layout.counters_offset;
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
        receiver_page[rci++] = layout.counters_offset + 2 * ri * l1_alignment;
        receiver_page[rci++] = layout.counters_offset + 2 * ri * l1_alignment + l1_alignment;
        receiver_page[rci++] = static_cast<uint32_t>(sender_phys.x);
        receiver_page[rci++] = static_cast<uint32_t>(sender_phys.y);
        config_pages_[receiver_vec[ri]] = std::move(receiver_page);
    }
}

void PrefetcherPipe::write_config_to_device() {
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

void PrefetcherPipe::setup_buffers(BufferType buffer_type) {
    validate_ring_geometry(device_, ring_size_, buffer_type);

    const auto context_id = extract_context_id(device_);
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    config_page_size_ = compute_prefetcher_pipe_config_page_layout(receiver_cores_.num_cores(), l1_alignment).page_size;

    auto& arena = device_->allocator_impl()->persistent_l1();
    auto data_allocation = arena.allocate(all_cores_, ring_size_, l1_alignment);
    data_allocation_id_ = data_allocation.id;
    TT_FATAL(
        data_allocation.address <= std::numeric_limits<uint32_t>::max(),
        "PrefetcherPipe ring address {} exceeds device-address width",
        data_allocation.address);
    data_address_ = static_cast<uint32_t>(data_allocation.address);

    auto config_allocation = arena.allocate(all_cores_, config_page_size_, l1_alignment);
    config_allocation_id_ = config_allocation.id;
    TT_FATAL(
        config_allocation.address <= std::numeric_limits<uint32_t>::max(),
        "PrefetcherPipe config address {} exceeds device-address width",
        config_allocation.address);
    config_address_ = static_cast<uint32_t>(config_allocation.address);

    const DeviceAddr persistent_end = config_allocation.address + config_allocation.size;
    for (const CoreCoord& core : corerange_to_cores(all_cores_)) {
        const auto& bank_ids = device_->allocator_impl()->get_bank_ids_from_logical_core(BufferType::L1, core);
        TT_FATAL(bank_ids.size() == 1, "Expected one L1 bank for PrefetcherPipe core {}", core.str());
        const auto lowest_global_allocation =
            device_->allocator_impl()->get_lowest_occupied_l1_address(bank_ids.front());
        TT_FATAL(
            !lowest_global_allocation.has_value() || persistent_end <= *lowest_global_allocation,
            "PrefetcherPipe persistent L1 region [{}, {}) overlaps an existing L1 allocation at {} on core {}",
            data_allocation.address,
            persistent_end,
            lowest_global_allocation.value_or(0),
            core.str());
    }
    build_config_pages();
    write_config_to_device();
}

PrefetcherPipe::~PrefetcherPipe() { release_allocations(); }

void PrefetcherPipe::release_allocations() noexcept {
    if (device_ == nullptr) {
        return;
    }
    auto& arena = device_->allocator_impl()->persistent_l1();
    try {
        arena.deallocate(config_allocation_id_);
        config_allocation_id_ = 0;
        arena.deallocate(data_allocation_id_);
        data_allocation_id_ = 0;
    } catch (...) {
        // Destructors must not throw. A missing allocation indicates an internal
        // lifetime error and will be caught by focused arena tests.
    }
}

uint32_t PrefetcherPipe::buffer_address() const { return data_address_; }

uint32_t PrefetcherPipe::config_address() const { return config_address_; }

const std::vector<uint32_t>& PrefetcherPipe::config_page(const CoreCoord& core) const {
    auto it = config_pages_.find(core);
    TT_FATAL(it != config_pages_.end(), "PrefetcherPipe has no host config page for core {}", core.str());
    return it->second;
}

const CoreRangeSet& PrefetcherPipe::sender_cores() const { return sender_cores_; }

const CoreRangeSet& PrefetcherPipe::receiver_cores() const { return receiver_cores_; }

const CoreRangeSet& PrefetcherPipe::all_cores() const { return all_cores_; }

PrefetcherPipe CreatePrefetcherPipe(
    distributed::MeshDevice* device,
    CoreCoord sender_core,
    const CoreRangeSet& receiver_cores,
    uint32_t ring_size,
    BufferType buffer_type) {
    return PrefetcherPipe(device, sender_core, receiver_cores, ring_size, buffer_type);
}

uint8_t AttachPrefetcherPipe(
    Program& program, PrefetcherPipe& prefetcher_pipe, const CoreRangeSet& cores, uint32_t entry_size) {
    return program.impl().add_prefetcher_pipe_attachment(prefetcher_pipe, cores, entry_size);
}

uint32_t CreatePrefetcherPipeRelayDataflowBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& receiver_core_spec,
    const dfb::DataflowBufferConfig& config,
    uint8_t prefetcher_pipe_id) {
    const PrefetcherPipe& pipe = program.impl().get_prefetcher_pipe_attachment(prefetcher_pipe_id);

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
    program.impl().register_prefetcher_pipe_relay_dfb(receiver_cores, prefetcher_pipe_id, relay_dfb_id);
    return relay_dfb_id;
}

}  // namespace tt::tt_metal::experimental
