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
#include "impl/dataflow_buffer/prefetcher_pipe_dram_sender_internal.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "tt_metal/api/tt-metalium/hal_types.hpp"
#include <tt_align.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hostdev/remote_dfb_config_layout.h"
#include "llrt/tt_cluster.hpp"
#include "mesh_device.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "tt_metal/api/tt-metalium/tt_metal.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_page_mapping.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/core_subset_write/mesh_command_queue.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::experimental {

namespace {

std::atomic<uint64_t> next_prefetcher_pipe_identity{1};

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

uint32_t l1_alignment_for(const distributed::MeshDevice* device) {
    return device->impl().metal_env().get_hal().get_alignment(HalMemType::L1);
}

// Page-relative block offsets are only line-aligned if the page itself is.
uint32_t config_page_alignment_for(const distributed::MeshDevice* device) {
    return std::max(l1_alignment_for(device), PREFETCHER_PIPE_CREDIT_BLOCK_ALIGN);
}

CoreRangeSet cores_of(const PrefetcherPipeConfigPages& pages) {
    std::vector<CoreCoord> cores;
    cores.reserve(pages.size());
    for (const auto& [core, page] : pages) {
        cores.push_back(core);
    }
    return CoreRangeSet(ttsl::Span<const CoreCoord>(cores));
}

uint32_t checked_l1_address(DeviceAddr address, std::string_view what) {
    TT_FATAL(
        address <= std::numeric_limits<uint32_t>::max(),
        "PrefetcherPipeSpace {} address {} exceeds device-address width",
        what,
        address);
    return static_cast<uint32_t>(address);
}

struct PrefetcherPipePageCommon {
    const PrefetcherPipeConfigPageLayout& layout;
    uint32_t l1_alignment;
    uint32_t num_receivers;
    uint32_t data_base_addr;
    uint32_t ring_size;
    uint32_t applied_entry_size;
};

uint32_t write_shared_config_words(
    std::vector<uint32_t>& page, const PrefetcherPipePageCommon& common, bool is_sender) {
    uint32_t i = 0;
    page[i++] = is_sender ? 1u : 0u;
    page[i++] = common.num_receivers;
    page[i++] = common.data_base_addr;
    page[i++] = common.ring_size;
    page[i++] = common.data_base_addr;
    page[i++] = common.applied_entry_size;
    page[i++] = common.layout.noc_xy_offset;
    return i;
}

std::vector<uint32_t> build_sender_config_page(
    const PrefetcherPipePageCommon& common,
    uint32_t peer_counter_offset,
    const std::vector<CoreCoord>& receiver_noc_xy) {
    std::vector<uint32_t> page(common.layout.page_size / sizeof(uint32_t), 0);
    uint32_t i = write_shared_config_words(page, common, true);
    page[i++] = common.layout.sent_offset;
    page[i++] = common.layout.acked_offset;
    page[i++] = peer_counter_offset;
    for (const CoreCoord& phys : receiver_noc_xy) {
        page[i++] = static_cast<uint32_t>(phys.x);
        page[i++] = static_cast<uint32_t>(phys.y);
    }
    return page;
}

std::vector<uint32_t> build_receiver_config_page(
    const PrefetcherPipePageCommon& common,
    uint32_t receiver_index,
    uint32_t peer_counter_offset,
    CoreCoord sender_noc_xy) {
    std::vector<uint32_t> page(common.layout.page_size / sizeof(uint32_t), 0);
    uint32_t i = write_shared_config_words(page, common, false);
    const uint32_t slot = receiver_index * common.l1_alignment;
    page[i++] = common.layout.sent_offset + slot;
    page[i++] = common.layout.acked_offset + slot;
    page[i++] = peer_counter_offset;
    page[i++] = static_cast<uint32_t>(sender_noc_xy.x);
    page[i++] = static_cast<uint32_t>(sender_noc_xy.y);
    return page;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// PrefetcherPipeSpaceImpl
// ---------------------------------------------------------------------------------------------

PrefetcherPipeSpaceImpl::PrefetcherPipeSpaceImpl(
    const distributed::MeshDevice& device, PrefetcherPipeSpaceConfig config) :
    device_(&device),
    config_(std::move(config)),
    // Quasar reserves lane slots up front; the active count (default 1) is raised per pipe by the
    // first multi-thread receiver bind. WH/BH stay single-lane.
    credit_lane_capacity_(device.arch() == tt::ARCH::QUASAR ? PREFETCHER_PIPE_MAX_CREDIT_LANES : 1u) {
    TT_FATAL(config_.buffer_type == BufferType::L1, "PrefetcherPipeSpace: buffer_type must be BufferType::L1");
    TT_FATAL(config_.ring_size > 0, "PrefetcherPipeSpace: ring_size must be > 0");
    const uint32_t l1_alignment = l1_alignment_for(device_);
    TT_FATAL(
        config_.ring_size % l1_alignment == 0,
        "PrefetcherPipeSpace: ring_size {} must be a multiple of L1_ALIGNMENT {}",
        config_.ring_size,
        l1_alignment);
    // Receivers are always worker cores drawn from receiver_domain, so a space with no receiver
    // domain could never carve a pipe, and a pipe never has more receivers than the domain holds.
    // Bounding the per-pipe maximum by the domain also keeps the config-page layout arithmetic
    // (receivers x lanes x slot bytes) far from 32-bit overflow.
    TT_FATAL(
        config_.receiver_domain.num_cores() > 0,
        "PrefetcherPipeSpace: receiver_domain is empty; every pipe needs at least one worker receiver");
    TT_FATAL(config_.max_receivers_per_pipe >= 1, "PrefetcherPipeSpace: max_receivers_per_pipe must be >= 1");
    TT_FATAL(
        config_.max_receivers_per_pipe <= config_.receiver_domain.num_cores(),
        "PrefetcherPipeSpace: max_receivers_per_pipe {} exceeds the {} cores in receiver_domain; a pipe cannot have "
        "more receivers than the domain it is carved from",
        config_.max_receivers_per_pipe,
        config_.receiver_domain.num_cores());
    TT_FATAL(
        config_.sender_cores.num_cores() > 0 || config_.num_dram_senders > 0,
        "PrefetcherPipeSpace: provide worker sender_cores, num_dram_senders, or both");
    if (config_.l1_source == PrefetcherPipeL1Source::Default) {
        config_.l1_source =
            config_.num_dram_senders > 0 ? PrefetcherPipeL1Source::Allocator : PrefetcherPipeL1Source::PersistentArena;
    }
    TT_FATAL(
        config_.num_dram_senders == 0 || config_.l1_source == PrefetcherPipeL1Source::Allocator,
        "PrefetcherPipeSpace: a space with DRAM senders ({} requested) takes its L1 from the allocator; "
        "PrefetcherPipeL1Source::PersistentArena is for worker-only spaces",
        config_.num_dram_senders);
    reservation_cores_ = config_.sender_cores.merge(config_.receiver_domain);
    // Worker-only public surface: sender_cores / receiver_domain are logical worker coordinates,
    // which the persistent arena and the L1 allocator validate against the worker grid.
    layout_ =
        compute_prefetcher_pipe_config_page_layout(config_.max_receivers_per_pipe, credit_lane_capacity_, l1_alignment);
}

void PrefetcherPipeSpaceImpl::attach_l1(std::shared_ptr<PrefetcherPipeSpaceL1> l1) {
    TT_FATAL(l1 != nullptr && l1_ == nullptr, "PrefetcherPipeSpace: L1 is attached exactly once");
    l1_ = std::move(l1);
    // Domain template: every reservation core holds a zeroed page at config_address(), so a page
    // exists (and is harmless) on cores no pipe ever claims. Carving overwrites the real pages. An
    // Allocator space's config buffer was zeroed when it was allocated.
    if (config_.l1_source == PrefetcherPipeL1Source::PersistentArena) {
        const std::vector<uint32_t> zero_page(layout_.page_size / sizeof(uint32_t), 0);
        write_page(reservation_cores_, zero_page);
    }
}

const PrefetcherPipeSpaceL1& PrefetcherPipeSpaceImpl::l1() const {
    TT_FATAL(l1_ != nullptr, "PrefetcherPipeSpace has no L1 attached (internal error)");
    return *l1_;
}

PrefetcherPipeSpaceImpl::~PrefetcherPipeSpaceImpl() {
    if (!live_pipes_.empty()) {
        // Lifetime violation: the handle was dropped while pipes carved from it are alive. Their
        // L1 is about to be released, so detach them; any later use fails with a clear message.
        log_fatal(
            LogMetal,
            "PrefetcherPipeSpace destroyed with {} live PrefetcherPipe(s) carved from it; a space must outlive its "
            "pipes. Those pipes are now unusable.",
            live_pipes_.size());
        for (PrefetcherPipeImpl* pipe : live_pipes_) {
            pipe->detach_from_space();
        }
        live_pipes_.clear();
    }
}

// ---------------------------------------------------------------------------------------------
// PrefetcherPipeSpaceL1
// ---------------------------------------------------------------------------------------------

PrefetcherPipeSpaceL1::PrefetcherPipeSpaceL1(
    distributed::MeshDevice& device,
    PrefetcherPipeL1Source source,
    const CoreRangeSet& cores,
    uint32_t ring_size,
    uint32_t config_page_size) :
    device_(&device) {
    TT_FATAL(
        source != PrefetcherPipeL1Source::Default,
        "PrefetcherPipeSpaceL1 requires a resolved L1 source (internal error)");
    try {
        if (source == PrefetcherPipeL1Source::PersistentArena) {
            allocate_from_arena(cores, ring_size, config_page_size);
        } else {
            allocate_from_allocator(cores, ring_size, config_page_size);
        }
    } catch (...) {
        release();
        throw;
    }
}

PrefetcherPipeSpaceL1::~PrefetcherPipeSpaceL1() { release(); }

void PrefetcherPipeSpaceL1::allocate_from_arena(
    const CoreRangeSet& cores, uint32_t ring_size, uint32_t config_page_size) {
    const uint32_t l1_alignment = l1_alignment_for(device_);
    auto& arena = device_->allocator_impl()->persistent_l1();

    // One ring allocation and one page allocation over the whole domain: every domain core gets
    // the same two addresses, which is the lockstep contract every carved pipe relies on.
    auto data_allocation = arena.allocate(cores, ring_size, l1_alignment);
    data_allocation_id_ = data_allocation.id;
    data_address_ = checked_l1_address(data_allocation.address, "ring");

    auto config_allocation = arena.allocate(cores, config_page_size, config_page_alignment_for(device_));
    config_allocation_id_ = config_allocation.id;
    config_address_ = checked_l1_address(config_allocation.address, "config");

    const DeviceAddr persistent_begin = std::min(data_allocation.address, config_allocation.address);
    const DeviceAddr persistent_end =
        std::max(data_allocation.address + data_allocation.size, config_allocation.address + config_allocation.size);
    for (const CoreCoord& core : corerange_to_cores(cores)) {
        const auto& bank_ids = device_->allocator_impl()->get_bank_ids_from_logical_core(BufferType::L1, core);
        TT_FATAL(bank_ids.size() == 1, "Expected one L1 bank for PrefetcherPipeSpace core {}", core.str());
        const auto lowest_global_allocation =
            device_->allocator_impl()->get_lowest_occupied_l1_address(bank_ids.front());
        TT_FATAL(
            !lowest_global_allocation.has_value() || persistent_end <= *lowest_global_allocation,
            "PrefetcherPipeSpace persistent L1 region [{}, {}) overlaps an existing L1 allocation at {} on core {}",
            persistent_begin,
            persistent_end,
            lowest_global_allocation.value_or(0),
            core.str());
    }
}

void PrefetcherPipeSpaceL1::allocate_from_allocator(
    const CoreRangeSet& cores, uint32_t ring_size, uint32_t config_page_size) {
    const uint32_t l1_alignment = l1_alignment_for(device_);
    const uint32_t num_cores = cores.num_cores();
    // One page per core, so every core holds its slice at the buffer's single lockstep address.
    auto allocate_per_core = [&](uint32_t bytes_per_core) {
        ShardSpecBuffer shard_parameters(cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = static_cast<DeviceAddr>(bytes_per_core) * num_cores},
            distributed::DeviceLocalBufferConfig{
                .page_size = bytes_per_core,
                .buffer_type = BufferType::L1,
                .sharding_args = BufferShardingArgs(std::move(shard_parameters), TensorMemoryLayout::HEIGHT_SHARDED),
            },
            device_);
    };

    data_buffer_ = allocate_per_core(ring_size);
    data_address_ = checked_l1_address(data_buffer_->address(), "ring");

    // The allocator only guarantees L1 alignment, so each core's slot carries the slack to move the
    // page up to config_page_alignment_for().
    const uint32_t config_alignment = config_page_alignment_for(device_);
    const uint32_t config_slot_size = tt::align(config_page_size, l1_alignment) + (config_alignment - l1_alignment);
    config_buffer_ = allocate_per_core(config_slot_size);
    config_address_ = checked_l1_address(tt::align(config_buffer_->address(), config_alignment), "config");

    // Zero every core's page, which is the template a core no pipe has claimed holds. Through the
    // command queue, like the pages carved later, because earlier work may still be using this L1.
    const std::vector<uint32_t> zeros(config_buffer_->size() / sizeof(uint32_t), 0);
    distributed::EnqueueWriteMeshBuffer(device_->mesh_command_queue(), config_buffer_, zeros, /*blocking=*/false);
}

void PrefetcherPipeSpaceL1::enqueue_config_pages(
    const CoreRangeSet& cores,
    const std::function<const PrefetcherPipeConfigPages&(const distributed::MeshCoordinate&)>& pages_on) const {
    TT_FATAL(config_buffer_ != nullptr, "PrefetcherPipeSpace: only Allocator L1 has a config buffer (internal error)");
    const auto& core_to_core_id = config_buffer_->get_reference_buffer()->get_buffer_page_mapping()->core_to_core_id;
    const size_t slot_words = config_buffer_->page_size() / sizeof(uint32_t);
    const size_t page_offset_words = (config_address_ - config_buffer_->address()) / sizeof(uint32_t);

    const PrefetcherPipeConfigPages* imaged_pages = nullptr;
    HostBuffer image;
    auto images = DistributedHostBuffer::create(device_->get_view());
    for (const auto& coord : distributed::MeshCoordinateRange(device_->shape())) {
        images.emplace_shard(coord, [&] {
            const PrefetcherPipeConfigPages& pages = pages_on(coord);
            if (&pages != imaged_pages) {
                std::vector<uint32_t> words(config_buffer_->size() / sizeof(uint32_t), 0);
                for (const auto& [core, page] : pages) {
                    const auto slot = core_to_core_id.find(core);
                    TT_FATAL(
                        slot != core_to_core_id.end() && page_offset_words + page.size() <= slot_words,
                        "PrefetcherPipeSpace: no config slot holds the {}-word page of core {} (internal error)",
                        page.size(),
                        core.str());
                    std::ranges::copy(page, words.begin() + slot->second * slot_words + page_offset_words);
                }
                image = HostBuffer(std::move(words));
                imaged_pages = &pages;
            }
            return image;
        });
    }
    core_subset_write::enqueue_write(device_->mesh_command_queue(), *config_buffer_, images, /*blocking=*/false, cores);
}

void PrefetcherPipeSpaceL1::release() noexcept {
    // The MeshBuffers release themselves, and handle a device that is already closed.
    const std::array allocation_ids = {std::exchange(config_allocation_id_, 0), std::exchange(data_allocation_id_, 0)};
    if (!device_->is_initialized()) {
        return;
    }
    try {
        auto& arena = device_->allocator_impl()->persistent_l1();
        for (const uint64_t id : allocation_ids) {
            if (id != 0) {
                arena.deallocate(id);
            }
        }
    } catch (...) {
        // Destructors must not throw. A missing allocation indicates an internal
        // lifetime error and will be caught by focused arena tests.
        log_warning(LogMetal, "PrefetcherPipeSpace: persistent L1 release failed with unknown exception");
    }
}

CoreRangeSet PrefetcherPipeSpaceImpl::unclaimed_cores() const {
    std::vector<CoreCoord> free_cores;
    for (const CoreCoord& core : corerange_to_cores(reservation_cores_)) {
        if (!claimed_.contains(core)) {
            free_cores.push_back(core);
        }
    }
    return free_cores.empty() ? CoreRangeSet{} : CoreRangeSet(ttsl::Span<const CoreCoord>(free_cores));
}

void PrefetcherPipeSpaceImpl::validate_carve(
    CoreCoord sender, const CoreRangeSet& receivers, const std::unordered_set<CoreCoord>* pending) const {
    TT_FATAL(
        config_.sender_cores.contains(sender),
        "PrefetcherPipeSpace::create_pipe: sender {} is not one of the space's sender_cores {} (worker cores only; "
        "DRAM senders go through the impl-only DRAM-sender helpers)",
        sender.str(),
        config_.sender_cores.str());
    TT_FATAL(receivers.num_cores() > 0, "PrefetcherPipeSpace::create_pipe: a pipe requires at least one receiver");
    TT_FATAL(
        receivers.num_cores() <= config_.max_receivers_per_pipe,
        "PrefetcherPipeSpace::create_pipe: {} receivers exceed the space's max_receivers_per_pipe {}",
        receivers.num_cores(),
        config_.max_receivers_per_pipe);
    TT_FATAL(
        config_.receiver_domain.contains(receivers),
        "PrefetcherPipeSpace::create_pipe: receivers {} are not all inside the space's receiver_domain {}",
        receivers.str(),
        config_.receiver_domain.str());
    TT_FATAL(
        !receivers.contains(sender),
        "PrefetcherPipeSpace::create_pipe: sender {} must not be one of the receivers {}",
        sender.str(),
        receivers.str());
    const auto check_free = [&](const CoreCoord& core) {
        TT_FATAL(
            !claimed_.contains(core),
            "PrefetcherPipeSpace::create_pipe: core {} is already claimed by a live pipe; destroy that pipe first",
            core.str());
        TT_FATAL(
            pending == nullptr || !pending->contains(core),
            "PrefetcherPipeSpace::create_pipes: core {} appears in more than one pipe of the batch",
            core.str());
    };
    check_free(sender);
    for (const CoreCoord& core : corerange_to_cores(receivers)) {
        check_free(core);
    }
}

void PrefetcherPipeSpaceImpl::claim(const CoreRangeSet& cores, PrefetcherPipeImpl& owner) {
    for (const CoreCoord& core : corerange_to_cores(cores)) {
        TT_FATAL(claimed_.insert(core).second, "PrefetcherPipeSpace: core {} is already claimed", core.str());
    }
    live_pipes_.insert(&owner);
}

void PrefetcherPipeSpaceImpl::unclaim(const CoreRangeSet& cores, PrefetcherPipeImpl& owner) noexcept {
    for (const CoreCoord& core : corerange_to_cores(cores)) {
        claimed_.erase(core);
    }
    live_pipes_.erase(&owner);
}

void PrefetcherPipeSpaceImpl::write_page(const CoreRangeSet& cores, const std::vector<uint32_t>& page) const {
    TT_FATAL(page.size() * sizeof(uint32_t) == layout_.page_size, "PrefetcherPipeSpace: page has the wrong size");
    const auto& cluster = device_->impl().metal_env().get_cluster();
    const uint32_t page_bytes = static_cast<uint32_t>(page.size() * sizeof(uint32_t));
    // Named: ranges() returns a view into the merged set, which must outlive the loop.
    const CoreRangeSet merged = cores.merge_ranges();
    for (auto* target_device : device_->impl().get_devices()) {
        for (const CoreRange& range : merged.ranges()) {
            if (range.size() == 1) {
                auto page_copy = page;
                TT_FATAL(
                    ::tt::tt_metal::detail::WriteToDeviceL1(
                        target_device, range.start_coord, config_address(), page_copy),
                    "Failed to write PrefetcherPipe config page to core {} on device {}",
                    range.start_coord.str(),
                    target_device->id());
                continue;
            }
            // Logical worker rectangles map to virtual rectangles (same mapping dispatch uses for
            // its multicasts), so one NOC multicast covers the range.
            const CoreCoord start = target_device->virtual_core_from_logical_core(range.start_coord, CoreType::WORKER);
            const CoreCoord end = target_device->virtual_core_from_logical_core(range.end_coord, CoreType::WORKER);
            cluster.noc_multicast_write(page.data(), page_bytes, target_device->id(), start, end, config_address());
        }
    }
}

void PrefetcherPipeSpaceImpl::write_pages(const PrefetcherPipeConfigPages& pages) const {
    if (config_.l1_source == PrefetcherPipeL1Source::Allocator) {
        l1().enqueue_config_pages(
            cores_of(pages),
            [&](const distributed::MeshCoordinate&) -> const PrefetcherPipeConfigPages& { return pages; });
        return;
    }
    // Group cores by payload so identical pages (a pipe's receivers once the page layout no
    // longer depends on the receiver index) go out as rectangles.
    std::map<std::vector<uint32_t>, std::vector<CoreCoord>> cores_by_page;
    for (const auto& [core, page] : pages) {
        cores_by_page[page].push_back(core);
    }
    for (const auto& [page, cores] : cores_by_page) {
        write_page(CoreRangeSet(ttsl::Span<const CoreCoord>(cores)), page);
    }
}

void PrefetcherPipeSpaceImpl::write_device_pages(
    const std::map<distributed::MeshCoordinate, PrefetcherPipeConfigPages>& pages) const {
    TT_FATAL(
        config_.l1_source == PrefetcherPipeL1Source::Allocator,
        "PrefetcherPipeSpace: per-device config pages need an Allocator space (internal error)");
    TT_FATAL(!pages.empty(), "PrefetcherPipeSpace: no device pages to write (internal error)");
    l1().enqueue_config_pages(
        cores_of(pages.begin()->second),
        [&](const distributed::MeshCoordinate& coord) -> const PrefetcherPipeConfigPages& {
            const auto it = pages.find(coord);
            TT_FATAL(it != pages.end(), "PrefetcherPipeSpace: no config pages for device {} (internal error)", coord);
            return it->second;
        });
}

void PrefetcherPipeSpaceImpl::order_dram_sender_pages() const {
    // A DRISC sender writes ring data and credits into receiver L1 as soon as a request reaches it,
    // and requests arrive over its H2D socket, not the command queue. A running Tensor prefetcher
    // holds every later request until the queue gets here. One started later is launched straight
    // from the host, so the queue has to have drained by then.
    auto& cq = device_->mesh_command_queue();
    if (device_->impl().is_tensor_prefetcher_active()) {
        WaitForCqOnTensorPrefetcher(cq, std::nullopt);
    } else {
        distributed::Finish(cq);
    }
}

PrefetcherPipe PrefetcherPipeSpaceImpl::create_pipe(CoreCoord sender, const CoreRangeSet& receivers) {
    return PrefetcherPipe(std::make_unique<PrefetcherPipeImpl>(*this, sender, receivers));
}

std::vector<PrefetcherPipe> PrefetcherPipeSpaceImpl::create_pipes(
    std::span<const std::pair<CoreCoord, CoreRangeSet>> pipes) {
    // Validate the whole batch before claiming anything, so a bad mapping leaves the space as
    // it was.
    std::unordered_set<CoreCoord> pending;
    for (const auto& [sender, receivers] : pipes) {
        validate_carve(sender, receivers, &pending);
        pending.insert(sender);
        for (const CoreCoord& core : corerange_to_cores(receivers)) {
            pending.insert(core);
        }
    }
    std::vector<PrefetcherPipe> result;
    result.reserve(pipes.size());
    for (const auto& [sender, receivers] : pipes) {
        result.push_back(create_pipe(sender, receivers));
    }
    return result;
}

void PrefetcherPipeSpaceImpl::set_dram_sender_cores(std::span<const CoreCoord> dram_senders) {
    TT_FATAL(config_.num_dram_senders > 0, "set_dram_sender_cores: this is a worker-only space");
    TT_FATAL(
        dram_senders.size() <= config_.num_dram_senders,
        "set_dram_sender_cores: {} cores exceed the space capacity {}",
        dram_senders.size(),
        config_.num_dram_senders);
    std::unordered_set<CoreCoord> distinct;
    for (const CoreCoord& sender : dram_senders) {
        TT_FATAL(distinct.insert(sender).second, "set_dram_sender_cores: duplicate DRAM sender {}", sender.str());
    }
    // Senders reserved by an earlier call keep their state; only the new ones are reserved here,
    // so a re-carve for another layout reuses every sender the two layouts share.
    std::vector<CoreCoord> new_senders;
    for (const CoreCoord& sender : dram_senders) {
        if (!dram_sender_allocations_.contains(sender)) {
            new_senders.push_back(sender);
        }
    }
    if (new_senders.empty()) {
        return;
    }
    TT_FATAL(
        dram_sender_allocations_.size() + new_senders.size() <= config_.num_dram_senders,
        "set_dram_sender_cores: {} DRAM senders are already reserved and {} more were requested, exceeding the "
        "space capacity {}. Size num_dram_senders for every sender any layout carved from this space uses.",
        dram_sender_allocations_.size(),
        new_senders.size(),
        config_.num_dram_senders);

    std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_mapping;
    sender_mapping.reserve(new_senders.size());
    for (const CoreCoord& sender : new_senders) {
        sender_mapping.emplace_back(sender, CoreRangeSet{});
    }
    validate_dram_senders_across_mesh(device_, sender_mapping);

    const auto& hal = MetalContext::instance(device_->impl().get_context_id()).hal();
    TT_FATAL(
        hal.has_programmable_core_type(HalProgrammableCoreType::DRAM),
        "set_dram_sender_cores requires programmable DRAM cores");
    const uint32_t alignment = config_page_alignment_for(device_);

    // Reserve each sender's state before a pipe on it is carved. A later carve on that sender only
    // stamps/reuses this storage, so destroy/recarve is allocation-free. The DRISC sender-state zone
    // is fixed, so reserving another sender later does not move any co-resident kernel's layout.
    // Allocate into a local map first so a full zone leaves the reservations as they were.
    std::unordered_map<CoreCoord, std::shared_ptr<DriscL1Allocation>> allocations;
    for (const CoreCoord& sender : new_senders) {
        allocations.emplace(sender, device_->impl().drisc_l1_arena().allocate_on(sender, layout_.page_size, alignment));
    }
    dram_sender_allocations_.merge(allocations);
}

// The checks a DRAM carve shares with every other DRAM carve, single-pipe or batched. The
// reservation check is the caller's, because the batch form runs before set_dram_sender_cores has
// reserved anything.
void PrefetcherPipeSpaceImpl::validate_dram_pipe_geometry(CoreCoord sender, const CoreRangeSet& receivers) const {
    TT_FATAL(receivers.num_cores() > 0, "DRAM pipe with sender {} has no receivers", sender.str());
    TT_FATAL(
        receivers.num_cores() <= config_.max_receivers_per_pipe,
        "DRAM pipe with sender {} has {} receivers, exceeding max_receivers_per_pipe {}",
        sender.str(),
        receivers.num_cores(),
        config_.max_receivers_per_pipe);
    TT_FATAL(
        config_.receiver_domain.contains(receivers),
        "DRAM pipe with sender {} has receivers {} outside receiver_domain {}",
        sender.str(),
        receivers.str(),
        config_.receiver_domain.str());
    TT_FATAL(!claimed_dram_senders_.contains(sender), "DRAM sender {} is already claimed by a live pipe", sender.str());
    for (const CoreCoord& receiver : corerange_to_cores(receivers)) {
        TT_FATAL(
            !claimed_.contains(receiver), "DRAM pipe receiver {} is already claimed by a live pipe", receiver.str());
    }
}

void PrefetcherPipeSpaceImpl::validate_dram_carve(CoreCoord sender, const CoreRangeSet& receivers) const {
    TT_FATAL(
        dram_sender_allocations_.contains(sender),
        "create_dram_sender_pipe: sender {} was not selected for this space",
        sender.str());
    validate_dram_pipe_geometry(sender, receivers);
}

void PrefetcherPipeSpaceImpl::validate_dram_carves(std::span<const std::pair<CoreCoord, CoreRangeSet>> pipes) const {
    TT_FATAL(
        pipes.size() <= config_.num_dram_senders,
        "DRAM pipe batch has {} senders, exceeding this space's capacity {}",
        pipes.size(),
        config_.num_dram_senders);
    std::unordered_set<CoreCoord> pending_senders;
    std::unordered_set<CoreCoord> pending_receivers;
    size_t num_unreserved_senders = 0;
    for (const auto& [sender, receivers] : pipes) {
        TT_FATAL(
            pending_senders.insert(sender).second,
            "DRAM sender {} appears in more than one pipe of the batch",
            sender.str());
        // Unlike the single-pipe form, a batch is validated before set_dram_sender_cores reserves
        // its new senders -- that is the order CreatePrefetcherPipesForTensorPrefetcher uses, so
        // that every check passes before the space is mutated. The capacity check below mirrors
        // the one set_dram_sender_cores applies.
        if (!dram_sender_allocations_.contains(sender)) {
            ++num_unreserved_senders;
        }
        validate_dram_pipe_geometry(sender, receivers);
        for (const CoreCoord& receiver : corerange_to_cores(receivers)) {
            TT_FATAL(
                pending_receivers.insert(receiver).second,
                "DRAM pipe receiver {} appears in more than one pipe of the batch",
                receiver.str());
        }
    }
    TT_FATAL(
        dram_sender_allocations_.size() + num_unreserved_senders <= config_.num_dram_senders,
        "DRAM pipe batch needs {} DRAM senders this space has not reserved, but {} of its capacity {} are already "
        "reserved. Size num_dram_senders for every sender any layout carved from this space uses.",
        num_unreserved_senders,
        dram_sender_allocations_.size(),
        config_.num_dram_senders);
}

PrefetcherPipe PrefetcherPipeSpaceImpl::create_dram_sender_pipe(
    CoreCoord sender,
    const CoreRangeSet& receivers,
    uint32_t recv_index_base,
    uint64_t tensor_prefetcher_factory_id,
    uint32_t tensor_prefetcher_factory_num_pipes) {
    validate_dram_carve(sender, receivers);
    return PrefetcherPipe(std::make_unique<PrefetcherPipeImpl>(
        *this,
        sender,
        receivers,
        dram_sender_allocations_.at(sender),
        recv_index_base,
        tensor_prefetcher_factory_id,
        tensor_prefetcher_factory_num_pipes));
}

// ---------------------------------------------------------------------------------------------
// PrefetcherPipeImpl
// ---------------------------------------------------------------------------------------------

PrefetcherPipeImpl::PrefetcherPipeImpl(
    PrefetcherPipeSpaceImpl& space, CoreCoord sender_core, const CoreRangeSet& receiver_cores) :
    space_(&space),
    identity_(PrefetcherPipeIdentity{next_prefetcher_pipe_identity.fetch_add(1, std::memory_order_relaxed)}),
    sender_core_(sender_core),
    receiver_cores_(receiver_cores),
    claimed_(true) {
    space_->validate_carve(sender_core_, receiver_cores_, nullptr);
    sender_cores_ = CoreRangeSet(CoreRange(sender_core_));
    all_cores_ = sender_cores_.merge(receiver_cores_);
    space_->claim(all_cores_, *this);
    try {
        build_config_pages();
        space_->write_pages(config_pages_);
    } catch (...) {
        space_->unclaim(all_cores_, *this);
        claimed_ = false;
        throw;
    }
}

PrefetcherPipeImpl::PrefetcherPipeImpl(
    PrefetcherPipeSpaceImpl& space,
    CoreCoord dram_sender,
    const CoreRangeSet& receiver_cores,
    std::shared_ptr<DriscL1Allocation> drisc_config_page,
    uint32_t recv_index_base,
    uint64_t tensor_prefetcher_factory_id,
    uint32_t tensor_prefetcher_factory_num_pipes) :
    space_(&space),
    identity_(PrefetcherPipeIdentity{next_prefetcher_pipe_identity.fetch_add(1, std::memory_order_relaxed)}),
    sender_core_(dram_sender),
    receiver_cores_(receiver_cores),
    all_cores_(receiver_cores),
    sender_core_type_(SenderCoreType::Dram),
    initial_entry_size_(l1_alignment_for(space_->get_device())),
    recv_index_base_(recv_index_base),
    tensor_prefetcher_factory_id_(tensor_prefetcher_factory_id),
    tensor_prefetcher_factory_num_pipes_(tensor_prefetcher_factory_num_pipes),
    drisc_config_page_(std::move(drisc_config_page)) {
    TT_FATAL(drisc_config_page_ != nullptr, "DRAM-sender PrefetcherPipe requires reserved DRISC L1");
    space_->validate_dram_carve(sender_core_, receiver_cores_);
    space_->claim(receiver_cores_, *this);
    space_->claimed_dram_senders_.insert(sender_core_);
    claimed_ = true;
    try {
        build_dram_sender_config_pages();
    } catch (...) {
        space_->unclaim(receiver_cores_, *this);
        space_->claimed_dram_senders_.erase(sender_core_);
        claimed_ = false;
        throw;
    }
}

PrefetcherPipeImpl::~PrefetcherPipeImpl() {
    if (claimed_ && space_ != nullptr) {
        space_->unclaim(all_cores_, *this);
        if (sender_core_type() == SenderCoreType::Dram) {
            space_->claimed_dram_senders_.erase(sender_core_);
        }
    }
}

const PrefetcherPipeSpaceImpl& PrefetcherPipeImpl::space() const {
    TT_FATAL(
        space_ != nullptr,
        "PrefetcherPipe (sender {}) is unusable: the PrefetcherPipeSpace it was carved from has been destroyed. A "
        "space must outlive its pipes.",
        sender_core_.str());
    return *space_;
}

void PrefetcherPipeImpl::detach_from_space() noexcept {
    space_ = nullptr;
    claimed_ = false;
}

void PrefetcherPipeImpl::build_config_pages() {
    const PrefetcherPipeSpaceImpl& space = this->space();
    const uint32_t l1_alignment = l1_alignment_for(space.get_device());
    const PrefetcherPipeConfigPageLayout& layout = space.layout();
    const uint32_t credit_lane_capacity = space.credit_lane_capacity();
    const uint32_t data_base_addr = space.buffer_address();
    const uint32_t ring_size = space.ring_size();
    const uint32_t words_per_page = layout.page_size / sizeof(uint32_t);
    const distributed::MeshDevice* device = space.get_device();
    config_pages_.clear();

    const auto receiver_vec = corerange_to_cores(receiver_cores_);
    const uint32_t num_recv = static_cast<uint32_t>(receiver_vec.size());

    // The page is laid out for the space's max receiver count; this pipe's real count goes in
    // word[1] and its XY table / credit slots occupy the head of each block.
    std::vector<uint32_t> sender_page(words_per_page, 0);
    uint32_t si = 0;
    sender_page[si++] = 1;
    sender_page[si++] = num_recv;
    sender_page[si++] = data_base_addr;
    sender_page[si++] = ring_size;
    sender_page[si++] = data_base_addr;  // word[4]: initial fifo_ptr checkpoint
    sender_page[si++] = 0;               // word[5]: applied_entry_size; set by the first program that binds
    sender_page[si++] = layout.noc_xy_offset;
    sender_page[si++] = layout.sent_offset;   // word[7]: local sent/wr block; same offset is the remote sent base
    sender_page[si++] = layout.acked_offset;  // word[8]: local acked block (receivers' NoC atomics land here)
    sender_page[si++] = layout.sent_offset;   // word[9]: receivers' SENT block; their pages share this address
    for (uint32_t ri = 0; ri < num_recv; ++ri) {
        auto phys = device->worker_core_from_logical_core(receiver_vec[ri]);
        sender_page[si++] = static_cast<uint32_t>(phys.x);
        sender_page[si++] = static_cast<uint32_t>(phys.y);
    }
    config_pages_[sender_core_] = std::move(sender_page);

    const auto sender_phys = device->worker_core_from_logical_core(sender_core_);
    for (uint32_t ri = 0; ri < num_recv; ++ri) {
        std::vector<uint32_t> receiver_page(words_per_page, 0);
        uint32_t rci = 0;
        receiver_page[rci++] = 0;
        receiver_page[rci++] = num_recv;
        receiver_page[rci++] = data_base_addr;
        receiver_page[rci++] = ring_size;
        receiver_page[rci++] = data_base_addr;
        receiver_page[rci++] = 0;
        receiver_page[rci++] = layout.noc_xy_offset;
        // This receiver's lane-0 slot in each block; device adds tid * L1_ALIGNMENT for other
        // lanes. Offsets use the allocated lane stride, not the active count. The same offsets
        // address this receiver's mirror slots on the sender page.
        const uint32_t slot = ri * credit_lane_capacity * l1_alignment;
        receiver_page[rci++] = layout.sent_offset + slot;   // word[7]: sender's NoC atomics land here
        receiver_page[rci++] = layout.acked_offset + slot;  // word[8]: local acked (cached stores)
        receiver_page[rci++] = layout.acked_offset + slot;  // word[9]: ack target, same offset on the sender page
        receiver_page[rci++] = static_cast<uint32_t>(sender_phys.x);
        receiver_page[rci++] = static_cast<uint32_t>(sender_phys.y);
        config_pages_[receiver_vec[ri]] = std::move(receiver_page);
    }
}

std::vector<CoreCoord> PrefetcherPipeImpl::dram_sender_receiver_table() const {
    return corerange_to_cores(receiver_cores_, /*max_cores=*/std::nullopt, /*row_wise=*/true);
}

void PrefetcherPipeImpl::build_dram_sender_config_pages() {
    TT_FATAL(space_->credit_lane_capacity() == 1, "DRAM-sender PrefetcherPipes support one credit lane per receiver");
    const uint32_t l1_alignment = l1_alignment_for(space_->get_device());
    const auto& layout = space_->layout();
    const auto receivers = dram_sender_receiver_table();
    const uint32_t num_receivers = static_cast<uint32_t>(receivers.size());
    const PrefetcherPipePageCommon common{
        .layout = layout,
        .l1_alignment = l1_alignment,
        .num_receivers = num_receivers,
        .data_base_addr = space_->buffer_address(),
        .ring_size = space_->ring_size(),
        // A DRISC sender has to start with a nonzero applied size before a request can re-grid it.
        .applied_entry_size = initial_entry_size_};
    const uint32_t sender_page_addr = static_cast<uint32_t>(drisc_config_page_->addr());
    const uint32_t receiver_sent_base = space_->config_address() + layout.sent_offset;

    const distributed::MeshDevice* mesh_device = space_->get_device();
    config_pages_.clear();
    std::map<distributed::MeshCoordinate, PrefetcherPipeConfigPages> device_pages;
    for (auto* target_device : mesh_device->impl().get_devices()) {
        std::vector<CoreCoord> receiver_phys;
        receiver_phys.reserve(receivers.size());
        for (const CoreCoord& receiver : receivers) {
            receiver_phys.push_back(target_device->worker_core_from_logical_core(receiver));
        }
        const auto sender_page = build_sender_config_page(common, receiver_sent_base - sender_page_addr, receiver_phys);
        write_dram_sender_l1(
            *mesh_device, target_device, sender_core_, sender_page_addr, std::as_bytes(std::span(sender_page)));

        const CoreCoord sender_virtual = target_device->virtual_core_from_logical_core(sender_core_, CoreType::DRAM);
        PrefetcherPipeConfigPages& pages = device_pages[mesh_device->get_view().find_device(target_device->id())];
        for (uint32_t r = 0; r < num_receivers; ++r) {
            const uint32_t sender_acked_slot = sender_page_addr + layout.acked_offset + r * l1_alignment;
            auto receiver_page =
                build_receiver_config_page(common, r, sender_acked_slot - space_->config_address(), sender_virtual);
            if (!config_pages_.contains(receivers[r])) {
                config_pages_.emplace(receivers[r], receiver_page);
            }
            pages.emplace(receivers[r], std::move(receiver_page));
        }
    }
    space_->write_device_pages(device_pages);
}

SenderCoreType PrefetcherPipeImpl::sender_core_type() const { return sender_core_type_; }

DeviceAddr PrefetcherPipeImpl::sender_state_drisc_l1_base() const {
    return drisc_config_page_ == nullptr ? 0 : drisc_config_page_->addr();
}

const std::vector<uint32_t>& PrefetcherPipeImpl::config_page(const CoreCoord& core) const {
    auto it = config_pages_.find(core);
    TT_FATAL(it != config_pages_.end(), "PrefetcherPipe has no host config page for core {}", core.str());
    return it->second;
}

void PrefetcherPipeImpl::validate_credit_lane_transition(uint32_t from_lanes, uint32_t num_lanes) const {
    TT_FATAL(num_lanes >= 1, "active credit lanes must be >= 1");
    TT_FATAL(
        num_lanes <= credit_lane_capacity(),
        "active credit lanes {} exceeds allocated capacity {} "
        "(Quasar spaces reserve PREFETCHER_PIPE_MAX_CREDIT_LANES per receiver slot)",
        num_lanes,
        credit_lane_capacity());
    TT_FATAL(
        num_lanes == from_lanes || from_lanes == 1,
        "PrefetcherPipe credit lanes already set to {}, cannot reprogram to {}",
        from_lanes,
        num_lanes);
}

void PrefetcherPipeImpl::set_active_credit_lanes(uint32_t num_lanes) {
    validate_credit_lane_transition(active_credit_lanes_, num_lanes);
    if (num_lanes == active_credit_lanes_) {
        return;
    }
    // Host state only. P reaches the device packed into each program's kernel-config slot
    // (build_prefetcher_pipe_config_payload), so it is ordered with the program that uses it;
    // nothing in the persistent config page is touched after carve. The one-shot 1 -> P guard
    // above stays because the persistent credit block is interpreted through P.
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
        ring_size() % entry_size == 0,
        "PrefetcherPipe with {} credit lanes requires entry_size {} to divide ring_size {}",
        num_lanes,
        entry_size,
        ring_size());
    TT_FATAL(
        (ring_size() / entry_size) % num_lanes == 0,
        "PrefetcherPipe ring holds {} entries of {} bytes, which is not a multiple of {} credit lanes",
        ring_size() / entry_size,
        entry_size,
        num_lanes);
}

// ---------------------------------------------------------------------------------------------
// Public handles
// ---------------------------------------------------------------------------------------------

PrefetcherPipe::PrefetcherPipe(std::unique_ptr<PrefetcherPipeImpl> impl) : pimpl_(std::move(impl)) {
    TT_FATAL(pimpl_ != nullptr, "PrefetcherPipe requires an implementation object");
}

PrefetcherPipe::PrefetcherPipe(PrefetcherPipe&&) noexcept = default;

// Defined here rather than in the header because assigning over a pipe destroys the
// PrefetcherPipeImpl it held, which needs the complete type.
PrefetcherPipe& PrefetcherPipe::operator=(PrefetcherPipe&&) noexcept = default;

PrefetcherPipe::~PrefetcherPipe() = default;

// A Program records a binding by pointing at the PrefetcherPipeImpl, so relocating a handle has to
// leave that object alone; holding it behind a pointer is what makes that true. Containers relocate
// on growth, and a throwing move there would strand the claim, so require the move to be noexcept.
static_assert(
    std::is_nothrow_move_constructible_v<PrefetcherPipe> && !std::is_copy_constructible_v<PrefetcherPipe>,
    "PrefetcherPipe must move without throwing and must not copy: two handles would release one claim");

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

const distributed::MeshDevice* PrefetcherPipe::get_device() const { return pimpl_->get_device(); }

SenderCoreType PrefetcherPipe::sender_core_type() const { return pimpl_->sender_core_type(); }

PrefetcherPipeIdentity PrefetcherPipe::identity() const { return pimpl_->identity(); }

uint32_t PrefetcherPipe::initial_entry_size() const { return pimpl_->initial_entry_size(); }

PrefetcherPipeSpace::PrefetcherPipeSpace(std::unique_ptr<PrefetcherPipeSpaceImpl> impl) : pimpl_(std::move(impl)) {
    TT_FATAL(pimpl_ != nullptr, "PrefetcherPipeSpace requires an implementation object");
}

PrefetcherPipeSpace::PrefetcherPipeSpace(PrefetcherPipeSpace&&) noexcept = default;

PrefetcherPipeSpace& PrefetcherPipeSpace::operator=(PrefetcherPipeSpace&&) noexcept = default;

PrefetcherPipeSpace::~PrefetcherPipeSpace() = default;

uint32_t PrefetcherPipeSpace::buffer_address() const { return pimpl_->buffer_address(); }

uint32_t PrefetcherPipeSpace::config_address() const { return pimpl_->config_address(); }

uint32_t PrefetcherPipeSpace::ring_size() const { return pimpl_->ring_size(); }

uint32_t PrefetcherPipeSpace::config_page_size() const { return pimpl_->config_page_size(); }

uint32_t PrefetcherPipeSpace::max_receivers_per_pipe() const { return pimpl_->max_receivers_per_pipe(); }

const CoreRangeSet& PrefetcherPipeSpace::sender_cores() const { return pimpl_->sender_cores(); }

uint32_t PrefetcherPipeSpace::num_dram_senders() const { return pimpl_->num_dram_senders(); }

const CoreRangeSet& PrefetcherPipeSpace::receiver_domain() const { return pimpl_->receiver_domain(); }

const CoreRangeSet& PrefetcherPipeSpace::reservation_cores() const { return pimpl_->reservation_cores(); }

CoreRangeSet PrefetcherPipeSpace::unclaimed_cores() const { return pimpl_->unclaimed_cores(); }

const distributed::MeshDevice* PrefetcherPipeSpace::get_device() const { return pimpl_->get_device(); }

PrefetcherPipe PrefetcherPipeSpace::create_pipe(CoreCoord sender, const CoreRangeSet& receivers) {
    return pimpl_->create_pipe(sender, receivers);
}

std::vector<PrefetcherPipe> PrefetcherPipeSpace::create_pipes(
    std::span<const std::pair<CoreCoord, CoreRangeSet>> pipes) {
    return pimpl_->create_pipes(pipes);
}

PrefetcherPipeL1Source PrefetcherPipeSpace::l1_source() const { return pimpl_->l1_source(); }

PrefetcherPipeSpace CreatePrefetcherPipeSpace(
    distributed::MeshDevice& device, const PrefetcherPipeSpaceConfig& config) {
    std::vector<PrefetcherPipeSpace> spaces = CreatePrefetcherPipeSpaces(device, std::span(&config, 1));
    return std::move(spaces.front());
}

std::vector<PrefetcherPipeSpace> CreatePrefetcherPipeSpaces(
    distributed::MeshDevice& device, std::span<const PrefetcherPipeSpaceConfig> configs) {
    TT_FATAL(!configs.empty(), "CreatePrefetcherPipeSpaces requires at least one config");
    std::vector<std::unique_ptr<PrefetcherPipeSpaceImpl>> impls;
    impls.reserve(configs.size());
    for (const PrefetcherPipeSpaceConfig& config : configs) {
        impls.push_back(std::make_unique<PrefetcherPipeSpaceImpl>(device, config));
    }

    // The spaces share one address per allocation, so they need one source and no core in common.
    const PrefetcherPipeL1Source source = impls.front()->l1_source();
    CoreRangeSet cores;
    uint32_t ring_size = 0;
    uint32_t config_page_size = 0;
    for (size_t i = 0; i < impls.size(); ++i) {
        const PrefetcherPipeSpaceImpl& impl = *impls[i];
        TT_FATAL(
            impl.l1_source() == source,
            "CreatePrefetcherPipeSpaces: config {} resolves to a different PrefetcherPipeL1Source than config 0; "
            "spaces that share L1 must take it from the same source",
            i);
        TT_FATAL(
            !cores.intersects(impl.reservation_cores()),
            "CreatePrefetcherPipeSpaces: config {} reserves cores {} that an earlier config also reserves; spaces that "
            "share L1 addresses must be on disjoint cores",
            i,
            impl.reservation_cores().intersection(cores).str());
        cores = cores.merge(impl.reservation_cores());
        ring_size = std::max(ring_size, impl.ring_size());
        config_page_size = std::max(config_page_size, impl.config_page_size());
    }

    auto l1 = std::make_shared<PrefetcherPipeSpaceL1>(device, source, cores, ring_size, config_page_size);
    std::vector<PrefetcherPipeSpace> spaces;
    spaces.reserve(impls.size());
    for (auto& impl : impls) {
        impl->attach_l1(l1);
        spaces.emplace_back(std::move(impl));
    }
    return spaces;
}

void set_dram_sender_cores(PrefetcherPipeSpace& space, std::span<const CoreCoord> dram_senders) {
    space.impl().set_dram_sender_cores(dram_senders);
}

std::vector<std::pair<CoreCoord, CoreRangeSet>> GetPrefetcherPipeSenderReceiverMapping(
    const std::vector<std::reference_wrapper<const PrefetcherPipe>>& pipes) {
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
    mapping.reserve(pipes.size());
    for (const PrefetcherPipe& pipe : pipes) {
        mapping.emplace_back(pipe.sender_core(), pipe.receiver_cores());
    }
    return mapping;
}

CoreRangeSet GetPrefetcherPipeReceiverCores(const std::vector<std::reference_wrapper<const PrefetcherPipe>>& pipes) {
    std::vector<CoreRange> ranges;
    for (const PrefetcherPipe& pipe : pipes) {
        const auto& receiver_ranges = pipe.receiver_cores().ranges();
        ranges.insert(ranges.end(), receiver_ranges.begin(), receiver_ranges.end());
    }
    return CoreRangeSet().merge(ranges);
}

}  // namespace tt::tt_metal::experimental
