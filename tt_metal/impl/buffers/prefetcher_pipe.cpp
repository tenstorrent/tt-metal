// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <buffer_types.hpp>
#include <core_coord.hpp>
#include <device.hpp>
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "impl/allocator/allocator.hpp"
#include "impl/buffers/prefetcher_pipe_dram_sender_internal.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/context_types.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/api/tt-metalium/hal_types.hpp"
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt_align.hpp>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "distributed/mesh_device_impl.hpp"
#include "hostdev/remote_dfb_config_layout.h"
#include "llrt/hal/generated/dev_msgs.hpp"
#include "llrt/tt_cluster.hpp"
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

// A config page is the 9-word remote-DFB header, then a 2-word NOC XY entry per receiver, then an
// L1-aligned entries_sent/entries_acked pair per receiver (the pairs are NOC-atomic targets, hence
// the alignment).
PrefetcherPipeConfigPageLayout compute_prefetcher_pipe_config_page_layout(
    uint32_t num_receivers, uint32_t l1_alignment) {
    const uint32_t noc_xy_offset = prefetcher_pipe_noc_xy_byte_offset();
    const uint32_t counters_offset =
        tt::align(noc_xy_offset + 2 * num_receivers * static_cast<uint32_t>(sizeof(uint32_t)), l1_alignment);
    const uint32_t page_size = counters_offset + 2 * num_receivers * l1_alignment;
    return PrefetcherPipeConfigPageLayout{
        .noc_xy_offset = noc_xy_offset, .counters_offset = counters_offset, .page_size = page_size};
}

// Fields every config page of one pipe repeats. Bundled so the shared receiver-page builder
// takes the per-receiver differences as its only arguments.
struct PrefetcherPipePageCommon {
    PrefetcherPipeConfigPageLayout layout;
    uint32_t words_per_page;
    uint32_t l1_alignment;
    uint32_t num_receivers;
    uint32_t data_base_addr;
    uint32_t ring_size;
    // word[5]: the entry size this endpoint is already sized for. A worker-sender pipe leaves it
    // 0 so the first Attach performs the resize; a DRAM-sender pipe pre-stamps its fixed size
    // because a DRAM sender never Attaches and so could never answer that resize.
    uint32_t applied_entry_size;
};

// One receiver's config page. `pages_acked_offset` is word[8], the delta from this page's own
// address to the peer counter slot this receiver NOC-increments; for a DRAM sender that delta
// crosses into DRISC L1 and may wrap.
std::vector<uint32_t> build_receiver_config_page(
    const PrefetcherPipePageCommon& common,
    uint32_t receiver_index,
    uint32_t pages_acked_offset,
    CoreCoord sender_noc_xy) {
    std::vector<uint32_t> page(common.words_per_page, 0);
    uint32_t i = 0;
    page[i++] = 0;  // word[0]: is_sender
    page[i++] = common.num_receivers;
    page[i++] = common.data_base_addr;
    page[i++] = common.ring_size;
    page[i++] = common.data_base_addr;  // word[4]: initial fifo_ptr checkpoint
    page[i++] = common.applied_entry_size;
    page[i++] = common.layout.noc_xy_offset;
    // word[7]: this receiver's own counter pair (sent at +0, acked at +L1_ALIGNMENT). Offsets are
    // page-relative and every receiver page of a pipe sits at the same L1 address, which is what
    // lets the sender reach receiver r's slot as remote_counters_base + 2*r*L1_ALIGNMENT.
    page[i++] = common.layout.counters_offset + 2 * receiver_index * common.l1_alignment;
    page[i++] = pages_acked_offset;
    page[i++] = static_cast<uint32_t>(sender_noc_xy.x);
    page[i++] = static_cast<uint32_t>(sender_noc_xy.y);
    return page;
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

PrefetcherPipe::PrefetcherPipe(
    distributed::MeshDevice* mesh_device,
    CoreCoord dram_sender_logical,
    const CoreRangeSet& receiver_cores,
    uint32_t ring_size,
    uint32_t fixed_entry_size,
    BufferType buffer_type,
    DramSenderTag) :
    device_(mesh_device),
    sender_core_(dram_sender_logical),
    receiver_cores_(receiver_cores),
    // A DRAM sender is never Attached, so it contributes no core to any Program: leaving
    // sender_cores_ empty is what makes the Attach role-completeness check pass on the receiver
    // set alone, and it keeps DRAM-logical coords out of a set of worker coords.
    all_cores_(receiver_cores),
    ring_size_(ring_size),
    sender_core_type_value_(static_cast<uint8_t>(SenderCoreType::Dram)),
    fixed_entry_size_(fixed_entry_size) {
    TT_FATAL(mesh_device != nullptr, "DRAM-sender PrefetcherPipe requires a non-null MeshDevice");
    const auto& hal = MetalContext::instance(mesh_device->impl().get_context_id()).hal();
    TT_FATAL(
        hal.has_programmable_core_type(HalProgrammableCoreType::DRAM),
        "DRAM-sender PrefetcherPipe requires programmable DRAM cores, which auto-enable on Blackhole with firmware "
        ">= 19.12.0.0");
    TT_FATAL(receiver_cores.num_cores() > 0, "DRAM-sender PrefetcherPipe requires at least one receiver");
    TT_FATAL(fixed_entry_size > 0, "DRAM-sender PrefetcherPipe entry size must be > 0");
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    TT_FATAL(
        fixed_entry_size % l1_alignment == 0,
        "DRAM-sender PrefetcherPipe entry size {} must be a multiple of L1_ALIGNMENT {}",
        fixed_entry_size,
        l1_alignment);
    TT_FATAL(
        ring_size % fixed_entry_size == 0,
        "DRAM-sender PrefetcherPipe ring size {} must be a whole number of {}-byte entries; a partial trailing entry "
        "would leave the sender's derived write cursor and the receiver's read cursor on different wrap points",
        ring_size,
        fixed_entry_size);
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

    const PrefetcherPipePageCommon common{
        .layout = layout,
        .words_per_page = words_per_page,
        .l1_alignment = l1_alignment,
        .num_receivers = num_recv,
        .data_base_addr = data_base_addr,
        .ring_size = ring_size_,
        .applied_entry_size = 0,  // set by the first Attach
    };
    const auto sender_phys = device_->worker_core_from_logical_core(sender_core_);
    for (uint32_t ri = 0; ri < num_recv; ++ri) {
        // Sender and receiver pages share one L1 address, so a receiver's ack target is the
        // matching slot in its own page.
        const uint32_t pages_acked_offset = layout.counters_offset + 2 * ri * l1_alignment + l1_alignment;
        config_pages_[receiver_vec[ri]] = build_receiver_config_page(common, ri, pages_acked_offset, sender_phys);
    }
}

void PrefetcherPipe::build_dram_sender_receiver_config_pages(IDevice* target_device) {
    TT_FATAL(config_address_ != 0, "PrefetcherPipe config allocation must exist before building pages");
    TT_FATAL(data_address_ != 0, "PrefetcherPipe data address must be set before building config pages");

    const uint32_t l1_alignment =
        MetalContext::instance(extract_context_id(device_)).hal().get_alignment(HalMemType::L1);
    const auto receiver_vec = corerange_to_cores(receiver_cores_, /*max_cores=*/std::nullopt, /*row_wise=*/true);
    const uint32_t num_recv = static_cast<uint32_t>(receiver_vec.size());
    const auto layout = compute_prefetcher_pipe_config_page_layout(num_recv, l1_alignment);

    const PrefetcherPipePageCommon common{
        .layout = layout,
        .words_per_page = config_page_size_ / static_cast<uint32_t>(sizeof(uint32_t)),
        .l1_alignment = l1_alignment,
        .num_receivers = num_recv,
        .data_base_addr = data_address_,
        .ring_size = ring_size_,
        .applied_entry_size = fixed_entry_size_,
    };

    // Base of the sender's counter pairs, inside its config page in DRISC L1.
    const uint32_t drisc_counters_base =
        static_cast<uint32_t>(drisc_config_page_alloc_->addr()) + layout.counters_offset;
    // The receiver's ack NOC-inc lands on the DRAM core, so it needs that core's virtual coord on
    // this device rather than a worker coord.
    const auto sender_virtual = target_device->virtual_core_from_logical_core(sender_core_, CoreType::DRAM);

    config_pages_.clear();
    for (uint32_t ri = 0; ri < num_recv; ++ri) {
        // setup_prefetcher_pipe_interface adds word[8] to the receiver's own page address, so
        // store the difference between the two L1 address spaces. It may wrap; the device side
        // does the same uint32 arithmetic.
        const uint32_t drisc_acked_slot = drisc_counters_base + 2 * ri * l1_alignment + l1_alignment;
        config_pages_[receiver_vec[ri]] =
            build_receiver_config_page(common, ri, drisc_acked_slot - config_address_, sender_virtual);
    }
}

void PrefetcherPipe::initialize_dram_sender_config_page() {
    auto& metal_ctx = MetalContext::instance(device_->impl().get_context_id());
    const uint32_t l1_alignment = metal_ctx.hal().get_alignment(HalMemType::L1);
    const auto receiver_vec = corerange_to_cores(receiver_cores_, /*max_cores=*/std::nullopt, /*row_wise=*/true);
    const uint32_t num_recv = static_cast<uint32_t>(receiver_vec.size());
    const auto layout = compute_prefetcher_pipe_config_page_layout(num_recv, l1_alignment);
    config_page_size_ = layout.page_size;
    credit_reset_offset_ = layout.counters_offset;
    credit_reset_size_ = config_page_size_ - credit_reset_offset_;

    // Reserved on this sender's core alone: a pipe on another bank can hold the same offset, so a
    // set of one-sender pipes costs the small DRISC zone one page rather than one page per pipe.
    // The page's own counters are NOC-atomic targets, hence the L1 alignment.
    drisc_config_page_alloc_ =
        device_->impl().drisc_l1_arena().allocate_on(sender_core_, layout.page_size, l1_alignment);
    const auto config_page_addr = static_cast<uint32_t>(drisc_config_page_alloc_->addr());

    // word[8] on a sender page is the base of the *receivers'* counter pairs. All of this pipe's
    // receiver pages share one L1 address, so a single base plus 2*r*L1_ALIGNMENT reaches receiver
    // r's own slot. setup_prefetcher_pipe_interface adds the stored delta to the sender's own page
    // address and then packs the result into 24 bits.
    const uint32_t receiver_counters_base = config_address_ + layout.counters_offset;
    TT_FATAL(
        (receiver_counters_base & ~dev_msgs::REMOTE_CB_PACKED_ADDR_MASK) == 0,
        "Receiver counter base 0x{:x} does not fit the packed remote-pointer field (mask 0x{:x}) used for sender "
        "credits",
        receiver_counters_base,
        dev_msgs::REMOTE_CB_PACKED_ADDR_MASK);

    std::vector<uint8_t> page_bytes(layout.page_size, 0);
    auto* page = reinterpret_cast<uint32_t*>(page_bytes.data());
    uint32_t i = 0;
    page[i++] = 1;  // word[0]: is_sender
    page[i++] = num_recv;
    page[i++] = data_address_;
    page[i++] = ring_size_;
    page[i++] = data_address_;      // word[4]: initial fifo_ptr checkpoint (unused by this sender)
    page[i++] = fixed_entry_size_;  // word[5]: applied_entry_size
    page[i++] = layout.noc_xy_offset;
    page[i++] = layout.counters_offset;                     // word[7]: local counter pairs, in DRISC L1
    page[i++] = receiver_counters_base - config_page_addr;  // word[8]: remote (receiver) base
    // The counters themselves stay zero from the page's zero-fill: a fresh pipe has no credits
    // outstanding, and every receiver's derived write cursor starts at the ring base.

    auto* noc_xy_words = reinterpret_cast<uint32_t*>(page_bytes.data() + layout.noc_xy_offset);
    const uint64_t write_addr =
        metal_ctx.hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM) + static_cast<uint64_t>(config_page_addr);
    auto& cluster = metal_ctx.get_cluster();
    for (IDevice* dev : device_->get_devices()) {
        // Both the receivers' worker coords and the sender's DRAM coord are resolved per device:
        // harvesting can place them differently on each.
        for (uint32_t r = 0; r < num_recv; ++r) {
            const CoreCoord receiver_phys = dev->worker_core_from_logical_core(receiver_vec[r]);
            noc_xy_words[2 * r + 0] = static_cast<uint32_t>(receiver_phys.x);
            noc_xy_words[2 * r + 1] = static_cast<uint32_t>(receiver_phys.y);
        }
        const CoreCoord virtual_core = dev->virtual_core_from_logical_core(sender_core_, CoreType::DRAM);
        cluster.write_core(
            dev->id(),
            tt_cxy_pair(dev->id(), virtual_core),
            std::span<const uint8_t>(page_bytes.data(), page_bytes.size()),
            write_addr);
    }
}

void PrefetcherPipe::write_config_to_device() {
    TT_FATAL(device_ != nullptr, "PrefetcherPipe device cannot be null");
    // Devices outermost: a DRAM-sender pipe's receiver pages name the sender's virtual DRAM coord,
    // which DRAM harvesting can place differently on each device, so they are rebuilt per device.
    const bool dram_sender = sender_core_type() == SenderCoreType::Dram;
    for (IDevice* target_device : device_->get_devices()) {
        if (dram_sender) {
            build_dram_sender_receiver_config_pages(target_device);
        }
        for (const auto& [core, page] : config_pages_) {
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
    if (sender_core_type() == SenderCoreType::Dram) {
        // Receiver pages aim their ack at a slot inside the sender's DRISC L1 config page, so that
        // page has to be placed and stamped before write_config_to_device builds them.
        initialize_dram_sender_config_page();
    } else {
        build_config_pages();
    }
    write_config_to_device();
}

PrefetcherPipe::~PrefetcherPipe() { release_allocations(); }

void PrefetcherPipe::release_allocations() noexcept {
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

SenderCoreType PrefetcherPipe::sender_core_type() const { return static_cast<SenderCoreType>(sender_core_type_value_); }

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
    // A DRAM sender never Attaches, so it cannot take part in the resize handshake a differing
    // entry size starts: the receivers would spin forever on pad credits nobody publishes. Reject
    // it here rather than letting the program hang.
    TT_FATAL(
        prefetcher_pipe.sender_core_type() != SenderCoreType::Dram || entry_size == prefetcher_pipe.fixed_entry_size(),
        "AttachPrefetcherPipe entry size {} does not match the DRAM-sender pipe's fixed entry size {}. A DRAM sender "
        "is not dispatched to and cannot answer a receiver-side resize, so every Attach must use the size the pipe "
        "was created with",
        entry_size,
        prefetcher_pipe.fixed_entry_size());
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

// ---- DRAM-sender extension -------------------------------------------------------------------
// PrefetcherPipeDramSenderInternals is the only thing that names PrefetcherPipe's private
// DRAM-sender constructor and state; its members are defined here so neither the impl header nor
// the factory has to spell out the friendship.

namespace prefetcher_pipe_dram_sender {

std::shared_ptr<PrefetcherPipe> PrefetcherPipeDramSenderInternals::make_dram_sender(
    distributed::MeshDevice* mesh_device,
    CoreCoord dram_sender_logical,
    const CoreRangeSet& receiver_cores,
    uint32_t ring_size,
    uint32_t fixed_entry_size,
    BufferType buffer_type) {
    // shared_ptr rather than a value: PrefetcherPipe is neither copyable nor movable, and callers
    // hold a list of them.
    return std::shared_ptr<PrefetcherPipe>(new PrefetcherPipe(
        mesh_device,
        dram_sender_logical,
        receiver_cores,
        ring_size,
        fixed_entry_size,
        buffer_type,
        PrefetcherPipe::DramSenderTag{}));
}

DeviceAddr PrefetcherPipeDramSenderInternals::sender_state_drisc_l1_base(const PrefetcherPipe& pipe) {
    return pipe.drisc_config_page_alloc_ == nullptr ? 0 : pipe.drisc_config_page_alloc_->addr();
}

}  // namespace prefetcher_pipe_dram_sender

}  // namespace tt::tt_metal::experimental
