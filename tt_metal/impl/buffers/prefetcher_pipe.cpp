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
#include "impl/buffers/prefetcher_pipe_dram_sender_internal.hpp"
#include "impl/buffers/dram_sender_topology.hpp"
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
#include <span>
#include <type_traits>
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

// Fields every config page of one pipe repeats. Bundled so the shared receiver-page builder
// takes the per-receiver differences as its only arguments.
struct PrefetcherPipePageCommon {
    PrefetcherPipeConfigPageLayout layout;
    uint32_t l1_alignment;
    uint32_t num_receivers;
    uint32_t data_base_addr;
    uint32_t ring_size;
    // word[5]: the entry size this endpoint is already sized for. A worker-sender pipe leaves it
    // 0 so the first Attach performs the resize; a DRAM-sender pipe pre-stamps its fixed size
    // because a DRAM sender never Attaches and so could never answer that resize.
    uint32_t applied_entry_size;
};

uint32_t words_per_page(const PrefetcherPipePageCommon& common) {
    return common.layout.page_size / static_cast<uint32_t>(sizeof(uint32_t));
}

// word[0] of a config page: which end of the pipe the page configures.
enum class PipeEndpoint : uint32_t { Receiver = 0, Sender = 1 };

// Words 0-6, everything both ends of a pipe agree on. Returns the index of word[7], the first
// word whose meaning depends on the endpoint.
uint32_t write_shared_config_words(
    std::vector<uint32_t>& page, const PrefetcherPipePageCommon& common, PipeEndpoint endpoint) {
    uint32_t i = 0;
    page[i++] = static_cast<uint32_t>(endpoint);
    page[i++] = common.num_receivers;
    page[i++] = common.data_base_addr;
    page[i++] = common.ring_size;
    page[i++] = common.data_base_addr;  // word[4]: initial fifo_ptr checkpoint
    page[i++] = common.applied_entry_size;
    page[i++] = common.layout.noc_xy_offset;
    return i;
}

// The sender's config page. Words 7 and 8 are this sender's own SENT and ACKED block bases, as on
// any pipe's sender page. `peer_counter_offset` is word[9]: the base of the *receivers'* SENT block
// relative to this page's own address, needed because a DRAM sender's page sits in DRISC L1 instead
// of at the receivers' page address. All of a pipe's receiver pages do share one L1 address, so
// that single base plus r*L1_ALIGNMENT reaches receiver r's slot. `receiver_noc_xy` is the
// receivers' physical coords on the device this page is destined for, in receiver order.
std::vector<uint32_t> build_sender_config_page(
    const PrefetcherPipePageCommon& common,
    uint32_t peer_counter_offset,
    const std::vector<CoreCoord>& receiver_noc_xy) {
    std::vector<uint32_t> page(words_per_page(common), 0);
    uint32_t i = write_shared_config_words(page, common, PipeEndpoint::Sender);
    page[i++] = common.layout.sent_offset;   // word[7]: local sent/wr block
    page[i++] = common.layout.acked_offset;  // word[8]: local acked block (receivers' NoC atomics)
    page[i++] = peer_counter_offset;
    for (const CoreCoord& phys : receiver_noc_xy) {
        page[i++] = static_cast<uint32_t>(phys.x);
        page[i++] = static_cast<uint32_t>(phys.y);
    }
    // The counters themselves stay zero from the zero-fill: a fresh pipe has no credits
    // outstanding, and the write cursor each counter slot carries starts at the ring base.
    return page;
}

// One receiver's config page. `peer_counter_offset` is word[9], the delta from this page's own
// address to the acked slot this receiver NOC-increments on the sender; for a DRAM sender that
// delta crosses into DRISC L1 and may wrap.
std::vector<uint32_t> build_receiver_config_page(
    const PrefetcherPipePageCommon& common,
    uint32_t receiver_index,
    uint32_t peer_counter_offset,
    CoreCoord sender_noc_xy) {
    std::vector<uint32_t> page(words_per_page(common), 0);
    uint32_t i = write_shared_config_words(page, common, PipeEndpoint::Receiver);
    // Words 7 and 8: this receiver's slot within each credit block. Offsets are page-relative and
    // every receiver page of a pipe sits at the same L1 address, which is what lets the sender
    // reach receiver r's sent slot as one base plus r*L1_ALIGNMENT. A DRAM-sender pipe is
    // single-lane, so a receiver's slot stride is one L1_ALIGNMENT.
    const uint32_t slot = receiver_index * common.l1_alignment;
    page[i++] = common.layout.sent_offset + slot;   // word[7]: sender's NoC atomics land here
    page[i++] = common.layout.acked_offset + slot;  // word[8]: local acked (cached stores)
    page[i++] = peer_counter_offset;
    page[i++] = static_cast<uint32_t>(sender_noc_xy.x);
    page[i++] = static_cast<uint32_t>(sender_noc_xy.y);
    return page;
}

// Receiver order, page layout, and the fields every config page of one pipe repeats: the preamble
// each of the two DRAM-sender page builders below would otherwise recompute.
//
// Receiver order is the pipe's slab-numbering contract, and the two sender flavours number
// differently. A DRAM-sender pipe traverses row-wise, matching build_dram_sender_mapping's
// ceil/floor receiver split and the GlobalCircularBuffer receiver tables, so a tensor laid out for
// one DRAM-sender transport is laid out for the other. A worker-sender pipe keeps
// corerange_to_cores' default order, which is what its receivers were created with.
struct PrefetcherPipePageContext {
    std::vector<CoreCoord> receivers;
    PrefetcherPipePageCommon common;
};

PrefetcherPipePageContext build_page_context(
    const CoreRangeSet& receiver_cores,
    SenderCoreType sender_core_type,
    uint32_t l1_alignment,
    uint32_t data_base_addr,
    uint32_t ring_size,
    uint32_t applied_entry_size) {
    auto receivers =
        corerange_to_cores(receiver_cores, /*max_cores=*/std::nullopt, sender_core_type == SenderCoreType::Dram);
    const auto num_recv = static_cast<uint32_t>(receivers.size());
    return PrefetcherPipePageContext{
        .receivers = std::move(receivers),
        .common =
            PrefetcherPipePageCommon{
                // A DRAM sender is never Attached and so can never be handed more than one
                // pipe-consumer thread: one credit lane per receiver.
                .layout = compute_prefetcher_pipe_config_page_layout(num_recv, /*num_credit_lanes=*/1, l1_alignment),
                .l1_alignment = l1_alignment,
                .num_receivers = num_recv,
                .data_base_addr = data_base_addr,
                .ring_size = ring_size,
                .applied_entry_size = applied_entry_size,
            },
    };
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
        (device != nullptr && device->arch() == tt::ARCH::QUASAR) ? PREFETCHER_PIPE_MAX_CREDIT_LANES : 1u),
    sender_core_type_(SenderCoreType::Worker) {
    initialize_prefetcher_pipe(device, sender_core, receiver_cores_, sender_cores_, all_cores_);
    try {
        setup_buffers(buffer_type);
    } catch (...) {
        release_allocations();
        throw;
    }
}

PrefetcherPipeImpl::PrefetcherPipeImpl(
    distributed::MeshDevice* mesh_device,
    CoreCoord dram_sender_logical,
    const CoreRangeSet& receiver_cores,
    uint32_t ring_size,
    uint32_t initial_entry_size,
    uint32_t recv_index_base,
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
    sender_core_type_(SenderCoreType::Dram),
    initial_entry_size_(initial_entry_size),
    recv_index_base_(recv_index_base) {
    TT_FATAL(mesh_device != nullptr, "DRAM-sender PrefetcherPipe requires a non-null MeshDevice");
    const auto& hal = MetalContext::instance(mesh_device->impl().get_context_id()).hal();
    TT_FATAL(
        hal.has_programmable_core_type(HalProgrammableCoreType::DRAM),
        "DRAM-sender PrefetcherPipe requires programmable DRAM cores, which auto-enable on Blackhole with firmware "
        ">= 19.12.0.0");
    TT_FATAL(receiver_cores.num_cores() > 0, "DRAM-sender PrefetcherPipe requires at least one receiver");
    TT_FATAL(initial_entry_size > 0, "DRAM-sender PrefetcherPipe entry size must be > 0");
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    TT_FATAL(
        initial_entry_size % l1_alignment == 0,
        "DRAM-sender PrefetcherPipe entry size {} must be a multiple of L1_ALIGNMENT {}",
        initial_entry_size,
        l1_alignment);
    TT_FATAL(
        initial_entry_size <= ring_size,
        "DRAM-sender PrefetcherPipe ring size {} must hold at least one {}-byte entry",
        ring_size,
        initial_entry_size);
    try {
        setup_buffers(buffer_type);
    } catch (...) {
        release_allocations();
        throw;
    }
}

void PrefetcherPipeImpl::set_config_page_geometry(uint32_t page_size, uint32_t credit_reset_offset) {
    config_page_size_ = page_size;
    credit_reset_offset_ = credit_reset_offset;
    credit_reset_size_ = page_size - credit_reset_offset;
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

std::unordered_map<CoreCoord, std::vector<uint32_t>> PrefetcherPipeImpl::build_dram_sender_receiver_config_pages(
    IDevice* target_device) const {
    TT_FATAL(config_address_ != 0, "PrefetcherPipe config allocation must exist before building pages");
    TT_FATAL(data_address_ != 0, "PrefetcherPipe data address must be set before building config pages");

    const uint32_t l1_alignment =
        MetalContext::instance(extract_context_id(device_)).hal().get_alignment(HalMemType::L1);
    const auto [receiver_vec, common] = build_page_context(
        receiver_cores_, sender_core_type_, l1_alignment, data_address_, ring_size_, initial_entry_size_);
    const auto& layout = common.layout;
    const uint32_t num_recv = common.num_receivers;

    // Base of the sender's ACKED block, inside its config page in DRISC L1: where this pipe's
    // receivers aim their ack atomics.
    const uint32_t drisc_acked_base = static_cast<uint32_t>(drisc_config_page_alloc_->addr()) + layout.acked_offset;
    // The receiver's ack NOC-inc lands on the DRAM core, so it needs that core's virtual coord on
    // this device rather than a worker coord.
    const auto sender_virtual = target_device->virtual_core_from_logical_core(sender_core_, CoreType::DRAM);

    std::unordered_map<CoreCoord, std::vector<uint32_t>> pages;
    pages.reserve(num_recv);
    for (uint32_t ri = 0; ri < num_recv; ++ri) {
        // setup_prefetcher_pipe_interface adds word[9] to the receiver's own page address, so
        // store the difference between the two L1 address spaces. It may wrap; the device side
        // does the same uint32 arithmetic.
        const uint32_t drisc_acked_slot = drisc_acked_base + ri * l1_alignment;
        pages[receiver_vec[ri]] =
            build_receiver_config_page(common, ri, drisc_acked_slot - config_address_, sender_virtual);
    }
    return pages;
}

void PrefetcherPipeImpl::initialize_dram_sender_config_page() {
    auto& metal_ctx = MetalContext::instance(device_->impl().get_context_id());
    const uint32_t l1_alignment = metal_ctx.hal().get_alignment(HalMemType::L1);
    // A DRAM sender never Attaches, so it could never answer a resize; pre-stamp its size.
    const auto [receiver_vec, common] = build_page_context(
        receiver_cores_, sender_core_type_, l1_alignment, data_address_, ring_size_, initial_entry_size_);
    const auto& layout = common.layout;
    const uint32_t num_recv = common.num_receivers;
    TT_FATAL(
        credit_lane_capacity_ == 1,
        "DRAM-sender PrefetcherPipe reserved {} credit lanes per receiver: the DRISC sender helpers address one slot "
        "per receiver, so Quasar's multi-lane pipe consumers are not supported from a DRAM sender",
        credit_lane_capacity_);
    set_config_page_geometry(layout.page_size, layout.sent_offset);

    // Reserved on this sender's core alone: a pipe on another bank can hold the same offset, so a
    // set of one-sender pipes costs the small DRISC zone one page rather than one page per pipe.
    // Page-relative block offsets are only line-aligned if the page itself is, and the page's own
    // counters are NOC-atomic targets -- hence the larger of the two alignments.
    drisc_config_page_alloc_ = device_->impl().drisc_l1_arena().allocate_on(
        sender_core_, layout.page_size, std::max(l1_alignment, PREFETCHER_PIPE_CREDIT_BLOCK_ALIGN));
    const auto config_page_addr = static_cast<uint32_t>(drisc_config_page_alloc_->addr());

    // word[9] on a DRAM sender's page is the base of the *receivers'* SENT block. All of this
    // pipe's receiver pages share one L1 address, so a single base plus r*L1_ALIGNMENT reaches
    // receiver r's own slot. The DRISC sender helpers add the stored delta to the sender's own page
    // address and then pack the result into 24 bits.
    const uint32_t receiver_sent_base = config_address_ + layout.sent_offset;
    TT_FATAL(
        (receiver_sent_base & ~dev_msgs::REMOTE_CB_PACKED_ADDR_MASK) == 0,
        "Receiver counter base 0x{:x} does not fit the packed remote-pointer field (mask 0x{:x}) used for sender "
        "credits",
        receiver_sent_base,
        dev_msgs::REMOTE_CB_PACKED_ADDR_MASK);

    std::vector<CoreCoord> receiver_phys(num_recv);
    for (IDevice* dev : device_->get_devices()) {
        // The receivers' worker coords are resolved per device: harvesting can place them
        // differently on each.
        for (uint32_t r = 0; r < num_recv; ++r) {
            receiver_phys[r] = dev->worker_core_from_logical_core(receiver_vec[r]);
        }
        const std::vector<uint32_t> page =
            build_sender_config_page(common, receiver_sent_base - config_page_addr, receiver_phys);
        write_dram_sender_l1(*device_, dev, sender_core_, config_page_addr, std::as_bytes(std::span(page)));
    }
}

void PrefetcherPipeImpl::write_config_to_device() {
    TT_FATAL(device_ != nullptr, "PrefetcherPipe device cannot be null");
    // Devices outermost: a DRAM-sender pipe's receiver pages name the sender's virtual DRAM coord,
    // which DRAM harvesting can place differently on each device, so they are rebuilt per device.
    // They stay local to this call -- config_pages_ describes every device or nothing.
    const bool dram_sender = sender_core_type() == SenderCoreType::Dram;
    for (IDevice* target_device : device_->get_devices()) {
        std::unordered_map<CoreCoord, std::vector<uint32_t>> per_device_pages;
        if (dram_sender) {
            per_device_pages = build_dram_sender_receiver_config_pages(target_device);
        }
        for (const auto& [core, page] : dram_sender ? per_device_pages : config_pages_) {
            const auto page_bytes =
                std::span(reinterpret_cast<const uint8_t*>(page.data()), page.size() * sizeof(uint32_t));
            TT_FATAL(
                tt_metal::detail::WriteToDeviceL1(target_device, core, config_address_, page_bytes),
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
    const DeviceAddr persistent_end =
        std::max(data_allocation.address + data_allocation.size, config_allocation.address + config_allocation.size);
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
    if (sender_core_type() == SenderCoreType::Dram) {
        // Receiver pages aim their ack at a slot inside the sender's DRISC L1 config page, so that
        // page has to be placed and stamped before write_config_to_device builds them.
        initialize_dram_sender_config_page();
    } else {
        build_config_pages();
    }
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

PrefetcherPipe::PrefetcherPipe(std::unique_ptr<PrefetcherPipeImpl> impl) : pimpl_(std::move(impl)) {}

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

SenderCoreType PrefetcherPipe::sender_core_type() const { return pimpl_->sender_core_type(); }

uint32_t PrefetcherPipe::initial_entry_size() const { return pimpl_->initial_entry_size(); }

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
    // An entry size the ring does not divide leaves a trailing gap that holds no entry. Both
    // endpoints stop at the page-aligned usable limit and credit the gap at the wrap, so the only
    // requirement is that the ring hold an entry at all.
    TT_FATAL(
        entry_size > 0 && entry_size <= prefetcher_pipe.ring_size(),
        "AttachPrefetcherPipe entry size {} must be greater than zero and fit the pipe's {} B ring",
        entry_size,
        prefetcher_pipe.ring_size());
    return program.impl().add_prefetcher_pipe_attachment(
        prefetcher_pipe.impl(), cores, entry_size, num_pipe_consumer_threads);
}

std::vector<uint8_t> AttachPrefetcherPipes(
    Program& program, const std::vector<std::shared_ptr<PrefetcherPipe>>& pipes, uint32_t entry_size) {
    std::vector<uint8_t> pipe_ids;
    pipe_ids.reserve(pipes.size());
    for (const auto& pipe : pipes) {
        TT_FATAL(pipe != nullptr, "AttachPrefetcherPipes was given a null pipe at index {}", pipe_ids.size());
        pipe_ids.push_back(AttachPrefetcherPipe(program, *pipe, pipe->receiver_cores(), entry_size));
    }
    return pipe_ids;
}

std::vector<std::pair<CoreCoord, CoreRangeSet>> prefetcher_pipe_sender_receiver_mapping(
    const std::vector<std::shared_ptr<PrefetcherPipe>>& pipes) {
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
    mapping.reserve(pipes.size());
    for (const auto& pipe : pipes) {
        TT_FATAL(pipe != nullptr, "PrefetcherPipe list holds a null pipe at index {}", mapping.size());
        mapping.emplace_back(pipe->sender_core(), pipe->receiver_cores());
    }
    return mapping;
}

CoreRangeSet prefetcher_pipe_receiver_cores(const std::vector<std::shared_ptr<PrefetcherPipe>>& pipes) {
    // One merge over every pipe's ranges: CoreRangeSet::merge rasterizes the whole bounding box, so
    // folding pipe by pipe would redo that work per pipe.
    std::vector<CoreRange> ranges;
    for (const auto& pipe : pipes) {
        TT_FATAL(pipe != nullptr, "PrefetcherPipe list holds a null pipe");
        const auto& receivers = pipe->receiver_cores().ranges();
        ranges.insert(ranges.end(), receivers.begin(), receivers.end());
    }
    return CoreRangeSet().merge(ranges);
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
    // Checked here as well as in register_prefetcher_pipe_relay_dfb below so a bad depth fails
    // before CreateDataflowBuffer leaves a stray DFB in the program. Floor division: see the note
    // there on the trailing gap an entry size that does not divide the ring leaves behind.
    TT_FATAL(
        config.num_entries == pipe.ring_size() / config.entry_size,
        "CreatePrefetcherPipeRelayDataflowBuffer: depth {} must equal the whole entries the ring holds, ring_size / "
        "entry_size = {} / {} = {}",
        config.num_entries,
        pipe.ring_size(),
        config.entry_size,
        pipe.ring_size() / config.entry_size);

    auto relay_config = config;
    relay_config.borrows_memory = true;
    relay_config.is_relay = true;
    const uint32_t relay_dfb_id = dfb::CreateDataflowBuffer(program, receiver_cores, relay_config);
    // register_prefetcher_pipe_relay_dfb programs active credit lanes from num_producers.
    program.impl().register_prefetcher_pipe_relay_dfb(receiver_cores, prefetcher_pipe_id, relay_dfb_id);
    return relay_dfb_id;
}

// ---- DRAM-sender extension -------------------------------------------------------------------
// PrefetcherPipeDramSenderInternals is the only thing that names PrefetcherPipeImpl's private
// DRAM-sender constructor and state; its members are defined here so neither the impl header nor
// the factory has to spell out the friendship.

namespace prefetcher_pipe_dram_sender {

std::shared_ptr<PrefetcherPipe> PrefetcherPipeDramSenderInternals::make_dram_sender(
    distributed::MeshDevice* mesh_device,
    CoreCoord dram_sender_logical,
    const CoreRangeSet& receiver_cores,
    uint32_t ring_size,
    uint32_t initial_entry_size,
    uint32_t recv_index_base,
    BufferType buffer_type) {
    // `new` rather than make_unique: only this friend may name the private constructor.
    std::unique_ptr<PrefetcherPipeImpl> impl(new PrefetcherPipeImpl(
        mesh_device,
        dram_sender_logical,
        receiver_cores,
        ring_size,
        initial_entry_size,
        recv_index_base,
        buffer_type,
        PrefetcherPipeImpl::DramSenderTag{}));
    // shared_ptr rather than a value: PrefetcherPipe is not copyable, and callers hold a list of
    // them.
    return std::make_shared<PrefetcherPipe>(std::move(impl));
}

DeviceAddr PrefetcherPipeDramSenderInternals::sender_state_drisc_l1_base(const PrefetcherPipe& pipe) {
    const auto& alloc = pipe.impl().drisc_config_page_alloc_;
    return alloc == nullptr ? 0 : alloc->addr();
}

}  // namespace prefetcher_pipe_dram_sender

}  // namespace tt::tt_metal::experimental
