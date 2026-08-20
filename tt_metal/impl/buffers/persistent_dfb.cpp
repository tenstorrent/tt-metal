// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <buffer.hpp>
#include <buffer_types.hpp>
#include <core_coord.hpp>
#include <device.hpp>
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/buffers/persistent_dfb_dram_sender_internal.hpp"
#include "impl/buffers/persistent_dfb_dram_sender_state.hpp"
#include "impl/dataflow_buffer/persistent_dfb.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/context_types.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/api/tt-metalium/hal_types.hpp"
#include <tt-metalium/experimental/persistent_dfb.hpp>
#include <tt_align.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <variant>
#include <vector>

#include "distributed.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "llrt/hal/generated/dev_msgs.hpp"
#include "hostdev/remote_dfb_config_layout.h"
#include "llrt/tt_cluster.hpp"
#include "mesh_buffer.hpp"
#include "mesh_device.hpp"

namespace tt::tt_metal::experimental {

namespace {

void initialize_persistent_dfb(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    CoreRangeSet& sender_cores_out,
    CoreRangeSet& receiver_cores_out,
    CoreRangeSet& all_cores_out,
    uint32_t& max_num_receivers_per_sender_out) {
    TT_FATAL(device != nullptr, "Device cannot be null");

    const uint32_t num_sender_cores = sender_receiver_mapping.size();
    TT_FATAL(num_sender_cores > 0, "At least one sender required");

    uint32_t num_receiver_cores = 0;
    uint32_t max_receivers = 0;
    std::vector<CoreRange> sender_ranges;
    sender_ranges.reserve(num_sender_cores);

    for (const auto& [sender_core, receiver_set] : sender_receiver_mapping) {
        const uint32_t n = receiver_set.num_cores();
        TT_FATAL(n > 0, "Sender core {} must have a non-empty receiver set", sender_core.str());
        num_receiver_cores += n;
        max_receivers = std::max(max_receivers, n);
        sender_ranges.emplace_back(sender_core);
        receiver_cores_out = receiver_cores_out.merge(receiver_set);
    }

    sender_cores_out = CoreRangeSet(sender_ranges);
    TT_FATAL(num_sender_cores == sender_cores_out.num_cores(), "Duplicate sender cores in sender_receiver_mapping");
    TT_FATAL(
        num_receiver_cores == receiver_cores_out.num_cores(),
        "Duplicate receiver cores detected across sender groups (receiver sets must be disjoint)");

    all_cores_out = sender_cores_out.merge(receiver_cores_out);
    TT_FATAL(
        all_cores_out.num_cores() == num_sender_cores + num_receiver_cores,
        "Sender and receiver core sets must be disjoint");

    max_num_receivers_per_sender_out = max_receivers;
}

void validate_entry_geometry(IDevice* device, uint32_t entry_size, uint32_t num_entries, BufferType buffer_type) {
    const auto context_id = extract_context_id(device);
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    TT_FATAL(entry_size > 0, "entry_size must be > 0");
    TT_FATAL(
        entry_size % l1_alignment == 0,
        "entry_size {} must be a multiple of L1_ALIGNMENT {}",
        entry_size,
        l1_alignment);
    TT_FATAL(num_entries > 0, "num_entries must be > 0");
    TT_FATAL(
        buffer_type == BufferType::L1 || buffer_type == BufferType::L1_SMALL,
        "PersistentDFB can only use L1 buffer types");
}

struct PersistentConfigPageLayout {
    uint32_t noc_xy_offset;
    uint32_t counters_offset;
    uint32_t page_size;
};

// DRAM-sender variant of initialize_persistent_dfb. Senders are DRAM cores, so their logical
// coords live in a different space than the worker receivers': they must not be merged into
// all_cores_ (which backs the receiver-sharded ring and config buffers), and a sender/receiver
// disjointness check across the two spaces would be comparing unrelated coordinates.
void initialize_dram_sender_persistent_dfb(
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    CoreRangeSet& sender_cores_out,
    CoreRangeSet& receiver_cores_out,
    CoreRangeSet& all_cores_out,
    uint32_t& max_num_receivers_per_sender_out) {
    const uint32_t num_sender_cores = sender_receiver_mapping.size();
    TT_FATAL(num_sender_cores > 0, "At least one DRAM sender required");

    uint32_t num_receiver_cores = 0;
    uint32_t max_receivers = 0;
    std::vector<CoreRange> sender_ranges;
    sender_ranges.reserve(num_sender_cores);

    for (const auto& [sender_core, receiver_set] : sender_receiver_mapping) {
        const uint32_t n = receiver_set.num_cores();
        TT_FATAL(n > 0, "DRAM sender core {} must have a non-empty receiver set", sender_core.str());
        num_receiver_cores += n;
        max_receivers = std::max(max_receivers, n);
        sender_ranges.emplace_back(sender_core);
        receiver_cores_out = receiver_cores_out.merge(receiver_set);
    }

    sender_cores_out = CoreRangeSet(sender_ranges);
    TT_FATAL(
        num_sender_cores == sender_cores_out.num_cores(),
        "Duplicate DRAM sender cores in sender_receiver_mapping ({} entries collapsed to {} cores)",
        num_sender_cores,
        sender_cores_out.num_cores());
    TT_FATAL(
        num_receiver_cores == receiver_cores_out.num_cores(),
        "Duplicate receiver cores detected across DRAM sender groups (receiver sets must be disjoint): {} receivers "
        "across senders collapsed to {} distinct cores",
        num_receiver_cores,
        receiver_cores_out.num_cores());

    // Receivers only: DRAM senders hold no slice of the ring.
    all_cores_out = receiver_cores_out;
    max_num_receivers_per_sender_out = max_receivers;
}

PersistentConfigPageLayout compute_persistent_config_page_layout(
    uint32_t max_num_receivers_per_sender, uint32_t l1_alignment) {
    const uint32_t noc_xy_offset = persistent_dfb_noc_xy_byte_offset();
    const uint32_t counters_offset = tt::align(
        noc_xy_offset + 2 * max_num_receivers_per_sender * static_cast<uint32_t>(sizeof(uint32_t)), l1_alignment);
    const uint32_t page_size = counters_offset + 2 * max_num_receivers_per_sender * l1_alignment;
    return PersistentConfigPageLayout{
        .noc_xy_offset = noc_xy_offset, .counters_offset = counters_offset, .page_size = page_size};
}

}  // namespace

PersistentDFB::PersistentDFB(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type) :
    device_(device),
    sender_receiver_mapping_(sender_receiver_mapping),
    entry_size_(entry_size),
    num_entries_(num_entries) {
    initialize_persistent_dfb(
        device, sender_receiver_mapping, sender_cores_, receiver_cores_, all_cores_, max_num_receivers_per_sender_);
    setup_buffers(buffer_type);
}

PersistentDFB::PersistentDFB(
    distributed::MeshDevice* mesh_device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type,
    DramSenderTag) :
    device_(mesh_device),
    sender_receiver_mapping_(sender_receiver_mapping),
    entry_size_(entry_size),
    num_entries_(num_entries),
    sender_core_type_value_(static_cast<uint8_t>(SenderCoreType::Dram)) {
    TT_FATAL(mesh_device != nullptr, "DRAM-sender PersistentDFB requires a non-null MeshDevice");
    const auto context_id = mesh_device->impl().get_context_id();
    const auto& hal = MetalContext::instance(context_id).hal();
    TT_FATAL(
        hal.has_programmable_core_type(HalProgrammableCoreType::DRAM),
        "DRAM-sender PersistentDFB requires programmable DRAM cores, which auto-enable on Blackhole with firmware "
        ">= 19.12.0.0 and either no harvested DRAM channels or a single device");

    initialize_dram_sender_persistent_dfb(
        sender_receiver_mapping, sender_cores_, receiver_cores_, all_cores_, max_num_receivers_per_sender_);

    // Physical worker NOC XY of each sender's receivers. Row-wise, matching the dual-sender
    // ceil/floor split (select_from_corerangeset with row_wise=true) and the consumer's receiver
    // flatten: the slab index recv_index_base + r must name the r-th receiver in this same order.
    receiver_coords_per_sender_.reserve(sender_receiver_mapping.size());
    for (const auto& [_sender_core, receivers] : sender_receiver_mapping) {
        const auto receivers_vec = corerange_to_cores(receivers, /*max_cores=*/std::nullopt, /*row_wise=*/true);
        std::vector<CoreCoord> phys;
        phys.reserve(receivers_vec.size());
        for (const auto& r : receivers_vec) {
            phys.emplace_back(mesh_device->worker_core_from_logical_core(r));
        }
        receiver_coords_per_sender_.push_back(std::move(phys));
    }

    setup_dram_sender_buffers(buffer_type);
}

void PersistentDFB::setup_dram_sender_buffers(BufferType buffer_type) {
    validate_entry_geometry(device_, entry_size_, num_entries_, buffer_type);

    auto* mesh_device = dynamic_cast<distributed::MeshDevice*>(device_);
    TT_FATAL(mesh_device != nullptr, "DRAM-sender PersistentDFB lost its MeshDevice");
    const auto context_id = mesh_device->impl().get_context_id();
    const uint32_t l1_alignment = MetalContext::instance(context_id).hal().get_alignment(HalMemType::L1);
    TT_FATAL(
        kPersistentDfbSenderPrefixBytes % l1_alignment == 0,
        "The DRISC sender-state prefix ({} B) must be a multiple of the L1 alignment ({} B) so the config page that "
        "follows it stays L1-aligned for NOC atomics",
        kPersistentDfbSenderPrefixBytes,
        l1_alignment);

    // Ring: one entry_size * num_entries slice per receiver, all at the same L1 address. Senders
    // are not included -- a DRAM sender owns no ring slice.
    const uint32_t num_receiver_cores = all_cores_.num_cores();
    const uint32_t ring_size = this->ring_size();
    auto shard_params_data =
        ShardSpecBuffer(all_cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_receiver_cores, 1});
    ShardedBufferConfig data_shard_cfg = {
        .device = device_,
        .size = ring_size * num_receiver_cores,
        .page_size = ring_size,
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(shard_params_data),
    };
    data_buffer_ = distributed::AnyBuffer::create(data_shard_cfg);
    data_address_ = static_cast<uint32_t>(data_buffer_.get_buffer()->address());

    // Receiver config pages (one per receiver, all at the same L1 address).
    allocate_config_buffer(buffer_type);

    // One DRISC-L1 block per sender, at a uniform offset across banks: the prefix followed by this
    // sender's config page. Held by RAII so the range returns to the arena with this object.
    const uint32_t config_page_bytes =
        compute_persistent_config_page_layout(max_num_receivers_per_sender_, l1_alignment).page_size;
    drisc_sender_state_alloc_ = mesh_device->impl().drisc_l1_arena().allocate(
        kPersistentDfbSenderPrefixBytes + config_page_bytes, l1_alignment);
    sender_state_drisc_l1_base_ = drisc_sender_state_alloc_->addr();

    // Receiver pages point their ack target into DRISC L1, so they need the sender base above.
    build_dram_sender_receiver_config_pages();
    write_config_to_device();
    initialize_dram_sender_state(mesh_device);
}

void PersistentDFB::build_dram_sender_receiver_config_pages() {
    TT_FATAL(config_buffer_.get_buffer() != nullptr, "PersistentDFB config buffer must exist before building pages");
    TT_FATAL(data_address_ != 0, "PersistentDFB data address must be set before building config pages");

    const auto context_id = extract_context_id(device_);
    const uint32_t l1_alignment = MetalContext::instance(context_id).hal().get_alignment(HalMemType::L1);
    const uint32_t ring_size = this->ring_size();

    const auto layout = compute_persistent_config_page_layout(max_num_receivers_per_sender_, l1_alignment);
    config_page_size_ = layout.page_size;
    credit_reset_offset_ = layout.counters_offset;
    credit_reset_size_ = config_page_size_ - credit_reset_offset_;

    const uint32_t words_per_page = config_page_size_ / sizeof(uint32_t);
    const uint32_t receiver_config_addr = this->config_address();
    // Base of the sender's counter pairs in DRISC L1 (inside its config page).
    const uint32_t drisc_counters_base = static_cast<uint32_t>(sender_state_drisc_l1_base_) +
                                         persistent_dfb_config_page_offset() + layout.counters_offset;

    config_pages_.clear();

    for (size_t s = 0; s < sender_receiver_mapping_.size(); ++s) {
        const auto& [sender_core, receiver_set] = sender_receiver_mapping_[s];
        const auto receiver_vec = corerange_to_cores(receiver_set, /*max_cores=*/std::nullopt, /*row_wise=*/true);
        const uint32_t num_recv = static_cast<uint32_t>(receiver_vec.size());

        // The receiver's ack NOC-inc lands on the DRAM core, so it needs the DRAM virtual coord
        // rather than a worker physical coord.
        const auto sender_virtual = device_->virtual_core_from_logical_core(sender_core, CoreType::DRAM);

        for (uint32_t ri = 0; ri < num_recv; ++ri) {
            std::vector<uint32_t> receiver_page(words_per_page, 0);
            uint32_t i = 0;
            receiver_page[i++] = 0;  // is_sender
            receiver_page[i++] = num_recv;
            receiver_page[i++] = data_address_;
            receiver_page[i++] = ring_size;
            receiver_page[i++] = data_address_;  // word[4]: initial fifo_ptr checkpoint
            receiver_page[i++] = entry_size_;    // word[5]: applied_entry_size
            receiver_page[i++] = layout.noc_xy_offset;
            // word[7]: this receiver's own counter pair (sent at +0, acked at +L1_ALIGNMENT).
            // Offsets are page-relative and every receiver page sits at the same L1 address, which
            // is what lets the sender reach receiver ri's slot as
            // remote_pages_sent_base + 2*ri*L1_ALIGNMENT.
            receiver_page[i++] = layout.counters_offset + 2 * ri * l1_alignment;
            // word[8]: where this receiver's pop_front NOC-incs entries_acked -- the matching slot
            // in the sender's DRISC-L1 page. setup_persistent_dfb_interface adds this to the
            // receiver's own page address, so store the (wrapping) difference between the two
            // address spaces.
            const uint32_t drisc_acked_slot = drisc_counters_base + 2 * ri * l1_alignment + l1_alignment;
            receiver_page[i++] = drisc_acked_slot - receiver_config_addr;
            receiver_page[i++] = static_cast<uint32_t>(sender_virtual.x);
            receiver_page[i++] = static_cast<uint32_t>(sender_virtual.y);
            config_pages_[receiver_vec[ri]] = std::move(receiver_page);
        }
    }
}

void PersistentDFB::initialize_dram_sender_state(distributed::MeshDevice* mesh_device) {
    const auto context_id = mesh_device->impl().get_context_id();
    auto& metal_ctx = MetalContext::instance(context_id);
    const uint32_t l1_alignment = metal_ctx.hal().get_alignment(HalMemType::L1);
    const auto layout = compute_persistent_config_page_layout(max_num_receivers_per_sender_, l1_alignment);
    const uint32_t ring_size = this->ring_size();

    const uint32_t block_bytes_size = kPersistentDfbSenderPrefixBytes + layout.page_size;
    const uint32_t config_page_addr =
        static_cast<uint32_t>(sender_state_drisc_l1_base_) + persistent_dfb_config_page_offset();

    // word[8] on a sender page is the base of the *receivers'* counter pairs. Receiver pages all
    // share one L1 address, so a single base plus 2*ri*L1_ALIGNMENT reaches every receiver's own
    // slot. setup_persistent_dfb_interface adds the stored offset to the sender's own page address
    // and then packs the result into 24 bits.
    const uint32_t receiver_counters_base = this->config_address() + layout.counters_offset;
    TT_FATAL(
        (receiver_counters_base & ~dev_msgs::REMOTE_CB_PACKED_ADDR_MASK) == 0,
        "Receiver counter base 0x{:x} does not fit the packed remote-pointer field (mask 0x{:x}) used for sender "
        "credits",
        receiver_counters_base,
        dev_msgs::REMOTE_CB_PACKED_ADDR_MASK);

    const std::vector<uint32_t> recv_index_bases = recv_index_bases_per_sender(sender_receiver_mapping_);

    const uint64_t dram_l1_noc_offset = metal_ctx.hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    const uint64_t write_addr = dram_l1_noc_offset + static_cast<uint64_t>(sender_state_drisc_l1_base_);
    auto& cluster = metal_ctx.get_cluster();
    const auto& devices = mesh_device->get_devices();

    for (size_t s = 0; s < sender_receiver_mapping_.size(); ++s) {
        const auto& [sender_logical, _receivers] = sender_receiver_mapping_[s];
        const auto& recv_phys = receiver_coords_per_sender_[s];
        const uint32_t this_num_receivers = static_cast<uint32_t>(recv_phys.size());

        std::vector<uint8_t> block_bytes(block_bytes_size, 0);
        auto* prefix = reinterpret_cast<PersistentDfbDramSenderState*>(block_bytes.data());
        prefix->recv_index_base = recv_index_bases[s];
        prefix->max_num_receivers = max_num_receivers_per_sender_;

        auto* page = reinterpret_cast<uint32_t*>(block_bytes.data() + persistent_dfb_config_page_offset());
        uint32_t i = 0;
        page[i++] = 1;  // is_sender
        page[i++] = this_num_receivers;
        page[i++] = data_address_;
        page[i++] = ring_size;
        page[i++] = data_address_;  // word[4]: initial fifo_ptr checkpoint (unused by this sender)
        page[i++] = entry_size_;    // word[5]: applied_entry_size
        page[i++] = layout.noc_xy_offset;
        page[i++] = layout.counters_offset;                     // word[7]: local counter pairs, in DRISC L1
        page[i++] = receiver_counters_base - config_page_addr;  // word[8]: remote (receiver) base

        auto* noc_xy_words = reinterpret_cast<uint32_t*>(
            block_bytes.data() + persistent_dfb_config_page_offset() + layout.noc_xy_offset);
        for (uint32_t r = 0; r < this_num_receivers; ++r) {
            noc_xy_words[2 * r + 0] = static_cast<uint32_t>(recv_phys[r].x);
            noc_xy_words[2 * r + 1] = static_cast<uint32_t>(recv_phys[r].y);
        }
        // Counters are already zero from the block's zero-fill: a fresh PersistentDFB starts with
        // no credits outstanding, and the derived write cursor of every receiver is 0.

        for (IDevice* dev : devices) {
            const CoreCoord virtual_core = dev->virtual_core_from_logical_core(sender_logical, CoreType::DRAM);
            cluster.write_core(
                dev->id(),
                tt_cxy_pair(dev->id(), virtual_core),
                std::span<const uint8_t>(block_bytes.data(), block_bytes.size()),
                write_addr);
        }
    }
}

void PersistentDFB::allocate_config_buffer(BufferType config_buffer_type) {
    const auto context_id = extract_context_id(device_);
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    const uint32_t num_all_cores = all_cores_.num_cores();
    config_page_size_ = compute_persistent_config_page_layout(max_num_receivers_per_sender_, l1_alignment).page_size;

    auto shard_params = ShardSpecBuffer(all_cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_all_cores, 1});
    ShardedBufferConfig config = {
        .device = device_,
        .size = config_page_size_ * num_all_cores,
        .page_size = config_page_size_,
        .buffer_type = config_buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(shard_params),
    };
    config_buffer_ = distributed::AnyBuffer::create(config);
}

void PersistentDFB::build_config_pages() {
    TT_FATAL(config_buffer_.get_buffer() != nullptr, "PersistentDFB config buffer must exist before building pages");
    TT_FATAL(data_address_ != 0, "PersistentDFB data address must be set before building config pages");

    const auto context_id = extract_context_id(device_);
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    const uint32_t ring_size = entry_size_ * num_entries_;

    const auto layout = compute_persistent_config_page_layout(max_num_receivers_per_sender_, l1_alignment);
    config_page_size_ = layout.page_size;
    credit_reset_offset_ = layout.counters_offset;
    credit_reset_size_ = config_page_size_ - credit_reset_offset_;

    const uint32_t data_base_addr = data_address_;
    const uint32_t words_per_page = config_page_size_ / sizeof(uint32_t);
    config_pages_.clear();

    for (const auto& [sender_core, receiver_set] : sender_receiver_mapping_) {
        const auto receiver_vec = corerange_to_cores(receiver_set);
        const uint32_t num_recv = static_cast<uint32_t>(receiver_vec.size());

        std::vector<uint32_t> sender_page(words_per_page, 0);
        uint32_t si = 0;
        sender_page[si++] = 1;
        sender_page[si++] = num_recv;
        sender_page[si++] = data_base_addr;
        sender_page[si++] = ring_size;
        sender_page[si++] = data_base_addr;  // word[4]: initial fifo_ptr checkpoint
        sender_page[si++] = entry_size_;     // word[5]: applied_entry_size
        sender_page[si++] = layout.noc_xy_offset;
        sender_page[si++] = layout.counters_offset;
        sender_page[si++] = layout.counters_offset;
        for (uint32_t ri = 0; ri < num_recv; ++ri) {
            auto phys = device_->worker_core_from_logical_core(receiver_vec[ri]);
            sender_page[si++] = static_cast<uint32_t>(phys.x);
            sender_page[si++] = static_cast<uint32_t>(phys.y);
        }
        config_pages_[sender_core] = std::move(sender_page);

        const auto sender_phys = device_->worker_core_from_logical_core(sender_core);
        for (uint32_t ri = 0; ri < num_recv; ++ri) {
            std::vector<uint32_t> receiver_page(words_per_page, 0);
            uint32_t rci = 0;
            receiver_page[rci++] = 0;
            receiver_page[rci++] = num_recv;
            receiver_page[rci++] = data_base_addr;
            receiver_page[rci++] = ring_size;
            receiver_page[rci++] = data_base_addr;
            receiver_page[rci++] = entry_size_;
            receiver_page[rci++] = layout.noc_xy_offset;
            receiver_page[rci++] = layout.counters_offset + 2 * ri * l1_alignment;
            receiver_page[rci++] = layout.counters_offset + 2 * ri * l1_alignment + l1_alignment;
            receiver_page[rci++] = static_cast<uint32_t>(sender_phys.x);
            receiver_page[rci++] = static_cast<uint32_t>(sender_phys.y);
            config_pages_[receiver_vec[ri]] = std::move(receiver_page);
        }
    }
}

void PersistentDFB::write_config_to_device() {
    TT_FATAL(config_buffer_.get_buffer() != nullptr, "PersistentDFB config buffer must exist before device write");
    Buffer* buffer = config_buffer_.get_buffer();
    const uint32_t config_size = static_cast<uint32_t>(buffer->size());
    std::vector<uint32_t> host_flat(config_size / sizeof(uint32_t), 0);

    const auto& page_mapping = buffer->get_buffer_page_mapping();
    TT_FATAL(page_mapping != nullptr, "PersistentDFB config buffer must have page mapping");

    for (const auto& [core, page] : config_pages_) {
        const auto core_it = page_mapping->core_to_core_id.find(core);
        TT_FATAL(
            core_it != page_mapping->core_to_core_id.end(),
            "PersistentDFB missing core {} in config mapping",
            core.str());
        const uint32_t dst_idx = core_it->second * (config_page_size_ / sizeof(uint32_t));
        TT_FATAL(
            dst_idx + page.size() <= host_flat.size(),
            "PersistentDFB config page for core {} overflows host staging buffer",
            core.str());
        std::copy(page.begin(), page.end(), host_flat.begin() + dst_idx);
    }

    auto mesh_buffer = config_buffer_.get_mesh_buffer();
    distributed::EnqueueWriteMeshBuffer(
        mesh_buffer->device()->mesh_command_queue(), mesh_buffer, host_flat, /*blocking=*/false);
}

void PersistentDFB::setup_buffers(BufferType buffer_type) {
    validate_entry_geometry(device_, entry_size_, num_entries_, buffer_type);

    const uint32_t num_all_cores = all_cores_.num_cores();
    const uint32_t ring_size = entry_size_ * num_entries_;
    auto shard_params_data =
        ShardSpecBuffer(all_cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_all_cores, 1});
    ShardedBufferConfig data_shard_cfg = {
        .device = device_,
        .size = ring_size * num_all_cores,
        .page_size = ring_size,
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(shard_params_data),
    };
    data_buffer_ = distributed::AnyBuffer::create(data_shard_cfg);
    data_address_ = static_cast<uint32_t>(data_buffer_.get_buffer()->address());
    allocate_config_buffer(buffer_type);
    build_config_pages();
    write_config_to_device();
}

const Buffer& PersistentDFB::config_buffer() const { return *config_buffer_.get_buffer(); }

uint32_t PersistentDFB::buffer_address() const { return data_address_; }

uint32_t PersistentDFB::config_address() const { return static_cast<uint32_t>(config_buffer().address()); }

const std::vector<uint32_t>& PersistentDFB::config_page(const CoreCoord& core) const {
    auto it = config_pages_.find(core);
    TT_FATAL(it != config_pages_.end(), "PersistentDFB has no host config page for core {}", core.str());
    return it->second;
}

uint32_t PersistentDFB::entry_size() const { return entry_size_; }

uint32_t PersistentDFB::num_entries() const { return num_entries_; }

const CoreRangeSet& PersistentDFB::sender_cores() const { return sender_cores_; }

const CoreRangeSet& PersistentDFB::receiver_cores() const { return receiver_cores_; }

const CoreRangeSet& PersistentDFB::all_cores() const { return all_cores_; }

const std::vector<std::pair<CoreCoord, CoreRangeSet>>& PersistentDFB::sender_receiver_core_mapping() const {
    return sender_receiver_mapping_;
}

PersistentDFB CreatePersistentDFB(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type) {
    return PersistentDFB(device, sender_receiver_mapping, entry_size, num_entries, buffer_type);
}

uint8_t AttachPersistentDFB(
    Program& program,
    PersistentDFB& persistent_dfb,
    const CoreRangeSet& cores,
    std::optional<uint32_t> entry_size_override) {
    return program.impl().add_persistent_dfb_attachment(persistent_dfb, cores, entry_size_override);
}

uint32_t CreatePersistentRelayDataflowBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& receiver_core_spec,
    const dfb::DataflowBufferConfig& config,
    uint8_t persistent_dfb_id) {
    const PersistentDFB& pdfb = program.impl().get_persistent_dfb_attachment(persistent_dfb_id);

    CoreRangeSet receiver_cores;
    if (std::holds_alternative<CoreCoord>(receiver_core_spec)) {
        receiver_cores = CoreRangeSet({CoreRange(std::get<CoreCoord>(receiver_core_spec))});
    } else if (std::holds_alternative<CoreRange>(receiver_core_spec)) {
        receiver_cores = CoreRangeSet({std::get<CoreRange>(receiver_core_spec)});
    } else {
        receiver_cores = std::get<CoreRangeSet>(receiver_core_spec);
    }

    TT_FATAL(
        pdfb.receiver_cores().contains(receiver_cores),
        "CreatePersistentRelayDataflowBuffer: relay cores {} must be a subset of receiver cores {}",
        receiver_cores.str(),
        pdfb.receiver_cores().str());
    TT_FATAL(
        pdfb.ring_size() % config.entry_size == 0,
        "CreatePersistentRelayDataflowBuffer: entry size {} must divide PersistentDFB ring size {}",
        config.entry_size,
        pdfb.ring_size());
    TT_FATAL(
        config.num_entries == pdfb.ring_size() / config.entry_size,
        "CreatePersistentRelayDataflowBuffer: depth {} must equal ring_size/entry_size ({})",
        config.num_entries,
        pdfb.ring_size() / config.entry_size);

    auto relay_config = config;
    relay_config.borrows_memory = true;
    relay_config.is_relay = true;
    const uint32_t relay_dfb_id = dfb::CreateDataflowBuffer(program, receiver_cores, relay_config);
    program.impl().register_persistent_relay_dfb(receiver_cores, persistent_dfb_id, relay_dfb_id);
    return relay_dfb_id;
}

// ---- DRAM-sender extension -------------------------------------------------------------------
// The free-function entrypoints declared in tt-metalium/experimental/persistent_dfb.hpp and
// impl/buffers/persistent_dfb_dram_sender_internal.hpp delegate to this struct, the only thing
// that names PersistentDFB's private DRAM-sender state. Defined here so neither header has to
// spell out the friend declaration.

namespace persistent_dfb_dram_sender {

struct PersistentDfbDramSenderInternals {
    static std::shared_ptr<PersistentDFB> make_dram_sender(
        distributed::MeshDevice* mesh_device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
        uint32_t entry_size,
        uint32_t num_entries,
        BufferType buffer_type) {
        // shared_ptr rather than a value: PersistentDFB is neither copyable nor movable, and
        // callers (op attributes, Python bindings) need a holder.
        return std::shared_ptr<PersistentDFB>(new PersistentDFB(
            mesh_device,
            sender_receiver_mapping,
            entry_size,
            num_entries,
            buffer_type,
            PersistentDFB::DramSenderTag{}));
    }

    static SenderCoreType sender_core_type(const PersistentDFB& pdfb) {
        return static_cast<SenderCoreType>(pdfb.sender_core_type_value_);
    }

    static DeviceAddr sender_state_drisc_l1_base(const PersistentDFB& pdfb) { return pdfb.sender_state_drisc_l1_base_; }

    static const std::vector<std::vector<CoreCoord>>& receiver_coords_per_sender(const PersistentDFB& pdfb) {
        return pdfb.receiver_coords_per_sender_;
    }
};

}  // namespace persistent_dfb_dram_sender

std::shared_ptr<PersistentDFB> CreatePersistentDFBForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type,
    bool support_multi_receiver_shards) {
    // Multi-receiver shards (legacy interleaved layout) force one sender per bank; the
    // receiver-contiguous layout that disallows them is what lets a bank use two senders. Shared
    // with the GlobalCircularBuffer factory so both transports place senders identically.
    auto mapping = build_dram_sender_mapping(
        &mesh_device, bank_to_receivers, /*dual_senders_per_bank=*/!support_multi_receiver_shards);
    return persistent_dfb_dram_sender::PersistentDfbDramSenderInternals::make_dram_sender(
        &mesh_device, mapping, entry_size, num_entries, buffer_type);
}

SenderCoreType persistent_dfb_sender_core_type(const PersistentDFB& persistent_dfb) {
    return persistent_dfb_dram_sender::PersistentDfbDramSenderInternals::sender_core_type(persistent_dfb);
}

uint32_t persistent_dfb_entry_size(const PersistentDFB& persistent_dfb) { return persistent_dfb.entry_size(); }

uint32_t persistent_dfb_num_entries(const PersistentDFB& persistent_dfb) { return persistent_dfb.num_entries(); }

uint32_t persistent_dfb_ring_size(const PersistentDFB& persistent_dfb) { return persistent_dfb.ring_size(); }

uint32_t persistent_dfb_buffer_address(const PersistentDFB& persistent_dfb) { return persistent_dfb.buffer_address(); }

uint32_t persistent_dfb_config_address(const PersistentDFB& persistent_dfb) { return persistent_dfb.config_address(); }

const CoreRangeSet& persistent_dfb_receiver_cores(const PersistentDFB& persistent_dfb) {
    return persistent_dfb.receiver_cores();
}

const CoreRangeSet& persistent_dfb_sender_cores(const PersistentDFB& persistent_dfb) {
    return persistent_dfb.sender_cores();
}

const std::vector<std::pair<CoreCoord, CoreRangeSet>>& persistent_dfb_sender_receiver_core_mapping(
    const PersistentDFB& persistent_dfb) {
    return persistent_dfb.sender_receiver_core_mapping();
}

DeviceAddr persistent_dfb_sender_state_drisc_l1_base(const PersistentDFB& persistent_dfb) {
    return persistent_dfb_dram_sender::PersistentDfbDramSenderInternals::sender_state_drisc_l1_base(persistent_dfb);
}

const std::vector<std::vector<CoreCoord>>& persistent_dfb_receiver_coords_per_sender(
    const PersistentDFB& persistent_dfb) {
    return persistent_dfb_dram_sender::PersistentDfbDramSenderInternals::receiver_coords_per_sender(persistent_dfb);
}

std::vector<std::vector<uint32_t>> persistent_dfb_receiver_slab_indices(const PersistentDFB& persistent_dfb) {
    const auto& mapping = persistent_dfb.sender_receiver_core_mapping();
    const std::vector<uint32_t> bases = recv_index_bases_per_sender(mapping);
    std::vector<std::vector<uint32_t>> slab(mapping.size());
    for (size_t s = 0; s < mapping.size(); ++s) {
        const uint32_t n = mapping[s].second.num_cores();
        slab[s].resize(n);
        for (uint32_t r = 0; r < n; ++r) {
            slab[s][r] = bases[s] + r;
        }
    }
    return slab;
}

}  // namespace tt::tt_metal::experimental
