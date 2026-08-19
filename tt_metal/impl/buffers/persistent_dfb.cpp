// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <buffer.hpp>
#include <buffer_types.hpp>
#include <core_coord.hpp>
#include <device.hpp>
#include "impl/dataflow_buffer/persistent_dfb.hpp"
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

#include "distributed.hpp"
#include "hostdev/remote_dfb_config_layout.h"
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

}  // namespace tt::tt_metal::experimental
