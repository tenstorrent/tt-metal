// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <buffer.hpp>
#include <buffer_types.hpp>
#include <core_coord.hpp>
#include <device.hpp>
#include <global_circular_buffer.hpp>
#include <host_api.hpp>
#include <tt-metalium/experimental/dram_subchannel.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt_align.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "distributed.hpp"
#include "hal_types.hpp"
#include "mesh_buffer.hpp"
#include "mesh_device.hpp"
#include <tt_stl/reflection.hpp>
#include "impl/context/metal_context.hpp"
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal::experimental {

namespace {

// Body shared by the public Worker ctor and the private DRAM-sender ctor (with tag).
// Populates the core sets, validates topology, and triggers the buffer/config L1 setup.
void initialize_global_circular_buffer(
    GlobalCircularBuffer& gcb,
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type,
    bool is_dram_sender,
    // Mutable references into gcb's private state (we're called from inside the class via
    // the friend struct, so all access is legal).
    CoreRangeSet& sender_cores_out,
    CoreRangeSet& receiver_cores_out,
    CoreRangeSet& all_cores_out,
    uint32_t& max_num_receivers_per_sender_out) {
    TT_FATAL(device != nullptr, "Device cannot be null");
    if (is_dram_sender) {
        const auto& hal = MetalContext::instance().hal();
        TT_FATAL(
            hal.has_programmable_core_type(HalProgrammableCoreType::DRAM),
            "DRAM-sender GlobalCircularBuffer requires programmable DRAM cores; set "
            "TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1");
    }

    uint32_t num_sender_cores = sender_receiver_core_mapping.size();
    TT_FATAL(num_sender_cores > 0, "At least one sender required");
    uint32_t num_receiver_cores = 0;
    uint32_t max_num_receivers_per_sender = 0;
    std::vector<CoreRange> sender_cores;
    sender_cores.reserve(num_sender_cores);
    for (const auto& [sender_core, receiver_cores] : sender_receiver_core_mapping) {
        num_receiver_cores += receiver_cores.num_cores();
        sender_cores.emplace_back(sender_core);
        receiver_cores_out = receiver_cores_out.merge(receiver_cores);
        max_num_receivers_per_sender = std::max(max_num_receivers_per_sender, receiver_cores.num_cores());
    }
    sender_cores_out = CoreRangeSet(sender_cores);
    TT_FATAL(num_sender_cores == sender_cores_out.num_cores(), "Duplicate sender cores found");
    TT_FATAL(num_receiver_cores == receiver_cores_out.num_cores(), "Duplicate receiver cores found");

    if (!is_dram_sender) {
        all_cores_out = sender_cores_out.merge(receiver_cores_out);
        TT_FATAL(all_cores_out.num_cores() == num_sender_cores + num_receiver_cores, "Duplicate cores found");
    } else {
        all_cores_out = receiver_cores_out;
        std::unordered_set<CoreCoord> sender_dram_phys;
        for (const auto& [sender_core, _receiver_cores] : sender_receiver_core_mapping) {
            CoreCoord phys = device->virtual_core_from_logical_core(sender_core, CoreType::DRAM);
            sender_dram_phys.insert(phys);
        }
        for (const auto& receiver_logical : corerange_to_cores(receiver_cores_out)) {
            CoreCoord receiver_phys = device->worker_core_from_logical_core(receiver_logical);
            TT_FATAL(
                sender_dram_phys.find(receiver_phys) == sender_dram_phys.end(),
                "Receiver worker core physical ({}, {}) collides with a DRAM sender physical coord",
                receiver_phys.x,
                receiver_phys.y);
        }
    }
    max_num_receivers_per_sender_out = max_num_receivers_per_sender;
    (void)gcb;
    (void)size;
    (void)buffer_type;
}

}  // namespace

GlobalCircularBuffer::GlobalCircularBuffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) :
    device_(device),
    sender_receiver_core_mapping_(sender_receiver_core_mapping),
    size_(size),
    sender_core_type_value_(0) {
    uint32_t max_num_receivers_per_sender = 0;
    initialize_global_circular_buffer(
        *this,
        device,
        sender_receiver_core_mapping,
        size,
        buffer_type,
        /*is_dram_sender=*/false,
        sender_cores_,
        receiver_cores_,
        all_cores_,
        max_num_receivers_per_sender);
    this->setup_cb_buffers(buffer_type, max_num_receivers_per_sender);
}

GlobalCircularBuffer::GlobalCircularBuffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type,
    DramSenderTag) :
    device_(device),
    sender_receiver_core_mapping_(sender_receiver_core_mapping),
    size_(size),
    sender_core_type_value_(1) {
    uint32_t max_num_receivers_per_sender = 0;
    initialize_global_circular_buffer(
        *this,
        device,
        sender_receiver_core_mapping,
        size,
        buffer_type,
        /*is_dram_sender=*/true,
        sender_cores_,
        receiver_cores_,
        all_cores_,
        max_num_receivers_per_sender);

    // Pre-compute physical worker NOC XY for each sender's receivers (the DRISC kernel
    // pushes to these as runtime args).
    receiver_coords_per_sender_.reserve(sender_receiver_core_mapping.size());
    for (const auto& [_sender_core, receivers] : sender_receiver_core_mapping) {
        const auto& receivers_vec = corerange_to_cores(receivers);
        std::vector<CoreCoord> phys;
        phys.reserve(receivers_vec.size());
        for (const auto& r : receivers_vec) {
            phys.emplace_back(device_->worker_core_from_logical_core(r));
        }
        receiver_coords_per_sender_.push_back(std::move(phys));
    }

    this->setup_cb_buffers(buffer_type, max_num_receivers_per_sender);
}

void GlobalCircularBuffer::setup_cb_buffers(BufferType buffer_type, uint32_t max_num_receivers_per_sender) {
    TT_FATAL(
        buffer_type == BufferType::L1 or buffer_type == BufferType::L1_SMALL,
        "Global circular buffer can only be created for L1 buffer types");
    uint32_t num_cores = all_cores_.num_cores();

    auto shard_parameters = ShardSpecBuffer(all_cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});

    uint32_t cb_buffer_size = size_ * num_cores;
    ShardedBufferConfig cb_buffer_shard_config = {
        .device = device_,
        .size = cb_buffer_size,
        .page_size = size_,
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = shard_parameters,
    };
    cb_buffer_ = distributed::AnyBuffer::create(cb_buffer_shard_config);

    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    // [0] is_sender, [1] num_receivers, [2] fifo_start_addr, [3] fifo_size, [4] fifo_ptr,
    // [5] noc_xy_addr, [6] aligned_pages_sent_addr (local), [7] remote_pages_addr_override
    // (0 means "no override; local == remote", which is true for sharded GCBs).
    constexpr uint32_t num_config_elements = 8;
    uint32_t num_noc_xy_words = 2 * max_num_receivers_per_sender;
    auto cb_config_page_size = tt::align((num_config_elements + num_noc_xy_words) * sizeof(uint32_t), l1_alignment) +
                               (2 * max_num_receivers_per_sender * l1_alignment);
    uint32_t cb_config_size = cb_config_page_size * num_cores;
    ShardedBufferConfig cb_config_buffer_shard_config = {
        .device = device_,
        .size = cb_config_size,
        .page_size = cb_config_page_size,
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(shard_parameters),
    };
    cb_config_buffer_ = distributed::AnyBuffer::create(cb_config_buffer_shard_config);

    // Write the config buffer to the device
    // Only block for the slow dispatch case
    auto config_buffer_address = cb_config_buffer_.get_buffer()->address();
    const auto& core_to_core_id = cb_config_buffer_.get_buffer()->get_buffer_page_mapping()->core_to_core_id;
    std::vector<uint32_t> cb_config_host_buffer(cb_config_size / sizeof(uint32_t), 0);
    uint32_t noc_xy_address = config_buffer_address + (num_config_elements * sizeof(uint32_t));
    uint32_t pages_sent_address = tt::align(noc_xy_address + (num_noc_xy_words * sizeof(uint32_t)), l1_alignment);
    auto buffer_address = cb_buffer().address();

    if (sender_core_type_value_ == 1) {
        // Where the DRISC sender's local pages_sent/acked counters live, agreed between host and
        // the DRISC kernel. Used as the receiver's remote_pages_acked_addr override so the
        // receiver's NOC inc lands in DRISC L1 instead of a phantom worker L1 offset.
        const auto& hal_inst = MetalContext::instance().hal();
        pages_sent_drisc_l1_base_ = hal_inst.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
        // Expose the worker-local pages_sent base so the DRISC kernel can NOC-inc pages_sent to it.
        pages_sent_worker_l1_base_ = pages_sent_address;
    }

    for (const auto& [sender_core, receiver_cores] : sender_receiver_core_mapping_) {
        const auto& receiver_cores_vec = corerange_to_cores(receiver_cores);
        uint32_t num_receivers = receiver_cores.num_cores();

        // Worker senders have their own config page in the sharded buffer; DRAM senders don't
        // (the DRISC kernel hand-rolls the sender iface state from compile-time args).
        if (sender_core_type_value_ == 0) {
            uint32_t sender_idx = core_to_core_id.at(sender_core) * cb_config_page_size / sizeof(uint32_t);
            cb_config_host_buffer[sender_idx++] = 1;
            cb_config_host_buffer[sender_idx++] = num_receivers;
            cb_config_host_buffer[sender_idx++] = buffer_address;
            cb_config_host_buffer[sender_idx++] = size_;
            cb_config_host_buffer[sender_idx++] = buffer_address;
            cb_config_host_buffer[sender_idx++] = noc_xy_address;
            cb_config_host_buffer[sender_idx++] = pages_sent_address;
            cb_config_host_buffer[sender_idx++] = 0;  // remote_pages_addr_override: 0 = local==remote
            for (const auto& receiver_logical : receiver_cores_vec) {
                auto receiver_physical_coord = device_->worker_core_from_logical_core(receiver_logical);
                cb_config_host_buffer[sender_idx++] = receiver_physical_coord.x;
                cb_config_host_buffer[sender_idx++] = receiver_physical_coord.y;
            }
        }

        // Sender's physical NOC coord -- where the receiver's pages_acked NOC-inc lands. For
        // worker senders this is the worker phys; for DRAM senders it's the DRAM virtual coord.
        CoreCoord sender_physical_coord = (sender_core_type_value_ == 0)
                                              ? device_->worker_core_from_logical_core(sender_core)
                                              : device_->virtual_core_from_logical_core(sender_core, CoreType::DRAM);

        for (uint32_t i = 0; i < receiver_cores_vec.size(); i++) {
            uint32_t receiver_idx = core_to_core_id.at(receiver_cores_vec[i]) * cb_config_page_size / sizeof(uint32_t);
            cb_config_host_buffer[receiver_idx++] = 0;  // is_sender
            cb_config_host_buffer[receiver_idx++] = num_receivers;
            cb_config_host_buffer[receiver_idx++] = buffer_address;
            cb_config_host_buffer[receiver_idx++] = size_;
            cb_config_host_buffer[receiver_idx++] = buffer_address;
            cb_config_host_buffer[receiver_idx++] = noc_xy_address;
            cb_config_host_buffer[receiver_idx++] = pages_sent_address + 2 * i * l1_alignment;
            // remote_pages_addr_override: 0 means "local==remote" (worker sharded GCB).
            // For DRAM senders this points at the per-receiver pages_acked slot in DRISC L1 so the
            // receiver's NoC-inc lands on the sender (DRAM core) side.
            cb_config_host_buffer[receiver_idx++] =
                (sender_core_type_value_ == 1)
                    ? static_cast<uint32_t>(pages_sent_drisc_l1_base_ + 2 * i * l1_alignment + l1_alignment)
                    : 0;
            cb_config_host_buffer[receiver_idx++] = sender_physical_coord.x;
            cb_config_host_buffer[receiver_idx++] = sender_physical_coord.y;
        }
    }
    auto mesh_buffer = cb_config_buffer_.get_mesh_buffer();
    distributed::EnqueueWriteMeshBuffer(
        mesh_buffer->device()->mesh_command_queue(), mesh_buffer, cb_config_host_buffer, false);
}

const Buffer& GlobalCircularBuffer::cb_buffer() const { return *cb_buffer_.get_buffer(); }

const CoreRangeSet& GlobalCircularBuffer::sender_cores() const { return sender_cores_; }

const CoreRangeSet& GlobalCircularBuffer::receiver_cores() const { return receiver_cores_; }

const CoreRangeSet& GlobalCircularBuffer::all_cores() const { return all_cores_; }

DeviceAddr GlobalCircularBuffer::buffer_address() const { return cb_buffer().address(); }

DeviceAddr GlobalCircularBuffer::config_address() const { return cb_config_buffer_.get_buffer()->address(); }

uint32_t GlobalCircularBuffer::size() const { return size_; }

const std::vector<std::pair<CoreCoord, CoreRangeSet>>& GlobalCircularBuffer::sender_receiver_core_mapping() const {
    return sender_receiver_core_mapping_;
}

std::ostream& operator<<(std::ostream& os, const GlobalCircularBuffer& value) {
    tt::stl::reflection::operator<<(os, value);
    return os;
}

// ---- Experimental DRAM-sender extension -------------------------------------------------
// Declarations live in tt-metalium/experimental/global_circular_buffer.hpp. The friend
// struct in that header is the only way to construct or query the DRAM-sender state on a
// GlobalCircularBuffer; the public GlobalCircularBuffer API is unchanged.

namespace global_circular_buffer_dram_sender {

GlobalCircularBuffer GlobalCircularBufferDramSenderInternals::make_dram_sender(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) {
    return GlobalCircularBuffer(
        device, sender_receiver_core_mapping, size, buffer_type, GlobalCircularBuffer::DramSenderTag{});
}

SenderCoreType GlobalCircularBufferDramSenderInternals::sender_core_type(const GlobalCircularBuffer& gcb) {
    return static_cast<SenderCoreType>(gcb.sender_core_type_value_);
}

DeviceAddr GlobalCircularBufferDramSenderInternals::pages_sent_drisc_l1_base(const GlobalCircularBuffer& gcb) {
    return gcb.pages_sent_drisc_l1_base_;
}

DeviceAddr GlobalCircularBufferDramSenderInternals::pages_sent_worker_l1_base(const GlobalCircularBuffer& gcb) {
    return gcb.pages_sent_worker_l1_base_;
}

const std::vector<std::vector<CoreCoord>>& GlobalCircularBufferDramSenderInternals::receiver_coords_per_sender(
    const GlobalCircularBuffer& gcb) {
    return gcb.receiver_coords_per_sender_;
}

}  // namespace global_circular_buffer_dram_sender

namespace {

// Map (bank_id, receivers) pairs to (DRAM-logical CoreCoord, receivers) pairs by picking
// an unused subchannel for each bank.
std::vector<std::pair<CoreCoord, CoreRangeSet>> build_dram_sender_mapping(
    IDevice* device, const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers) {
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
    mapping.reserve(bank_to_receivers.size());
    for (const auto& [bank_id, receivers] : bank_to_receivers) {
        uint32_t sub = ::tt::tt_metal::experimental::pick_unused_dram_subchannel(device, bank_id);
        mapping.emplace_back(CoreCoord{bank_id, sub}, receivers);
    }
    return mapping;
}

}  // namespace

GlobalCircularBuffer CreateGlobalCircularBufferWithDramSenders(
    IDevice* device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type) {
    auto mapping = build_dram_sender_mapping(device, bank_to_receivers);
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::make_dram_sender(
        device, mapping, size, buffer_type);
}

GlobalCircularBuffer CreateGlobalCircularBufferWithDramSenders(
    distributed::MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type) {
    auto mapping = build_dram_sender_mapping(mesh_device, bank_to_receivers);
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::make_dram_sender(
        mesh_device, mapping, size, buffer_type);
}

SenderCoreType sender_core_type(const GlobalCircularBuffer& gcb) {
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::sender_core_type(gcb);
}

DeviceAddr pages_sent_drisc_l1_base(const GlobalCircularBuffer& gcb) {
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::pages_sent_drisc_l1_base(gcb);
}

DeviceAddr pages_sent_worker_l1_base(const GlobalCircularBuffer& gcb) {
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::pages_sent_worker_l1_base(gcb);
}

const std::vector<std::vector<CoreCoord>>& receiver_coords_per_sender(const GlobalCircularBuffer& gcb) {
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::receiver_coords_per_sender(gcb);
}

}  // namespace tt::tt_metal::experimental

namespace std {

std::size_t hash<tt::tt_metal::experimental::GlobalCircularBuffer>::operator()(
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_circular_buffer) const {
    return tt::stl::hash::hash_objects_with_default_seed(global_circular_buffer.attribute_values());
}

}  // namespace std
