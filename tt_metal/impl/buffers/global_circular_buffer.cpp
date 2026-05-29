// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <buffer.hpp>
#include <buffer_types.hpp>
#include <circular_buffer_constants.h>
#include <tt-metalium/remote_circular_buffer_packing.h>
#include <core_coord.hpp>
#include <device.hpp>
#include <global_circular_buffer.hpp>
#include <host_api.hpp>
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/dram_sender_state_block.hpp"
#include "impl/context/context_types.hpp"
#include "distributed/mesh_device_impl.hpp"
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt_align.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "distributed.hpp"
#include "hal_types.hpp"
#include "mesh_buffer.hpp"
#include "mesh_device.hpp"
#include <tt_stl/reflection.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal::experimental {

namespace {

// Body shared by the public Worker ctor and the private DRAM-sender ctor (with tag).
// Populates the core sets and reports the per-sender max receiver count.
void initialize_global_circular_buffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    bool is_dram_sender,
    CoreRangeSet& sender_cores_out,
    CoreRangeSet& receiver_cores_out,
    CoreRangeSet& all_cores_out,
    uint32_t& max_num_receivers_per_sender_out) {
    TT_FATAL(device != nullptr, "Device cannot be null");

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
        // DRAM senders and worker receivers live in disjoint programmable-core types, so their
        // physical NoC coords can never collide — no extra cross-type check needed.
        all_cores_out = receiver_cores_out;
    }
    max_num_receivers_per_sender_out = max_num_receivers_per_sender;
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
    sender_core_type_value_(static_cast<uint8_t>(experimental::SenderCoreType::Worker)) {
    uint32_t max_num_receivers_per_sender = 0;
    initialize_global_circular_buffer(
        device,
        sender_receiver_core_mapping,
        /*is_dram_sender=*/false,
        sender_cores_,
        receiver_cores_,
        all_cores_,
        max_num_receivers_per_sender);
    this->setup_cb_buffers(buffer_type, max_num_receivers_per_sender);
}

GlobalCircularBuffer::GlobalCircularBuffer(
    distributed::MeshDevice* mesh_device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type,
    DramSenderTag) :
    device_(mesh_device),
    sender_receiver_core_mapping_(sender_receiver_core_mapping),
    size_(size),
    sender_core_type_value_(static_cast<uint8_t>(experimental::SenderCoreType::Dram)) {
    TT_FATAL(mesh_device != nullptr, "DRAM-sender GlobalCircularBuffer requires a non-null MeshDevice");
    const auto context_id = mesh_device->impl().get_context_id();
    const auto& hal = MetalContext::instance(context_id).hal();
    TT_FATAL(
        hal.has_programmable_core_type(HalProgrammableCoreType::DRAM),
        "DRAM-sender GlobalCircularBuffer requires programmable DRAM cores; set "
        "TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1");
    uint32_t max_num_receivers_per_sender = 0;
    initialize_global_circular_buffer(
        mesh_device,
        sender_receiver_core_mapping,
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

    // Reserve a single per-GCB block in the per-mesh DRISC L1 arena holding both this
    // GCB's per-receiver pages_sent/pages_acked counters and its sender state block
    // (RemoteSenderCBInterface bytes + sender config + receiver NOC XY table). One
    // allocation, one RAII handle (drisc_sender_state_alloc_), released when the last
    // GCB copy goes out of scope. The arena returns a single uniform offset used by
    // every sender bank.
    //
    // Layout: [pages_sent region | sender state block]. pages_sent goes first so its
    // base equals the allocation base. On the DRISC side the pages_sent slots are
    // packed at uint32 stride (2 * 4 B per receiver) — NoC atomic inc only needs
    // 4-byte alignment, and the kernel walks them via REMOTE_CB_LOCAL_PAGES_STRIDE.
    // The receiver-side layout in worker L1 stays at the standard 2 * L1_ALIGNMENT.
    const uint32_t l1_alignment = MetalContext::instance(context_id).hal().get_alignment(HalMemType::L1);
    const uint32_t pages_sent_bytes = 2 * sizeof(uint32_t) * max_num_receivers_per_sender;
    const uint32_t pages_sent_region = tt::align(pages_sent_bytes, l1_alignment);
    const uint32_t state_block_size = dram_sender_state_block_size(max_num_receivers_per_sender);
    drisc_sender_state_alloc_ =
        mesh_device->impl().drisc_l1_arena().allocate(pages_sent_region + state_block_size, l1_alignment);
    pages_sent_drisc_l1_base_ = drisc_sender_state_alloc_->addr();
    sender_state_drisc_l1_base_ = pages_sent_drisc_l1_base_ + pages_sent_region;

    this->setup_cb_buffers(buffer_type, max_num_receivers_per_sender);
    this->initialize_dram_sender_state_block(mesh_device, max_num_receivers_per_sender);
}

void GlobalCircularBuffer::initialize_dram_sender_state_block(
    distributed::MeshDevice* mesh_device, uint32_t max_num_receivers_per_sender) {
    const auto context_id = mesh_device->impl().get_context_id();
    // The block was already allocated by the DRAM-sender ctor (combined with the
    // pages_sent region); here we just compose its bytes and write them to L1.
    const uint32_t state_block_size = dram_sender_state_block_size(max_num_receivers_per_sender);

    // The receiver NOC XY table follows the fixed struct; config_ptr points at the
    // sender config block embedded inside the struct.
    const auto noc_xy_table_addr = static_cast<uint32_t>(sender_state_drisc_l1_base_) + sizeof(DramSenderStateBlock);
    const auto config_block_addr =
        static_cast<uint32_t>(sender_state_drisc_l1_base_) + offsetof(DramSenderStateBlock, is_sender);
    const auto buffer_address = static_cast<uint32_t>(cb_buffer().address());

    const uint32_t packed_num_recv_and_remote =
        remote_cb_pack(max_num_receivers_per_sender, static_cast<uint32_t>(pages_sent_worker_l1_base_));

    // Block is identical across (device, sender) pairs; only the receiver NOC XY
    // table varies per sender. Compose once, swap in the table per sender.
    std::vector<uint8_t> block_bytes(state_block_size, 0);
    auto* hdr = reinterpret_cast<DramSenderStateBlock*>(block_bytes.data());
    hdr->config_ptr = config_block_addr;
    hdr->fifo_start_addr = buffer_address;
    hdr->fifo_limit_page_aligned = 0;  // set per-tensor at request time by the kernel
    hdr->fifo_page_size = 0;           // set per-tensor at request time by the kernel
    hdr->fifo_wr_ptr = buffer_address;
    hdr->receiver_noc_xy_ptr = noc_xy_table_addr;
    hdr->aligned_pages_sent_ptr = static_cast<uint32_t>(pages_sent_drisc_l1_base_);
    hdr->num_receivers_and_remote_pages_sent_ptr = packed_num_recv_and_remote;
    hdr->is_sender = 1;
    hdr->num_receivers = max_num_receivers_per_sender;
    hdr->buffer_address = buffer_address;
    hdr->fifo_size_per_receiver = size_;

    auto* noc_xy_words = reinterpret_cast<uint32_t*>(block_bytes.data() + sizeof(DramSenderStateBlock));

    // Host writes to a DRAM core's L1 go over NOC and need the DRAM-L1 NOC offset
    // added on top of the local L1 address. (Worker L1 has local==NOC space so the
    // EnqueueWriteMeshBuffer path used for the receiver-side config buffer doesn't
    // need this; DRAM-core L1 sits at a high NOC offset on Blackhole.)
    const uint64_t dram_l1_noc_offset =
        MetalContext::instance(context_id).hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    const uint64_t write_addr = dram_l1_noc_offset + static_cast<uint64_t>(sender_state_drisc_l1_base_);

    const auto& devices = mesh_device->get_devices();
    const std::vector<uint8_t> pages_sent_zero_bytes(2 * sizeof(uint32_t) * max_num_receivers_per_sender, 0);
    const uint64_t pages_sent_write_addr = dram_l1_noc_offset + static_cast<uint64_t>(pages_sent_drisc_l1_base_);
    for (size_t s = 0; s < sender_receiver_core_mapping_.size(); ++s) {
        const auto& [sender_logical, _receivers] = sender_receiver_core_mapping_[s];
        const auto& recv_phys = receiver_coords_per_sender_[s];
        for (uint32_t i = 0; i < max_num_receivers_per_sender; ++i) {
            const auto& c = recv_phys[i];
            noc_xy_words[2 * i + 0] = static_cast<uint32_t>(c.x);
            noc_xy_words[2 * i + 1] = static_cast<uint32_t>(c.y);
        }
        for (IDevice* dev : devices) {
            const CoreCoord virtual_core = dev->virtual_core_from_logical_core(sender_logical, CoreType::DRAM);
            MetalContext::instance().get_cluster().write_core(
                dev->id(),
                tt_cxy_pair(dev->id(), virtual_core),
                std::span<const uint8_t>(pages_sent_zero_bytes.data(), pages_sent_zero_bytes.size()),
                pages_sent_write_addr);
            MetalContext::instance().get_cluster().write_core(
                dev->id(),
                tt_cxy_pair(dev->id(), virtual_core),
                std::span<const uint8_t>(block_bytes.data(), block_bytes.size()),
                write_addr);
        }
    }
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

    auto l1_alignment = MetalContext::instance(extract_context_id(device_)).hal().get_alignment(HalMemType::L1);
    // [0] is_sender, [1] num_receivers, [2] fifo_start_addr, [3] fifo_size, [4] fifo_ptr,
    // [5] noc_xy_addr, [6] aligned_pages_sent_addr (local), [7] remote_pages_addr_override
    // (the canonical remote NoC target — equal to slot 6 for sharded GCB, points across L1
    // address spaces for DRAM-sender GCB).
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

    const auto sender_core_type = static_cast<experimental::SenderCoreType>(sender_core_type_value_);
    if (sender_core_type == experimental::SenderCoreType::Dram) {
        // pages_sent_drisc_l1_base_ has already been populated by the DRAM-sender ctor from
        // the per-mesh DriscL1Arena. We just need to expose the worker-local pages_sent base
        // here so the DRISC kernel can NOC-inc pages_sent into the receivers' config pages.
        pages_sent_worker_l1_base_ = pages_sent_address;
    }

    for (const auto& [sender_core, receiver_cores] : sender_receiver_core_mapping_) {
        const auto& receiver_cores_vec = corerange_to_cores(receiver_cores);
        uint32_t num_receivers = receiver_cores.num_cores();

        // Worker senders have their own config page in the sharded buffer; DRAM senders don't
        // (the DRISC kernel hand-rolls the sender iface state from compile-time args).
        if (sender_core_type == experimental::SenderCoreType::Worker) {
            uint32_t sender_idx = core_to_core_id.at(sender_core) * cb_config_page_size / sizeof(uint32_t);
            cb_config_host_buffer[sender_idx++] = 1;
            cb_config_host_buffer[sender_idx++] = num_receivers;
            cb_config_host_buffer[sender_idx++] = buffer_address;
            cb_config_host_buffer[sender_idx++] = size_;
            cb_config_host_buffer[sender_idx++] = buffer_address;
            cb_config_host_buffer[sender_idx++] = noc_xy_address;
            cb_config_host_buffer[sender_idx++] = pages_sent_address;
            // Sharded GCB: the sender's NOC inc lands at the same L1 offset on the receiver's
            // side, so the remote address equals aligned_pages_sent_ptr.
            cb_config_host_buffer[sender_idx++] = pages_sent_address;
            for (const auto& receiver_logical : receiver_cores_vec) {
                auto receiver_physical_coord = device_->worker_core_from_logical_core(receiver_logical);
                cb_config_host_buffer[sender_idx++] = receiver_physical_coord.x;
                cb_config_host_buffer[sender_idx++] = receiver_physical_coord.y;
            }
        }

        // Sender's physical NOC coord -- where the receiver's pages_acked NOC-inc lands. For
        // worker senders this is the worker phys; for DRAM senders it's the DRAM virtual coord.
        CoreCoord sender_physical_coord = (sender_core_type == experimental::SenderCoreType::Worker)
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
            // aligned_pages_sent_ptr; pages_acked for this receiver lives at +L1_ALIGNMENT.
            cb_config_host_buffer[receiver_idx++] = pages_sent_address + 2 * i * l1_alignment;
            // remote_pages_addr_override: the address on the SENDER's L1 where this receiver's
            // NoC-inc for pages_acked lands. For a sharded GCB sender and receiver share an L1
            // layout so this equals aligned_pages_acked_ptr. For a DRAM-sender GCB it points at
            // the per-receiver pages_acked slot in DRISC L1 (packed at uint32 stride, mirroring
            // the kernel's REMOTE_CB_LOCAL_PAGES_*).
            constexpr uint32_t drisc_slot = sizeof(uint32_t);
            cb_config_host_buffer[receiver_idx++] =
                (sender_core_type == experimental::SenderCoreType::Dram)
                    ? static_cast<uint32_t>(pages_sent_drisc_l1_base_ + 2 * i * drisc_slot + drisc_slot)
                    : (pages_sent_address + 2 * i * l1_alignment + l1_alignment);
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
// The free-function entrypoints declared in tt-metalium/experimental/global_circular_buffer.hpp
// delegate to this struct, which is the only thing that names GlobalCircularBuffer's private
// DRAM-sender state. Defined here (impl-only) so the experimental public header doesn't have
// to spell out the friend struct.

namespace global_circular_buffer_dram_sender {

struct GlobalCircularBufferDramSenderInternals {
    static GlobalCircularBuffer make_dram_sender(
        distributed::MeshDevice* mesh_device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
        uint32_t size,
        BufferType buffer_type);

    static SenderCoreType sender_core_type(const GlobalCircularBuffer& gcb);
    static DeviceAddr pages_sent_drisc_l1_base(const GlobalCircularBuffer& gcb);
    static DeviceAddr pages_sent_worker_l1_base(const GlobalCircularBuffer& gcb);
    static DeviceAddr sender_state_drisc_l1_base(const GlobalCircularBuffer& gcb);
    static const std::vector<std::vector<CoreCoord>>& receiver_coords_per_sender(const GlobalCircularBuffer& gcb);
};

GlobalCircularBuffer GlobalCircularBufferDramSenderInternals::make_dram_sender(
    distributed::MeshDevice* mesh_device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) {
    return GlobalCircularBuffer(
        mesh_device, sender_receiver_core_mapping, size, buffer_type, GlobalCircularBuffer::DramSenderTag{});
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

DeviceAddr GlobalCircularBufferDramSenderInternals::sender_state_drisc_l1_base(const GlobalCircularBuffer& gcb) {
    return gcb.sender_state_drisc_l1_base_;
}

const std::vector<std::vector<CoreCoord>>& GlobalCircularBufferDramSenderInternals::receiver_coords_per_sender(
    const GlobalCircularBuffer& gcb) {
    return gcb.receiver_coords_per_sender_;
}

}  // namespace global_circular_buffer_dram_sender

namespace {

// Map (bank_id, receivers) pairs to (DRAM-logical CoreCoord, receivers) pairs by picking
// an unused logical DRAM core for each bank.
std::vector<std::pair<CoreCoord, CoreRangeSet>> build_dram_sender_mapping(
    distributed::MeshDevice* mesh_device, const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers) {
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
    mapping.reserve(bank_to_receivers.size());
    for (const auto& [bank_id, receivers] : bank_to_receivers) {
        mapping.emplace_back(mesh_device->impl().pick_unused_dram_logical_core(bank_id), receivers);
    }
    return mapping;
}

}  // namespace

GlobalCircularBuffer CreateGlobalCircularBufferWithDramSenders(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type) {
    auto mapping = build_dram_sender_mapping(&mesh_device, bank_to_receivers);
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::make_dram_sender(
        &mesh_device, mapping, size, buffer_type);
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

DeviceAddr sender_state_drisc_l1_base(const GlobalCircularBuffer& gcb) {
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::sender_state_drisc_l1_base(gcb);
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
