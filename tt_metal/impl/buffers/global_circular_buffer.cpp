// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <buffer.hpp>
#include <buffer_types.hpp>
#include <circular_buffer_constants.h>
#include "llrt/hal/generated/dev_msgs.hpp"
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
#include <limits>
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
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal::experimental {

namespace {

// Host-side mirror of the device remote_cb_pack (circular_buffer_interface.h): packs
// num_receivers into bits [31:24] and the remote pages_sent L1 address into bits [23:0]
// of the single RemoteSenderCBInterface::num_receivers_and_remote_pages_sent_ptr field.
// The REMOTE_CB_PACKED_* constants live in dev_msgs.h (host-visible via tt::tt_metal::dev_msgs).
uint32_t remote_cb_pack(uint32_t num_receivers, uint32_t remote_pages_sent_ptr) {
    return (num_receivers << dev_msgs::REMOTE_CB_PACKED_COUNT_SHIFT) |
           (remote_pages_sent_ptr & dev_msgs::REMOTE_CB_PACKED_ADDR_MASK);
}

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
        // Per-sender receiver counts need not be uniform: the dual-sender split gives a bank's
        // two senders ceil/floor receiver counts. The sender state block sizes its NOC XY table
        // at max_num_receivers_per_sender and zero-fills the shorter senders, each sender carries
        // its own num_receivers, and serialize_request_pages sums receivers across senders rather
        // than assuming a uniform count.
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
        "DRAM-sender GlobalCircularBuffer requires programmable DRAM cores, which auto-enable on Blackhole "
        "with firmware >= 19.12.0.0");
    uint32_t max_num_receivers_per_sender = 0;
    initialize_global_circular_buffer(
        mesh_device,
        sender_receiver_core_mapping,
        /*is_dram_sender=*/true,
        sender_cores_,
        receiver_cores_,
        all_cores_,
        max_num_receivers_per_sender);

    // Preserve the reference device's physical worker NOC XY for the experimental
    // receiver_coords_per_sender accessor. Device-bound sender-state initialization
    // below resolves worker coordinates independently for every device.
    IDevice* reference_device = mesh_device->get_devices().at(0);
    receiver_coords_per_sender_.reserve(sender_receiver_core_mapping.size());
    for (const auto& [_sender_core, receivers] : sender_receiver_core_mapping) {
        // Row-wise, to match both the dual-sender ceil/floor split (select_from_corerangeset
        // with row_wise=true) and the validator's receiver flatten. The slab index
        // recv_index_base+r maps to the r-th receiver in this order, so all three must agree;
        // they only diverge when a bank's receiver set spans multiple rows and columns.
        const auto& receivers_vec = corerange_to_cores(receivers, /*max_cores=*/std::nullopt, /*row_wise=*/true);
        std::vector<CoreCoord> phys;
        phys.reserve(receivers_vec.size());
        for (const auto& r : receivers_vec) {
            phys.emplace_back(reference_device->worker_core_from_logical_core(r));
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

    this->setup_cb_buffers(buffer_type, max_num_receivers_per_sender, mesh_device);
    this->initialize_dram_sender_state_block(mesh_device, max_num_receivers_per_sender);
}

namespace {
// Per-sender bank-local recv_index_base. Senders are ordered [bank b s0, bank b s1, bank b+1 s0,
// ...] (sender_logical.x == bank_id); recv_index_base resets to 0 on a bank change and accumulates
// within a bank (dual senders share a bank). Returns one value per sender in mapping order. Single
// source for both the L1 state-block stamping and the experimental receiver_slab_indices accessor.
std::vector<uint32_t> recv_index_bases_per_sender(const std::vector<std::pair<CoreCoord, CoreRangeSet>>& mapping) {
    std::vector<uint32_t> bases(mapping.size(), 0);
    uint32_t recv_index_base = 0;
    uint32_t prev_bank = std::numeric_limits<uint32_t>::max();
    for (size_t s = 0; s < mapping.size(); ++s) {
        const uint32_t bank = static_cast<uint32_t>(mapping[s].first.x);
        recv_index_base = (bank == prev_bank) ? recv_index_base : 0u;
        prev_bank = bank;
        bases[s] = recv_index_base;
        recv_index_base += mapping[s].second.num_cores();
    }
    return bases;
}

// A DRAM-sender GCB exposes the reference device's logical sender coordinates for API
// compatibility. `canonical_sender_role` recovers a canonical coordinate's stable role
// (its index within the bank's ordered sender list). It depends only on the reference
// device, so callers resolve it once per sender rather than once per (sender, device).
size_t canonical_sender_role(distributed::MeshDevice* mesh_device, const CoreCoord& canonical_sender) {
    const uint32_t bank_id = static_cast<uint32_t>(canonical_sender.x);
    const std::vector<CoreCoord> canonical_senders = mesh_device->impl().dram_sender_logical_cores(bank_id);
    const auto canonical_it = std::find(canonical_senders.begin(), canonical_senders.end(), canonical_sender);
    TT_FATAL(
        canonical_it != canonical_senders.end(),
        "Canonical DRAM sender ({}, {}) is not a provisioned sender for bank {}",
        canonical_sender.x,
        canonical_sender.y,
        bank_id);
    return static_cast<size_t>(std::distance(canonical_senders.begin(), canonical_it));
}

// Selects the logical DRAM core that plays `sender_role` for `bank_id` on `device`'s
// harvested topology.
CoreCoord device_sender_for_role(
    distributed::MeshDevice* mesh_device, const IDevice* device, uint32_t bank_id, size_t sender_role) {
    const std::vector<CoreCoord> device_senders = mesh_device->impl().dram_sender_logical_cores(device, bank_id);
    TT_FATAL(
        sender_role < device_senders.size(),
        "Device {} has {} DRAM senders for bank {}, but canonical sender role {} is required",
        device->id(),
        device_senders.size(),
        bank_id,
        sender_role);
    return device_senders[sender_role];
}
}  // namespace

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

    // The fixed header fields are common, while the receiver NOC XY table is composed
    // separately for each (device, sender) pair.
    std::vector<uint8_t> block_bytes(state_block_size, 0);
    auto* hdr = reinterpret_cast<DramSenderStateBlock*>(block_bytes.data());
    hdr->config_ptr = config_block_addr;
    hdr->fifo_start_addr = buffer_address;
    hdr->fifo_wr_ptr = buffer_address;
    hdr->receiver_noc_xy_ptr = noc_xy_table_addr;
    hdr->aligned_pages_sent_ptr = static_cast<uint32_t>(pages_sent_drisc_l1_base_);
    hdr->num_receivers_and_remote_pages_sent_ptr = packed_num_recv_and_remote;
    hdr->is_sender = 1;
    hdr->num_receivers = max_num_receivers_per_sender;
    hdr->buffer_address = buffer_address;
    hdr->fifo_size_per_receiver = size_;
    hdr->max_num_receivers = max_num_receivers_per_sender;

    auto* noc_xy_words = reinterpret_cast<uint32_t*>(block_bytes.data() + sizeof(DramSenderStateBlock));

    // Host writes to a DRAM core's L1 go over NOC and need the DRAM-L1 NOC offset
    // added on top of the local L1 address. (Worker L1 has local==NOC space so the
    // EnqueueWriteMeshBuffer path used for the receiver-side config buffer doesn't
    // need this; DRAM-core L1 sits at a high NOC offset on Blackhole.)
    auto& metal_ctx = MetalContext::instance(context_id);
    const uint64_t dram_l1_noc_offset = metal_ctx.hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    const uint64_t write_addr = dram_l1_noc_offset + static_cast<uint64_t>(sender_state_drisc_l1_base_);

    const auto& devices = mesh_device->get_devices();
    const std::vector<uint8_t> pages_sent_zero_bytes(2 * sizeof(uint32_t) * max_num_receivers_per_sender, 0);
    const uint64_t pages_sent_write_addr = dram_l1_noc_offset + static_cast<uint64_t>(pages_sent_drisc_l1_base_);
    auto& cluster = metal_ctx.get_cluster();
    // Per-sender bank-local slab offset (see recv_index_bases_per_sender): senders are ordered
    // [bank b sender 0, bank b sender 1, bank b+1 sender 0, ...] (sender_logical.x == bank_id);
    // when two senders share a bank, the second reads slabs that start where the first's end.
    const std::vector<uint32_t> recv_index_bases = recv_index_bases_per_sender(sender_receiver_core_mapping_);
    for (size_t s = 0; s < sender_receiver_core_mapping_.size(); ++s) {
        const auto& [canonical_sender, receivers] = sender_receiver_core_mapping_[s];
        const auto receivers_vec = corerange_to_cores(receivers, /*max_cores=*/std::nullopt, /*row_wise=*/true);
        const uint32_t this_num_receivers = static_cast<uint32_t>(receivers_vec.size());

        // Per-sender header fields (the rest of block_bytes is constant across senders).
        hdr->num_receivers = this_num_receivers;
        hdr->recv_index_base = recv_index_bases[s];
        hdr->num_receivers_and_remote_pages_sent_ptr =
            remote_cb_pack(this_num_receivers, static_cast<uint32_t>(pages_sent_worker_l1_base_));

        // The canonical sender role is device-independent; resolve it once per sender.
        const uint32_t bank_id = static_cast<uint32_t>(canonical_sender.x);
        const size_t sender_role = canonical_sender_role(mesh_device, canonical_sender);

        for (IDevice* dev : devices) {
            for (uint32_t i = 0; i < max_num_receivers_per_sender; ++i) {
                if (i < this_num_receivers) {
                    const CoreCoord receiver_phys = dev->worker_core_from_logical_core(receivers_vec[i]);
                    noc_xy_words[2 * i + 0] = static_cast<uint32_t>(receiver_phys.x);
                    noc_xy_words[2 * i + 1] = static_cast<uint32_t>(receiver_phys.y);
                } else {
                    noc_xy_words[2 * i + 0] = 0;
                    noc_xy_words[2 * i + 1] = 0;
                }
            }
            const CoreCoord sender_logical = device_sender_for_role(mesh_device, dev, bank_id, sender_role);
            const CoreCoord virtual_core = dev->virtual_core_from_logical_core(sender_logical, CoreType::DRAM);
            cluster.write_core(
                dev->id(),
                tt_cxy_pair(dev->id(), virtual_core),
                std::span<const uint8_t>(pages_sent_zero_bytes.data(), pages_sent_zero_bytes.size()),
                pages_sent_write_addr);
            cluster.write_core(
                dev->id(),
                tt_cxy_pair(dev->id(), virtual_core),
                std::span<const uint8_t>(block_bytes.data(), block_bytes.size()),
                write_addr);
        }
    }
}

void GlobalCircularBuffer::setup_cb_buffers(
    BufferType buffer_type, uint32_t max_num_receivers_per_sender, distributed::MeshDevice* dram_sender_mesh_device) {
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

    // The canonical (reference-device) sender role for each mapping entry is device-independent;
    // resolve it once here so the per-device config build only performs the target-device index lookup.
    std::vector<std::pair<uint32_t, size_t>> canonical_roles;  // {bank_id, role}
    if (sender_core_type == experimental::SenderCoreType::Dram) {
        TT_FATAL(dram_sender_mesh_device != nullptr, "DRAM-sender GCB config requires its MeshDevice");
        canonical_roles.reserve(sender_receiver_core_mapping_.size());
        for (const auto& [canonical_sender, _receivers] : sender_receiver_core_mapping_) {
            canonical_roles.emplace_back(
                static_cast<uint32_t>(canonical_sender.x),
                canonical_sender_role(dram_sender_mesh_device, canonical_sender));
        }
    }

    const auto make_config_host_buffer = [&](IDevice* config_device) {
        std::vector<uint32_t> cb_config_host_buffer(cb_config_size / sizeof(uint32_t), 0);
        for (size_t s = 0; s < sender_receiver_core_mapping_.size(); ++s) {
            const auto& [canonical_sender, receiver_cores] = sender_receiver_core_mapping_[s];
            const auto& receiver_cores_vec = corerange_to_cores(receiver_cores);
            uint32_t num_receivers = receiver_cores.num_cores();

            // Worker senders have their own config page in the sharded buffer; DRAM senders don't
            // (the DRISC kernel hand-rolls the sender iface state from compile-time args).
            if (sender_core_type == experimental::SenderCoreType::Worker) {
                uint32_t sender_idx = core_to_core_id.at(canonical_sender) * cb_config_page_size / sizeof(uint32_t);
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
                    auto receiver_physical_coord = config_device->worker_core_from_logical_core(receiver_logical);
                    cb_config_host_buffer[sender_idx++] = receiver_physical_coord.x;
                    cb_config_host_buffer[sender_idx++] = receiver_physical_coord.y;
                }
            }

            // Sender's physical NOC coord -- where the receiver's pages_acked NOC-inc lands. For
            // worker senders this is the worker phys; for DRAM senders it is the target device's
            // DRAM virtual coord for the canonical sender's stable bank/role.
            const CoreCoord sender_logical =
                (sender_core_type == experimental::SenderCoreType::Dram)
                    ? device_sender_for_role(
                          dram_sender_mesh_device, config_device, canonical_roles[s].first, canonical_roles[s].second)
                    : canonical_sender;
            CoreCoord sender_physical_coord =
                (sender_core_type == experimental::SenderCoreType::Worker)
                    ? config_device->worker_core_from_logical_core(sender_logical)
                    : config_device->virtual_core_from_logical_core(sender_logical, CoreType::DRAM);

            for (uint32_t i = 0; i < receiver_cores_vec.size(); i++) {
                uint32_t receiver_idx =
                    core_to_core_id.at(receiver_cores_vec[i]) * cb_config_page_size / sizeof(uint32_t);
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
        return cb_config_host_buffer;
    };

    auto mesh_buffer = cb_config_buffer_.get_mesh_buffer();
    if (sender_core_type == experimental::SenderCoreType::Dram) {
        for (IDevice* config_device : dram_sender_mesh_device->get_devices()) {
            std::vector<uint32_t> cb_config_host_buffer = make_config_host_buffer(config_device);
            const distributed::MeshCoordinate device_coord =
                dram_sender_mesh_device->get_view().find_device(config_device->id());
            distributed::WriteShard(
                mesh_buffer->device()->mesh_command_queue(),
                mesh_buffer,
                cb_config_host_buffer,
                device_coord,
                /*blocking=*/true);
        }
    } else {
        std::vector<uint32_t> cb_config_host_buffer = make_config_host_buffer(device_);
        distributed::EnqueueWriteMeshBuffer(
            mesh_buffer->device()->mesh_command_queue(), mesh_buffer, cb_config_host_buffer, false);
    }
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
    ttsl::reflection::operator<<(os, value);
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
    static std::vector<std::vector<uint32_t>> receiver_slab_indices(const GlobalCircularBuffer& gcb);
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

std::vector<std::vector<uint32_t>> GlobalCircularBufferDramSenderInternals::receiver_slab_indices(
    const GlobalCircularBuffer& gcb) {
    // Order-agnostic: just the bank-local slab index (recv_index_base + r) each receiver reads.
    // The caller maps (bank, slab index) -> a global position using the tensor's shard
    // distribution; that ordering is not a GCB concept, so it is not applied here. This is always
    // well-defined regardless of bank density/uniformity.
    const auto& mapping = gcb.sender_receiver_core_mapping();
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

}  // namespace global_circular_buffer_dram_sender

namespace {

// Map (bank_id, receivers) pairs to (DRAM-logical CoreCoord, receivers) pairs. In dual mode each
// bank is driven by two DRISC sender cores (a free non-endpoint subchannel on NOC0 and the
// NOC1-endpoint subchannel, also running on NOC0): the bank's ordered receiver list is split
// ceil/floor across them, so each core delivers roughly half. Receiver order is preserved
// (receiver-table order == bank-local slab order, the recv-contig contract); the second sender's
// slabs start where the first sender's receivers end (tracked host-side via
// DramSenderStateBlock.recv_index_base, whose per-bank reset assumes a bank's senders are
// contiguous in this mapping — hence the no-duplicate-bank guard below).
std::vector<std::pair<CoreCoord, CoreRangeSet>> build_dram_sender_mapping(
    distributed::MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    bool dual_senders_per_bank) {
    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
    mapping.reserve((dual_senders_per_bank ? 2 : 1) * bank_to_receivers.size());
    std::unordered_set<uint32_t> seen_banks;
    for (const auto& [bank_id, receivers] : bank_to_receivers) {
        const uint32_t n = receivers.num_cores();
        TT_FATAL(n > 0, "DRAM bank {} has no receivers", bank_id);
        TT_FATAL(
            seen_banks.insert(bank_id).second,
            "DRAM bank {} appears more than once in bank_to_receivers; each bank must be listed exactly once "
            "(the per-bank recv_index_base / slab assignment assumes one contiguous group of senders per bank).",
            bank_id);

        if (!dual_senders_per_bank) {
            // Single sender per bank (the free non-endpoint subchannel).
            mapping.emplace_back(mesh_device->impl().pick_unused_dram_logical_core(bank_id), receivers);
            continue;
        }

        // A single receiver cannot be split across two senders. Since the prefetcher always
        // provisions both senders per bank and routes PREFETCH only to the senders this GCB
        // actually maps, we can map just the primary sender for such a bank and leave the
        // secondary parked — same as the single-sender path. Dual- and single-sender banks may
        // therefore coexist in one dual-mode GCB.
        const std::vector<CoreCoord> sender_cores = mesh_device->impl().dram_sender_logical_cores(bank_id);
        if (n == 1) {
            mapping.emplace_back(sender_cores.at(0), receivers);
            continue;
        }

        // Two sender cores per bank: split the bank's ordered receivers ceil/floor.
        // select_from_corerangeset indices are inclusive and traverse row-wise (matching
        // corerange_to_cores used elsewhere), so the receiver-table / bank-local slab
        // order is preserved.
        const uint32_t first_count = (n + 1) / 2;
        mapping.emplace_back(sender_cores.at(0), select_from_corerangeset(receivers, 0, first_count - 1, true));
        mapping.emplace_back(sender_cores.at(1), select_from_corerangeset(receivers, first_count, n - 1, true));
    }
    return mapping;
}

}  // namespace

GlobalCircularBuffer CreateGlobalCircularBufferForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type,
    bool support_multi_receiver_shards) {
    // Multi-receiver shards (legacy interleaved layout) force one sender per bank; the
    // receiver-contiguous layout that disallows them is what lets a bank use two senders.
    auto mapping = build_dram_sender_mapping(
        &mesh_device, bank_to_receivers, /*dual_senders_per_bank=*/!support_multi_receiver_shards);
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

std::vector<std::vector<uint32_t>> receiver_slab_indices(const GlobalCircularBuffer& gcb) {
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::receiver_slab_indices(gcb);
}

}  // namespace tt::tt_metal::experimental

namespace std {

std::size_t hash<tt::tt_metal::experimental::GlobalCircularBuffer>::operator()(
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_circular_buffer) const {
    return ttsl::hash::hash_objects_with_default_seed(global_circular_buffer.attribute_values());
}

}  // namespace std
