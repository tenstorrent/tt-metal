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
#include "impl/buffers/dram_receiver_config_page.hpp"
#include "impl/buffers/dram_sender_state_block.hpp"
#include "impl/buffers/dram_sender_topology.hpp"
#include "impl/buffers/global_circular_buffer_dram_sender_internal.hpp"
#include "impl/buffers/global_circular_buffer_impl.hpp"
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

// Bytes of one DRAM sender's pages_sent/pages_acked counters for `max_num_receivers_per_sender` receivers: packed
// uint32 pairs for worker receivers, one line per counter for DRAM receivers (dram_receiver_config_page.hpp).
uint32_t sender_counters_bytes(bool dram_receivers, uint32_t max_num_receivers_per_sender) {
    return dram_receivers ? dram_receiver_sender_counters_bytes(max_num_receivers_per_sender)
                          : 2 * sizeof(uint32_t) * max_num_receivers_per_sender;
}

// Body shared by the public Worker ctor and the private DRAM-sender ctor (with tag).
// Populates the core sets and reports the per-sender max receiver count.
void initialize_global_circular_buffer(
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    bool is_dram_sender,
    CoreRangeSet& sender_cores_out,
    CoreRangeSet& receiver_cores_out,
    CoreRangeSet& all_cores_out,
    uint32_t& max_num_receivers_per_sender_out) {
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

std::vector<std::vector<CoreCoord>> row_wise_receivers_per_sender(
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping) {
    std::vector<std::vector<CoreCoord>> receivers_per_sender;
    receivers_per_sender.reserve(sender_receiver_core_mapping.size());
    for (const auto& [_sender_core, receivers] : sender_receiver_core_mapping) {
        // Row-wise, to match both the dual-sender ceil/floor split (select_from_corerangeset
        // with row_wise=true) and the validator's receiver flatten. The slab index
        // recv_index_base+r maps to the r-th receiver in this order, so all three must agree;
        // they only diverge when a bank's receiver set spans multiple rows and columns.
        receivers_per_sender.push_back(corerange_to_cores(receivers, /*max_cores=*/std::nullopt, /*row_wise=*/true));
    }
    return receivers_per_sender;
}

// A DRAM receiver must be its own core: one CCE runs one Program at a time, so a receiver that is also
// a sender, or two entries naming one core, would lose a launch. Two logical coords can name one core,
// so compare package indices on every device.
void reject_conflicting_dram_receivers(
    const distributed::MeshDevice& mesh_device,
    const std::vector<CoreCoord>& senders,
    const std::vector<std::vector<CoreCoord>>& receivers_per_sender) {
    const Cluster& cluster = MetalContext::instance(mesh_device.impl().get_context_id()).get_cluster();
    for (const IDevice* device : mesh_device.impl().get_devices()) {
        const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(device->id());
        std::unordered_map<uint32_t, CoreCoord> sender_by_index;
        for (const CoreCoord& sender_logical : senders) {
            sender_by_index.emplace(package_dram_core_index(soc_desc, sender_logical), sender_logical);
        }
        std::unordered_map<uint32_t, CoreCoord> receiver_by_index;
        for (const std::vector<CoreCoord>& receivers : receivers_per_sender) {
            for (const CoreCoord& receiver_logical : receivers) {
                const uint32_t core_index = package_dram_core_index(soc_desc, receiver_logical);
                const std::unordered_map<uint32_t, CoreCoord>::const_iterator sender_it =
                    sender_by_index.find(core_index);
                TT_FATAL(
                    sender_it == sender_by_index.end(),
                    "DRAM receiver {} is core {} on device {}, which is also this GlobalCircularBuffer's sender {}",
                    receiver_logical.str(),
                    core_index,
                    device->id(),
                    sender_it->second.str());
                const auto [receiver_it, inserted] = receiver_by_index.try_emplace(core_index, receiver_logical);
                TT_FATAL(
                    inserted || receiver_it->second != receiver_logical,
                    "DRAM receiver {} is listed more than once",
                    receiver_logical.str());
                TT_FATAL(
                    inserted,
                    "DRAM receivers {} and {} are logical aliases of core {} on device {}",
                    receiver_it->second.str(),
                    receiver_logical.str(),
                    core_index,
                    device->id());
            }
        }
    }
}

}  // namespace

GlobalCircularBufferImpl::GlobalCircularBufferImpl(
    distributed::MeshDevice& device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) :
    device_(&device),
    sender_receiver_core_mapping_(sender_receiver_core_mapping),
    size_(size),
    buffer_type_(buffer_type),
    sender_core_type_value_(static_cast<uint8_t>(experimental::SenderCoreType::Worker)) {
    uint32_t max_num_receivers_per_sender = 0;
    initialize_global_circular_buffer(
        sender_receiver_core_mapping,
        /*is_dram_sender=*/false,
        sender_cores_,
        receiver_cores_,
        all_cores_,
        max_num_receivers_per_sender);
    this->setup_cb_buffers(buffer_type, max_num_receivers_per_sender);
}

GlobalCircularBufferImpl::GlobalCircularBufferImpl(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    std::vector<std::vector<CoreCoord>> receiver_logical_cores_per_sender,
    uint32_t size,
    BufferType buffer_type,
    DramSenderTag) :
    device_(&mesh_device),
    sender_receiver_core_mapping_(sender_receiver_core_mapping),
    size_(size),
    buffer_type_(buffer_type),
    receiver_core_type_(mesh_device.arch() == tt::ARCH::QUASAR ? CoreType::DRAM : CoreType::WORKER),
    sender_core_type_value_(static_cast<uint8_t>(experimental::SenderCoreType::Dram)),
    receiver_logical_cores_per_sender_(std::move(receiver_logical_cores_per_sender)) {
    const auto context_id = mesh_device.impl().get_context_id();
    const auto& hal = MetalContext::instance(context_id).hal();
    TT_FATAL(
        hal.has_programmable_core_type(HalProgrammableCoreType::DRAM),
        "DRAM-sender GlobalCircularBuffer requires programmable DRAM cores, which auto-enable on Blackhole "
        "with firmware >= 19.12.0.0");
    uint32_t max_num_receivers_per_sender = 0;
    initialize_global_circular_buffer(
        sender_receiver_core_mapping,
        /*is_dram_sender=*/true,
        sender_cores_,
        receiver_cores_,
        all_cores_,
        max_num_receivers_per_sender);

    validate_dram_senders_across_mesh(&mesh_device, sender_receiver_core_mapping);

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
    // DRAM receivers instead take a pages_sent block and a pages_acked block, each counter on its
    // own line (dram_receiver_config_page.hpp).
    const uint32_t counters_alignment = has_dram_receivers()
                                            ? kDramReceiverLineBytes
                                            : MetalContext::instance(context_id).hal().get_alignment(HalMemType::L1);
    const uint32_t pages_sent_bytes = sender_counters_bytes(has_dram_receivers(), max_num_receivers_per_sender);
    const uint32_t pages_sent_region = tt::align(pages_sent_bytes, counters_alignment);
    const uint32_t state_block_size = dram_sender_state_block_size(max_num_receivers_per_sender);
    drisc_sender_state_alloc_ =
        mesh_device.impl().drisc_l1_arena().allocate(pages_sent_region + state_block_size, counters_alignment);
    pages_sent_drisc_l1_base_ = drisc_sender_state_alloc_->addr();
    sender_state_drisc_l1_base_ = pages_sent_drisc_l1_base_ + pages_sent_region;

    if (has_dram_receivers()) {
        TT_FATAL(
            size_ > 0 && size_ % kDramReceiverCreditUnitBytes == 0,
            "DRAM-receiver GlobalCircularBuffer size {} B must be a non-zero multiple of the {} B credit unit",
            size_,
            kDramReceiverCreditUnitBytes);
        const uint32_t receiver_zone_size = mesh_device.impl().drisc_l1_arena().receiver_zone_size();
        TT_FATAL(
            static_cast<uint64_t>(kDramReceiverConfigPageBytes) + size_ <= receiver_zone_size,
            "DRAM-receiver GlobalCircularBuffer size {} B and its {} B config page exceed the {} B DRISC L1 receiver "
            "zone",
            size_,
            kDramReceiverConfigPageBytes,
            receiver_zone_size);
        drisc_receiver_alloc_ = mesh_device.impl().drisc_l1_arena().allocate_receiver(
            kDramReceiverConfigPageBytes + size_, kDramReceiverLineBytes);
        pages_sent_receiver_l1_base_ = drisc_receiver_alloc_->addr() + kDramReceiverCreditLineOffset;
        this->write_dram_receiver_config_pages();
    } else {
        this->setup_cb_buffers(buffer_type, max_num_receivers_per_sender);
    }
    this->initialize_dram_sender_state_block(max_num_receivers_per_sender);
}

void GlobalCircularBufferImpl::initialize_dram_sender_state_block(uint32_t max_num_receivers_per_sender) {
    // The block was already allocated by the DRAM-sender ctor (combined with the
    // pages_sent region); here we just compose its bytes and write them to L1.
    const uint32_t state_block_size = dram_sender_state_block_size(max_num_receivers_per_sender);

    // The receiver NOC XY table follows the fixed struct; config_ptr points at the
    // sender config block embedded inside the struct.
    const auto noc_xy_table_addr = static_cast<uint32_t>(sender_state_drisc_l1_base_) + sizeof(DramSenderStateBlock);
    const auto config_block_addr =
        static_cast<uint32_t>(sender_state_drisc_l1_base_) + offsetof(DramSenderStateBlock, is_sender);
    const uint32_t buffer_address = static_cast<uint32_t>(this->buffer_address());

    const uint32_t packed_num_recv_and_remote =
        remote_cb_pack(max_num_receivers_per_sender, static_cast<uint32_t>(pages_sent_receiver_l1_base_));

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

    auto* noc_xy_words = reinterpret_cast<uint32_t*>(block_bytes.data() + sizeof(DramSenderStateBlock));

    const auto& devices = device_->impl().get_devices();
    const Cluster& cluster = MetalContext::instance(extract_context_id(device_)).get_cluster();
    const std::vector<uint8_t> pages_sent_zero_bytes(
        sender_counters_bytes(has_dram_receivers(), max_num_receivers_per_sender), 0);
    for (size_t s = 0; s < sender_receiver_core_mapping_.size(); ++s) {
        const CoreCoord& sender_logical = sender_receiver_core_mapping_[s].first;
        const std::vector<CoreCoord>& receivers_vec = receiver_logical_cores_per_sender_[s];
        const uint32_t this_num_receivers = static_cast<uint32_t>(receivers_vec.size());

        // Per-sender header fields (the rest of block_bytes is constant across senders).
        hdr->num_receivers = this_num_receivers;
        hdr->num_receivers_and_remote_pages_sent_ptr =
            remote_cb_pack(this_num_receivers, static_cast<uint32_t>(pages_sent_receiver_l1_base_));

        for (IDevice* dev : devices) {
            for (uint32_t i = 0; i < max_num_receivers_per_sender; ++i) {
                if (i < this_num_receivers && has_dram_receivers()) {
                    noc_xy_words[2 * i + 0] =
                        package_dram_core_index(cluster.get_soc_desc(dev->id()), receivers_vec[i]);
                    noc_xy_words[2 * i + 1] = 0;
                } else if (i < this_num_receivers) {
                    const CoreCoord receiver_phys = dev->worker_core_from_logical_core(receivers_vec[i]);
                    noc_xy_words[2 * i + 0] = static_cast<uint32_t>(receiver_phys.x);
                    noc_xy_words[2 * i + 1] = static_cast<uint32_t>(receiver_phys.y);
                } else {
                    noc_xy_words[2 * i + 0] = 0;
                    noc_xy_words[2 * i + 1] = 0;
                }
            }
            write_dram_core_l1(
                *device_,
                dev,
                sender_logical,
                pages_sent_drisc_l1_base_,
                std::as_bytes(std::span(pages_sent_zero_bytes)));
            write_dram_core_l1(
                *device_, dev, sender_logical, sender_state_drisc_l1_base_, std::as_bytes(std::span(block_bytes)));
        }
    }
}

void GlobalCircularBufferImpl::write_dram_receiver_config_pages() {
    const Cluster& cluster = MetalContext::instance(extract_context_id(device_)).get_cluster();
    const uint32_t buffer_address = static_cast<uint32_t>(this->buffer_address());
    std::vector<uint32_t> page_words(kDramReceiverConfigPageBytes / sizeof(uint32_t), 0);
    page_words[kDramReceiverFifoStartAddrOffset / sizeof(uint32_t)] = buffer_address;
    page_words[kDramReceiverFifoSizeOffset / sizeof(uint32_t)] = size_;
    page_words[kDramReceiverFifoRdPtrOffset / sizeof(uint32_t)] = buffer_address;
    for (IDevice* dev : device_->impl().get_devices()) {
        const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(dev->id());
        for (size_t s = 0; s < sender_receiver_core_mapping_.size(); ++s) {
            const std::vector<CoreCoord>& receivers_vec = receiver_logical_cores_per_sender_[s];
            const uint32_t this_num_receivers = static_cast<uint32_t>(receivers_vec.size());
            const DeviceAddr pages_acked_base =
                pages_sent_drisc_l1_base_ + dram_receiver_sender_pages_acked_offset(this_num_receivers);
            page_words[kDramReceiverSenderIndexOffset / sizeof(uint32_t)] =
                package_dram_core_index(soc_desc, sender_receiver_core_mapping_[s].first);
            for (uint32_t i = 0; i < this_num_receivers; ++i) {
                page_words[kDramReceiverRemotePagesAckedAddrOffset / sizeof(uint32_t)] =
                    static_cast<uint32_t>(pages_acked_base + (i * kDramReceiverCreditStrideBytes));
                write_dram_core_l1(
                    *device_, dev, receivers_vec[i], config_address(), std::as_bytes(std::span(page_words)));
            }
        }
    }
}

void GlobalCircularBufferImpl::setup_cb_buffers(BufferType buffer_type, uint32_t max_num_receivers_per_sender) {
    TT_FATAL(
        buffer_type == BufferType::L1 or buffer_type == BufferType::L1_SMALL,
        "Global circular buffer can only be created for L1 buffer types");
    uint32_t num_cores = all_cores_.num_cores();

    auto shard_parameters = ShardSpecBuffer(all_cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});

    uint32_t cb_buffer_size = size_ * num_cores;
    cb_buffer_ = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = cb_buffer_size},
        distributed::DeviceLocalBufferConfig{
            .page_size = size_,
            .buffer_type = buffer_type,
            .sharding_args = BufferShardingArgs(shard_parameters, TensorMemoryLayout::HEIGHT_SHARDED),
        },
        device_);

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
    cb_config_buffer_ = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = cb_config_size},
        distributed::DeviceLocalBufferConfig{
            .page_size = cb_config_page_size,
            .buffer_type = buffer_type,
            .sharding_args = BufferShardingArgs(std::move(shard_parameters), TensorMemoryLayout::HEIGHT_SHARDED),
        },
        device_);

    // Write the config buffer to the device
    // Only block for the slow dispatch case
    auto config_buffer_address = cb_config_buffer_->address();
    const auto& core_to_core_id = cb_config_buffer_->get_reference_buffer()->get_buffer_page_mapping()->core_to_core_id;
    uint32_t noc_xy_address = config_buffer_address + (num_config_elements * sizeof(uint32_t));
    uint32_t pages_sent_address = tt::align(noc_xy_address + (num_noc_xy_words * sizeof(uint32_t)), l1_alignment);
    auto buffer_address = cb_buffer().address();

    const auto sender_core_type = static_cast<experimental::SenderCoreType>(sender_core_type_value_);
    if (sender_core_type == experimental::SenderCoreType::Dram) {
        // pages_sent_drisc_l1_base_ has already been populated by the DRAM-sender ctor from
        // the per-mesh DriscL1Arena. We just need to expose the worker-local pages_sent base
        // here so the DRISC kernel can NOC-inc pages_sent into the receivers' config pages.
        pages_sent_receiver_l1_base_ = pages_sent_address;
    }

    const auto make_config_host_buffer = [&](IDevice* config_device) {
        std::vector<uint32_t> cb_config_host_buffer(cb_config_size / sizeof(uint32_t), 0);
        for (const auto& [sender_logical, receiver_cores] : sender_receiver_core_mapping_) {
            // Row-wise: this vector's index is the receiver's credit slot -- it picks both the
            // sender's NOC XY table order and each receiver's pages_sent/pages_acked offsets below,
            // and the sender walks the two in lockstep. A DRAM sender's table is built from
            // receiver_logical_cores_per_sender_ (also row-wise), so a column-wise order here would
            // credit the wrong receiver whenever a sender's receiver set spans several rows and
            // columns. Worker senders take their table from this same vector, so they stay
            // self-consistent either way.
            const auto& receiver_cores_vec =
                corerange_to_cores(receiver_cores, /*max_cores=*/std::nullopt, /*row_wise=*/true);
            uint32_t num_receivers = receiver_cores.num_cores();

            // Worker senders have their own config page in the sharded buffer; DRAM senders don't
            // (the DRISC kernel hand-rolls the sender iface state from compile-time args).
            if (sender_core_type == experimental::SenderCoreType::Worker) {
                uint32_t sender_idx = core_to_core_id.at(sender_logical) * cb_config_page_size / sizeof(uint32_t);
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
            // worker senders this is the worker phys; for DRAM senders it is `config_device`'s own
            // DRAM virtual coord for this sender's bank/role, which the logical coord does not pin
            // down (see metal_SocDescriptor::dram_bank_endpoint_coords).
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

    if (sender_core_type == experimental::SenderCoreType::Dram) {
        // The DRAM sender's virtual coord is per device (its bank/role resolves against that
        // device's DRAM harvest mask), so each device gets its own config page. One
        // enqueue_write_shards fans the shards out in parallel and syncs once, rather than
        // draining the queue per device.
        const distributed::MeshCoordinateRange device_coords(device_->shape());
        std::vector<std::vector<uint32_t>> per_device_config;
        per_device_config.reserve(device_coords.shape().mesh_size());
        std::vector<distributed::ShardDataTransfer> shard_data_transfers;
        shard_data_transfers.reserve(device_coords.shape().mesh_size());
        for (const auto& coord : device_coords) {
            // A mesh view can span slots whose devices belong to another host/rank; get_device
            // TT_FATALs on those. Writing only the local shards matches what the broadcast path
            // below does (enqueue_write_shard_to_sub_grid skips non-local coords) -- each rank
            // fills in its own.
            if (!device_->is_local(coord)) {
                continue;
            }
            per_device_config.push_back(make_config_host_buffer(device_->get_device(coord)));
            shard_data_transfers.push_back(
                distributed::ShardDataTransfer(coord).host_data(per_device_config.back().data()));
        }
        device_->mesh_command_queue().enqueue_write_shards(cb_config_buffer_, shard_data_transfers, /*blocking=*/true);
    } else {
        // Every device gets the same config page, so one broadcast write covers the mesh.
        std::vector<uint32_t> cb_config_host_buffer = make_config_host_buffer(device_);
        distributed::EnqueueWriteMeshBuffer(
            device_->mesh_command_queue(), cb_config_buffer_, cb_config_host_buffer, false);
    }
}

const Buffer& GlobalCircularBufferImpl::cb_buffer() const {
    if (has_dram_receivers()) {
        TT_THROW(
            "A GlobalCircularBuffer with DRAM receivers has no backing Buffer; its FIFO and config pages are in the "
            "DRISC L1 receiver zone at buffer_address() and config_address()");
    }
    return *cb_buffer_->get_reference_buffer();
}

const CoreRangeSet& GlobalCircularBufferImpl::sender_cores() const { return sender_cores_; }

const CoreRangeSet& GlobalCircularBufferImpl::receiver_cores() const { return receiver_cores_; }

const CoreRangeSet& GlobalCircularBufferImpl::all_cores() const { return all_cores_; }

DeviceAddr GlobalCircularBufferImpl::buffer_address() const {
    if (has_dram_receivers()) {
        return drisc_receiver_alloc_->addr() + kDramReceiverConfigPageBytes;
    }
    return cb_buffer().address();
}

DeviceAddr GlobalCircularBufferImpl::config_address() const {
    if (has_dram_receivers()) {
        return drisc_receiver_alloc_->addr();
    }
    return cb_config_buffer_->address();
}

uint32_t GlobalCircularBufferImpl::size() const { return size_; }

const std::vector<std::pair<CoreCoord, CoreRangeSet>>& GlobalCircularBufferImpl::sender_receiver_core_mapping() const {
    return sender_receiver_core_mapping_;
}

// GlobalCircularBuffer (public pimpl wrapper) implementation

GlobalCircularBuffer::GlobalCircularBuffer(
    distributed::MeshDevice& device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) :
    GlobalCircularBuffer(GlobalCircularBufferImpl(device, sender_receiver_core_mapping, size, buffer_type)) {}

GlobalCircularBuffer CreateGlobalCircularBuffer(
    distributed::MeshDevice& device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) {
    return GlobalCircularBuffer(GlobalCircularBufferImpl(device, sender_receiver_core_mapping, size, buffer_type));
}

GlobalCircularBuffer::GlobalCircularBuffer(GlobalCircularBufferImpl impl) :
    impl_(std::make_unique<GlobalCircularBufferImpl>(std::move(impl))) {}

GlobalCircularBuffer::GlobalCircularBuffer(const GlobalCircularBuffer& other) :
    impl_(other.impl_ ? std::make_unique<GlobalCircularBufferImpl>(*other.impl_) : nullptr) {}

GlobalCircularBuffer& GlobalCircularBuffer::operator=(const GlobalCircularBuffer& other) {
    if (this != &other) {
        impl_ = other.impl_ ? std::make_unique<GlobalCircularBufferImpl>(*other.impl_) : nullptr;
    }
    return *this;
}

GlobalCircularBuffer::GlobalCircularBuffer(GlobalCircularBuffer&& other) noexcept = default;
GlobalCircularBuffer& GlobalCircularBuffer::operator=(GlobalCircularBuffer&& other) noexcept = default;
GlobalCircularBuffer::~GlobalCircularBuffer() = default;

GlobalCircularBufferImpl& GlobalCircularBuffer::impl() {
    TT_FATAL(impl_ != nullptr, "GlobalCircularBuffer is in a moved-from state.");
    return *impl_;
}

const GlobalCircularBufferImpl& GlobalCircularBuffer::impl() const {
    TT_FATAL(impl_ != nullptr, "GlobalCircularBuffer is in a moved-from state.");
    return *impl_;
}

const Buffer& GlobalCircularBuffer::cb_buffer() const { return impl().cb_buffer(); }

const CoreRangeSet& GlobalCircularBuffer::sender_cores() const { return impl().sender_cores(); }

const CoreRangeSet& GlobalCircularBuffer::receiver_cores() const { return impl().receiver_cores(); }

DeviceAddr GlobalCircularBuffer::buffer_address() const { return impl().buffer_address(); }

DeviceAddr GlobalCircularBuffer::config_address() const { return impl().config_address(); }

uint32_t GlobalCircularBuffer::size() const { return impl().size(); }

const std::vector<std::pair<CoreCoord, CoreRangeSet>>& GlobalCircularBuffer::sender_receiver_core_mapping() const {
    return impl().sender_receiver_core_mapping();
}

std::tuple<std::vector<std::pair<CoreCoord, CoreRangeSet>>, uint32_t, BufferType>
GlobalCircularBuffer::attribute_values() const {
    return impl().attribute_values();
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
        distributed::MeshDevice& mesh_device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
        std::vector<std::vector<CoreCoord>> receiver_logical_cores_per_sender,
        uint32_t size,
        BufferType buffer_type);

    static SenderCoreType sender_core_type(const GlobalCircularBuffer& gcb);
    static DeviceAddr pages_sent_drisc_l1_base(const GlobalCircularBuffer& gcb);
    static DeviceAddr pages_sent_receiver_l1_base(const GlobalCircularBuffer& gcb);
    static DeviceAddr sender_state_drisc_l1_base(const GlobalCircularBuffer& gcb);
    static std::shared_ptr<DriscL1Allocation> sender_state_drisc_l1_allocation(const GlobalCircularBuffer& gcb);
    static std::shared_ptr<DriscL1Allocation> receiver_drisc_l1_allocation(const GlobalCircularBuffer& gcb);
    static const std::vector<std::vector<CoreCoord>>& receiver_logical_cores_per_sender(
        const GlobalCircularBuffer& gcb);
};

GlobalCircularBuffer GlobalCircularBufferDramSenderInternals::make_dram_sender(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    std::vector<std::vector<CoreCoord>> receiver_logical_cores_per_sender,
    uint32_t size,
    BufferType buffer_type) {
    return GlobalCircularBuffer(GlobalCircularBufferImpl(
        mesh_device,
        sender_receiver_core_mapping,
        std::move(receiver_logical_cores_per_sender),
        size,
        buffer_type,
        GlobalCircularBufferImpl::DramSenderTag{}));
}

SenderCoreType GlobalCircularBufferDramSenderInternals::sender_core_type(const GlobalCircularBuffer& gcb) {
    return static_cast<SenderCoreType>(gcb.impl().sender_core_type_value_);
}

DeviceAddr GlobalCircularBufferDramSenderInternals::pages_sent_drisc_l1_base(const GlobalCircularBuffer& gcb) {
    return gcb.impl().pages_sent_drisc_l1_base_;
}

DeviceAddr GlobalCircularBufferDramSenderInternals::pages_sent_receiver_l1_base(const GlobalCircularBuffer& gcb) {
    return gcb.impl().pages_sent_receiver_l1_base_;
}

DeviceAddr GlobalCircularBufferDramSenderInternals::sender_state_drisc_l1_base(const GlobalCircularBuffer& gcb) {
    return gcb.impl().sender_state_drisc_l1_base_;
}

std::shared_ptr<DriscL1Allocation> GlobalCircularBufferDramSenderInternals::sender_state_drisc_l1_allocation(
    const GlobalCircularBuffer& gcb) {
    return gcb.impl().drisc_sender_state_alloc_;
}

std::shared_ptr<DriscL1Allocation> GlobalCircularBufferDramSenderInternals::receiver_drisc_l1_allocation(
    const GlobalCircularBuffer& gcb) {
    return gcb.impl().drisc_receiver_alloc_;
}

const std::vector<std::vector<CoreCoord>>& GlobalCircularBufferDramSenderInternals::receiver_logical_cores_per_sender(
    const GlobalCircularBuffer& gcb) {
    return gcb.impl().receiver_logical_cores_per_sender_;
}

}  // namespace global_circular_buffer_dram_sender

GlobalCircularBuffer CreateGlobalCircularBufferForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type,
    bool support_multi_receiver_shards) {
    if (mesh_device.arch() == tt::ARCH::QUASAR) {
        TT_THROW(
            "Quasar has no worker receivers: pass each bank's receiver CCEs as ordered DRAM-logical coords to the "
            "CreateGlobalCircularBufferForTensorPrefetcher overload that takes bank_to_dram_receivers");
    }
    // Multi-receiver shards (legacy interleaved layout) force one sender per bank; the
    // receiver-contiguous layout that disallows them is what lets a bank use two senders.
    auto mapping = build_dram_sender_mapping(
        &mesh_device,
        bank_to_receivers,
        support_multi_receiver_shards ? DramSenderSplit::OnePerBank : DramSenderSplit::TwoPerBank);
    std::vector<std::vector<CoreCoord>> receivers_per_sender = row_wise_receivers_per_sender(mapping);
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::make_dram_sender(
        mesh_device, mapping, std::move(receivers_per_sender), size, buffer_type);
}

GlobalCircularBuffer CreateGlobalCircularBufferForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, std::vector<CoreCoord>>>& bank_to_dram_receivers,
    uint32_t size) {
    if (mesh_device.arch() != tt::ARCH::QUASAR) {
        TT_THROW(
            "DRAM receivers are Quasar CCEs; on {} pass worker receivers as a CoreRangeSet per bank",
            mesh_device.arch());
    }
    const std::vector<IDevice*> devices = mesh_device.impl().get_devices();
    TT_FATAL(!devices.empty(), "Cannot build a DRAM-receiver GlobalCircularBuffer for a mesh with no local devices");
    std::vector<CoreCoord> senders;
    std::vector<std::vector<CoreCoord>> receivers_per_sender;
    std::unordered_set<uint32_t> seen_banks;
    for (const auto& [bank_id, receivers] : bank_to_dram_receivers) {
        TT_FATAL(!receivers.empty(), "DRAM bank {} has no receivers", bank_id);
        TT_FATAL(seen_banks.insert(bank_id).second, "DRAM bank {} appears more than once", bank_id);
        senders.push_back(mesh_device.impl().dram_sender_logical_cores(devices.front(), bank_id).front());
        receivers_per_sender.push_back(receivers);
    }
    reject_conflicting_dram_receivers(mesh_device, senders, receivers_per_sender);

    std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping;
    mapping.reserve(senders.size());
    for (size_t s = 0; s < senders.size(); ++s) {
        mapping.emplace_back(senders[s], CoreRangeSet(ttsl::Span<const CoreCoord>(receivers_per_sender[s])));
    }
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::make_dram_sender(
        mesh_device, mapping, std::move(receivers_per_sender), size, BufferType::L1);
}

SenderCoreType sender_core_type(const GlobalCircularBuffer& gcb) {
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::sender_core_type(gcb);
}

DeviceAddr pages_sent_drisc_l1_base(const GlobalCircularBuffer& gcb) {
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::pages_sent_drisc_l1_base(gcb);
}

DeviceAddr pages_sent_receiver_l1_base(const GlobalCircularBuffer& gcb) {
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::pages_sent_receiver_l1_base(
        gcb);
}

DeviceAddr sender_state_drisc_l1_base(const GlobalCircularBuffer& gcb) {
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::sender_state_drisc_l1_base(gcb);
}

std::shared_ptr<DriscL1Allocation> sender_state_drisc_l1_allocation(const GlobalCircularBuffer& gcb) {
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::
        sender_state_drisc_l1_allocation(gcb);
}

std::shared_ptr<DriscL1Allocation> receiver_drisc_l1_allocation(const GlobalCircularBuffer& gcb) {
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::receiver_drisc_l1_allocation(
        gcb);
}

const std::vector<std::vector<CoreCoord>>& receiver_logical_cores_per_sender(const GlobalCircularBuffer& gcb) {
    return global_circular_buffer_dram_sender::GlobalCircularBufferDramSenderInternals::
        receiver_logical_cores_per_sender(gcb);
}

}  // namespace tt::tt_metal::experimental

namespace std {

std::size_t hash<tt::tt_metal::experimental::GlobalCircularBuffer>::operator()(
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_circular_buffer) const {
    return ttsl::hash::hash_objects_with_default_seed(global_circular_buffer.attribute_values());
}

}  // namespace std
