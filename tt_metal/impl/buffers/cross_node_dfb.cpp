// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <buffer.hpp>
#include <buffer_types.hpp>
#include <core_coord.hpp>
#include <device.hpp>
#include "impl/dataflow_buffer/cross_node_dfb.hpp"
#include <host_api.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/context/context_types.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/api/tt-metalium/hal_types.hpp"
#include <tt_align.hpp>
#include <tt_metal.hpp>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

#include "distributed.hpp"
#include "mesh_buffer.hpp"
#include "mesh_device.hpp"
#include <tt_stl/reflection.hpp>
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include <umd/device/types/xy_pair.hpp>

// Device-visible constant: number of words per remote DFB kernel-config entry.
static constexpr uint32_t UINT32_WORDS_PER_REMOTE_DFB_CONFIG = 2;

// Bit flag packed into the entry_size word of the kernel-config entry.
// Bit 31 set = auto_commit enabled for this slot.
static constexpr uint32_t REMOTE_DFB_AUTO_COMMIT_FLAG = (1u << 31);

namespace tt::tt_metal::experimental {

namespace {

void initialize_cross_node_dfb(
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
        num_receiver_cores += n;
        max_receivers = std::max(max_receivers, n);
        sender_ranges.emplace_back(sender_core);
        receiver_cores_out = receiver_cores_out.merge(receiver_set);
    }

    sender_cores_out = CoreRangeSet(sender_ranges);
    TT_FATAL(num_sender_cores == sender_cores_out.num_cores(), "Duplicate sender cores in sender_receiver_mapping");
    TT_FATAL(num_receiver_cores == receiver_cores_out.num_cores(),
        "Duplicate receiver cores detected across sender groups (receiver sets must be disjoint)");

    all_cores_out = sender_cores_out.merge(receiver_cores_out);
    TT_FATAL(
        all_cores_out.num_cores() == num_sender_cores + num_receiver_cores,
        "Sender and receiver core sets must be disjoint");

    max_num_receivers_per_sender_out = max_receivers;
}

}  // namespace

CrossNodeDFB::CrossNodeDFB(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type) :
    device_(device),
    sender_receiver_mapping_(sender_receiver_mapping),
    entry_size_(entry_size),
    num_entries_(num_entries) {

    uint32_t max_num_receivers_per_sender = 0;
    initialize_cross_node_dfb(
        device,
        sender_receiver_mapping,
        sender_cores_,
        receiver_cores_,
        all_cores_,
        max_num_receivers_per_sender);

    this->setup_buffers(buffer_type, max_num_receivers_per_sender);
}

void CrossNodeDFB::setup_buffers(BufferType buffer_type, uint32_t max_num_receivers_per_sender) {
    TT_FATAL(
        buffer_type == BufferType::L1 || buffer_type == BufferType::L1_SMALL,
        "CrossNodeDFB can only use L1 buffer types");

    const auto context_id = extract_context_id(device_);
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);

    TT_FATAL(entry_size_ > 0, "entry_size must be > 0");
    TT_FATAL(
        entry_size_ % l1_alignment == 0,
        "entry_size {} must be a multiple of L1_ALIGNMENT {}",
        entry_size_,
        l1_alignment);
    TT_FATAL(num_entries_ > 0, "num_entries must be > 0");

    const uint32_t num_all_cores = all_cores_.num_cores();

    // --- Data ring buffer (sharded over all cores) ---
    // One ring FIFO per receiver: size = entry_size * num_entries.
    const uint32_t ring_size = entry_size_ * num_entries_;
    auto shard_params_data = ShardSpecBuffer(
        all_cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_all_cores, 1});
    ShardedBufferConfig data_shard_cfg = {
        .device         = device_,
        .size           = ring_size * num_all_cores,
        .page_size      = ring_size,
        .buffer_type    = buffer_type,
        .buffer_layout  = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(shard_params_data),
    };
    dfb_buffer_ = distributed::AnyBuffer::create(data_shard_cfg);

    // --- Config sideband buffer (sharded over all_cores = senders ∪ receivers) ---
    // Config page layout per core (words):
    //   [0]  is_sender
    //   [1]  num_receivers
    //   [2]  fifo_start_addr
    //   [3]  fifo_size (entry_size * num_entries)
    //   [4]  fifo_wr_ptr / fifo_rd_ptr (cross-program checkpoint)
    //   [5]  noc_xy_ptr: address of word[8] (start of NOC XY / sender-coord data)
    //   [6]  aligned_entries_sent_ptr or entries_sent slot addr
    //   [7]  remote_entries_acked_ptr
    // Sender pages additionally store:
    //   words[8..8+2N-1] = NOC XY table: x0,y0,x1,y1,... for N receivers
    //   Then entries_sent[i] / entries_acked[i] pairs at L1_ALIGNMENT stride
    // Receiver pages additionally store:
    //   word[8] = sender_physical_coord.x
    //   word[9] = sender_physical_coord.y

    constexpr uint32_t num_header_words = 8;
    const uint32_t noc_xy_words  = 2 * max_num_receivers_per_sender;
    const uint32_t header_bytes  = num_header_words * sizeof(uint32_t);
    const uint32_t noc_xy_bytes  = noc_xy_words * sizeof(uint32_t);
    const uint32_t counters_size = 2 * max_num_receivers_per_sender * l1_alignment;
    const uint32_t config_page_size =
        tt::align(header_bytes + noc_xy_bytes, l1_alignment) + counters_size;

    auto shard_params_cfg = ShardSpecBuffer(
        all_cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_all_cores, 1});
    ShardedBufferConfig cfg_shard_config = {
        .device         = device_,
        .size           = config_page_size * num_all_cores,
        .page_size      = config_page_size,
        .buffer_type    = buffer_type,
        .buffer_layout  = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(shard_params_cfg),
    };
    config_buffer_ = distributed::AnyBuffer::create(cfg_shard_config);

    // Write config pages to device.
    const uint32_t config_base_addr = static_cast<uint32_t>(config_buffer_.get_buffer()->address());
    const auto& core_to_core_id = config_buffer_.get_buffer()->get_buffer_page_mapping()->core_to_core_id;
    const uint32_t data_base_addr = static_cast<uint32_t>(dfb_buffer_.get_buffer()->address());

    // Mirror GlobalCB layout: noc_xy table at word[8], then L1-aligned pages_sent/pages_acked pairs.
    const uint32_t noc_xy_address = config_base_addr + num_header_words * sizeof(uint32_t);
    const uint32_t pages_sent_address =
        tt::align(noc_xy_address + noc_xy_words * sizeof(uint32_t), l1_alignment);

    std::vector<uint32_t> config_host_buffer(config_page_size * num_all_cores / sizeof(uint32_t), 0);

    for (const auto& [sender_core, receiver_set] : sender_receiver_mapping_) {
        const auto receiver_vec = corerange_to_cores(receiver_set);
        const uint32_t num_recv = static_cast<uint32_t>(receiver_vec.size());

        // --- Sender config page (matches GlobalCircularBuffer worker-sender layout) ---
        uint32_t si = core_to_core_id.at(sender_core) * config_page_size / sizeof(uint32_t);
        config_host_buffer[si++] = 1;                // is_sender
        config_host_buffer[si++] = num_recv;         // num_receivers
        config_host_buffer[si++] = data_base_addr;   // fifo_start_addr
        config_host_buffer[si++] = ring_size;        // fifo_size
        config_host_buffer[si++] = data_base_addr;   // fifo_wr_ptr checkpoint
        config_host_buffer[si++] = noc_xy_address;   // noc_xy_ptr → word[8]
        config_host_buffer[si++] = pages_sent_address; // aligned_pages_sent_ptr
        // Sharded layout: remote pages_sent target equals local pages_sent base.
        config_host_buffer[si++] = pages_sent_address;
        for (uint32_t ri = 0; ri < num_recv; ++ri) {
            auto phys = device_->worker_core_from_logical_core(receiver_vec[ri]);
            config_host_buffer[si++] = static_cast<uint32_t>(phys.x);
            config_host_buffer[si++] = static_cast<uint32_t>(phys.y);
        }
        // entries_sent/entries_acked pairs are zero-initialized in config_host_buffer.

        // --- Receiver config pages (matches GlobalCircularBuffer receiver layout) ---
        const auto sender_phys = device_->worker_core_from_logical_core(sender_core);
        for (uint32_t ri = 0; ri < num_recv; ++ri) {
            uint32_t rci = core_to_core_id.at(receiver_vec[ri]) * config_page_size / sizeof(uint32_t);
            config_host_buffer[rci++] = 0;  // is_sender
            config_host_buffer[rci++] = num_recv;
            config_host_buffer[rci++] = data_base_addr;
            config_host_buffer[rci++] = ring_size;
            config_host_buffer[rci++] = data_base_addr;  // fifo_rd_ptr checkpoint
            config_host_buffer[rci++] = noc_xy_address;  // points to word[8] on this core's page
            // This receiver's local pages_sent slot; pages_acked at +L1_ALIGNMENT.
            config_host_buffer[rci++] = pages_sent_address + 2 * ri * l1_alignment;
            // Canonical remote pages_acked target on the sender (same numeric L1 offset for sharded GCB).
            config_host_buffer[rci++] = pages_sent_address + 2 * ri * l1_alignment + l1_alignment;
            config_host_buffer[rci++] = static_cast<uint32_t>(sender_phys.x);
            config_host_buffer[rci++] = static_cast<uint32_t>(sender_phys.y);
        }
    }

    if (auto mesh_buffer = config_buffer_.get_mesh_buffer()) {
        distributed::EnqueueWriteMeshBuffer(
            mesh_buffer->device()->mesh_command_queue(), mesh_buffer, config_host_buffer, true);
    } else {
        tt::tt_metal::detail::WriteToBuffer(
            *config_buffer_.get_buffer(),
            tt::stl::Span<const uint8_t>(
                reinterpret_cast<const uint8_t*>(config_host_buffer.data()),
                config_host_buffer.size() * sizeof(uint32_t)));
    }
}

// Accessors -------------------------------------------------------------------

const Buffer& CrossNodeDFB::dfb_buffer() const { return *dfb_buffer_.get_buffer(); }
const Buffer& CrossNodeDFB::config_buffer() const { return *config_buffer_.get_buffer(); }
uint32_t CrossNodeDFB::buffer_address() const { return static_cast<uint32_t>(dfb_buffer().address()); }
uint32_t CrossNodeDFB::config_address() const { return static_cast<uint32_t>(config_buffer().address()); }
uint32_t CrossNodeDFB::entry_size() const { return entry_size_; }
uint32_t CrossNodeDFB::num_entries() const { return num_entries_; }
const CoreRangeSet& CrossNodeDFB::sender_cores() const { return sender_cores_; }
const CoreRangeSet& CrossNodeDFB::receiver_cores() const { return receiver_cores_; }
const CoreRangeSet& CrossNodeDFB::all_cores() const { return all_cores_; }
const std::vector<std::pair<CoreCoord, CoreRangeSet>>&
CrossNodeDFB::sender_receiver_core_mapping() const { return sender_receiver_mapping_; }

// Free functions --------------------------------------------------------------

CrossNodeDFB CreateCrossNodeDFB(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type) {
    return CrossNodeDFB(device, sender_receiver_mapping, entry_size, num_entries, buffer_type);
}

uint8_t AttachCrossNodeDFB(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CrossNodeDFB& gdfb,
    const std::vector<std::string>& relay_dfb_names,
    bool auto_commit) {

    IDevice* device = gdfb.get_device();
    TT_FATAL(device != nullptr, "CrossNodeDFB has a null device");

    // Resolve core_spec to CoreRangeSet.
    CoreRangeSet cores;
    if (std::holds_alternative<CoreCoord>(core_spec)) {
        cores = CoreRangeSet({CoreRange(std::get<CoreCoord>(core_spec))});
    } else if (std::holds_alternative<CoreRange>(core_spec)) {
        cores = CoreRangeSet({std::get<CoreRange>(core_spec)});
    } else {
        cores = std::get<CoreRangeSet>(core_spec);
    }

    // Validate: cores must be a subset of gdfb's all_cores.
    TT_FATAL(
        gdfb.all_cores().contains(cores),
        "AttachCrossNodeDFB: core_spec {} is not a subset of CrossNodeDFB all_cores {}",
        cores.str(),
        gdfb.all_cores().str());

    return program.impl().attach_cross_node_dfb(cores, gdfb, relay_dfb_names, auto_commit);
}

void UpdateDynamicCrossNodeDFBAddress(Program& program, const CrossNodeDFB& gdfb) {
    program.impl().update_dynamic_cross_node_dfb_address(gdfb);
}

}  // namespace tt::tt_metal::experimental

namespace std {

std::size_t hash<tt::tt_metal::experimental::CrossNodeDFB>::operator()(
    const tt::tt_metal::experimental::CrossNodeDFB& gdfb) const {
    return tt::stl::hash::hash_objects_with_default_seed(gdfb.attribute_values());
}

}  // namespace std
