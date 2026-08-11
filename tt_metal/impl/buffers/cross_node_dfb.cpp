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
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include <umd/device/types/xy_pair.hpp>

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
    TT_FATAL(
        num_receiver_cores == receiver_cores_out.num_cores(),
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
    initialize_cross_node_dfb(
        device, sender_receiver_mapping, sender_cores_, receiver_cores_, all_cores_, max_num_receivers_per_sender_);

    this->setup_buffers(buffer_type);
}

CrossNodeDFB::CrossNodeDFB(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    uint32_t entry_size,
    uint32_t num_entries,
    Buffer& data_buffer) :
    device_(device),
    sender_receiver_mapping_(sender_receiver_mapping),
    entry_size_(entry_size),
    num_entries_(num_entries) {
    initialize_cross_node_dfb(
        device, sender_receiver_mapping, sender_cores_, receiver_cores_, all_cores_, max_num_receivers_per_sender_);

    this->setup_buffers_with_borrowed_data(data_buffer);
}

namespace {

void validate_entry_geometry(IDevice* device, uint32_t entry_size, uint32_t num_entries) {
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
}

// Byte offsets within one config page. Every per-receiver region is sized by the largest
// fanout in the topology so all pages share one size (the buffer is sharded uniformly).
// Zeroing [counters_offset, page_size) on launch resets all credits; the sender derives
// each receiver's write position from those credits, so no cursor state is stored.
struct ConfigPageLayout {
    uint32_t noc_xy_offset;
    uint32_t counters_offset;
    uint32_t page_size;
};

ConfigPageLayout compute_config_page_layout(uint32_t max_num_receivers_per_sender, uint32_t l1_alignment) {
    constexpr uint32_t num_header_words = 8;
    const uint32_t noc_xy_offset = num_header_words * sizeof(uint32_t);
    const uint32_t counters_offset = tt::align(
        noc_xy_offset + 2 * max_num_receivers_per_sender * static_cast<uint32_t>(sizeof(uint32_t)), l1_alignment);
    const uint32_t page_size = counters_offset + 2 * max_num_receivers_per_sender * l1_alignment;
    return ConfigPageLayout{.noc_xy_offset = noc_xy_offset, .counters_offset = counters_offset, .page_size = page_size};
}

bool is_compatible_borrowed_device(IDevice* expected, IDevice* buffer_device) {
    if (expected == buffer_device) {
        return true;
    }
    if (auto* mesh = dynamic_cast<distributed::MeshDevice*>(expected)) {
        for (IDevice* local : mesh->get_devices()) {
            if (local == buffer_device) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

void CrossNodeDFB::allocate_config_and_write_pages(BufferType config_buffer_type) {
    TT_FATAL(
        config_buffer_type == BufferType::L1 || config_buffer_type == BufferType::L1_SMALL,
        "CrossNodeDFB can only use L1 buffer types");

    const auto context_id = extract_context_id(device_);
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    const uint32_t num_all_cores = all_cores_.num_cores();

    const uint32_t config_page_size = compute_config_page_layout(max_num_receivers_per_sender_, l1_alignment).page_size;

    auto shard_params_cfg =
        ShardSpecBuffer(all_cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_all_cores, 1});
    ShardedBufferConfig cfg_shard_config = {
        .device = device_,
        .size = config_page_size * num_all_cores,
        .page_size = config_page_size,
        .buffer_type = config_buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(shard_params_cfg),
    };
    config_buffer_ = distributed::AnyBuffer::create(cfg_shard_config);
    write_config_pages();
}

void CrossNodeDFB::write_config_pages() {
    TT_FATAL(config_buffer_.get_buffer() != nullptr, "CrossNodeDFB config buffer must exist before write");
    TT_FATAL(data_address_ != 0, "CrossNodeDFB data address must be set before config write");

    const auto context_id = extract_context_id(device_);
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    const uint32_t num_all_cores = all_cores_.num_cores();
    const uint32_t ring_size = entry_size_ * num_entries_;

    // --- Config sideband buffer (sharded over all_cores = senders ∪ receivers) ---
    // Config page layout per core (words) — shared with future GlobalDFB header:
    //   [0]  is_sender
    //   [1]  num_receivers
    //   [2]  fifo_start_addr
    //   [3]  fifo_size (entry_size * num_entries)
    //   [4]  fifo_wr_ptr / fifo_rd_ptr checkpoint (reserved for GlobalDFB; CrossNode FW
    //        ignores this and always inits iface ptrs from fifo_start_addr)
    //   [5]  noc_xy_ptr: address of word[8] (start of NOC XY / sender-coord data)
    //   [6]  aligned_entries_sent_ptr or entries_sent slot addr
    //   [7]  remote_entries_acked_ptr
    // Sender pages additionally store:
    //   words[8..8+2N-1] = NOC XY table: x0,y0,x1,y1,... for N receivers
    //   Then entries_sent[i] / entries_acked[i] pairs at L1_ALIGNMENT stride.
    //   No write-cursor state: the sender derives each receiver's write position from
    //   that receiver's entries_sent counter.
    // Receiver pages additionally store:
    //   word[8] = sender_physical_coord.x
    //   word[9] = sender_physical_coord.y

    const auto layout = compute_config_page_layout(max_num_receivers_per_sender_, l1_alignment);
    const uint32_t config_page_size = layout.page_size;

    const uint32_t config_base_addr = static_cast<uint32_t>(config_buffer_.get_buffer()->address());
    const auto& core_to_core_id = config_buffer_.get_buffer()->get_buffer_page_mapping()->core_to_core_id;
    const uint32_t data_base_addr = data_address_;

    const uint32_t noc_xy_address = config_base_addr + layout.noc_xy_offset;
    const uint32_t pages_sent_address = config_base_addr + layout.counters_offset;
    // Zeroing this window resets all credits, which also rewinds every receiver's
    // derived write position to the start of the ring.
    credit_reset_offset_ = layout.counters_offset;
    credit_reset_size_ = config_page_size - credit_reset_offset_;

    std::vector<uint32_t> config_host_buffer(config_page_size * num_all_cores / sizeof(uint32_t), 0);

    for (const auto& [sender_core, receiver_set] : sender_receiver_mapping_) {
        const auto receiver_vec = corerange_to_cores(receiver_set);
        const uint32_t num_recv = static_cast<uint32_t>(receiver_vec.size());

        // --- Sender config page (matches GlobalCircularBuffer worker-sender layout) ---
        uint32_t si = core_to_core_id.at(sender_core) * config_page_size / sizeof(uint32_t);
        config_host_buffer[si++] = 1;                   // is_sender
        config_host_buffer[si++] = num_recv;            // num_receivers
        config_host_buffer[si++] = data_base_addr;      // fifo_start_addr
        config_host_buffer[si++] = ring_size;           // fifo_size
        config_host_buffer[si++] = data_base_addr;      // word[4]: reserved checkpoint (FW ignores for CrossNode)
        config_host_buffer[si++] = noc_xy_address;      // noc_xy_ptr → word[8]
        config_host_buffer[si++] = pages_sent_address;  // aligned_pages_sent_ptr
        // Sharded layout: remote pages_sent target equals local pages_sent base.
        config_host_buffer[si++] = pages_sent_address;
        for (uint32_t ri = 0; ri < num_recv; ++ri) {
            auto phys = device_->worker_core_from_logical_core(receiver_vec[ri]);
            config_host_buffer[si++] = static_cast<uint32_t>(phys.x);
            config_host_buffer[si++] = static_cast<uint32_t>(phys.y);
        }
        // entries_sent/entries_acked pairs are zero-initialized in config_host_buffer;
        // zero credits put every derived write position at the start of the ring.

        // --- Receiver config pages (matches GlobalCircularBuffer receiver layout) ---
        const auto sender_phys = device_->worker_core_from_logical_core(sender_core);
        for (uint32_t ri = 0; ri < num_recv; ++ri) {
            uint32_t rci = core_to_core_id.at(receiver_vec[ri]) * config_page_size / sizeof(uint32_t);
            config_host_buffer[rci++] = 0;  // is_sender
            config_host_buffer[rci++] = num_recv;
            config_host_buffer[rci++] = data_base_addr;
            config_host_buffer[rci++] = ring_size;
            config_host_buffer[rci++] = data_base_addr;  // word[4]: reserved checkpoint (FW ignores for CrossNode)
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

void CrossNodeDFB::setup_buffers(BufferType buffer_type) {
    TT_FATAL(
        buffer_type == BufferType::L1 || buffer_type == BufferType::L1_SMALL,
        "CrossNodeDFB can only use L1 buffer types");
    validate_entry_geometry(device_, entry_size_, num_entries_);

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
    owned_dfb_buffer_ = distributed::AnyBuffer::create(data_shard_cfg);
    data_address_ = static_cast<uint32_t>(owned_dfb_buffer_.get_buffer()->address());
    allocate_config_and_write_pages(buffer_type);
}

void CrossNodeDFB::validate_data_buffer(Buffer& data_buffer) const {
    validate_entry_geometry(device_, entry_size_, num_entries_);

    const BufferType buffer_type = data_buffer.buffer_type();
    TT_FATAL(
        buffer_type == BufferType::L1 || buffer_type == BufferType::L1_SMALL,
        "CrossNodeDFB data buffer must be L1 or L1_SMALL, got {}",
        static_cast<int>(buffer_type));
    TT_FATAL(
        is_compatible_borrowed_device(device_, data_buffer.device()),
        "CrossNodeDFB data buffer device does not match Create device");
    TT_FATAL(
        data_buffer.buffer_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "CrossNodeDFB data buffer must be HEIGHT_SHARDED");
    TT_FATAL(data_buffer.has_shard_spec(), "CrossNodeDFB data buffer must have a shard spec");

    const uint32_t ring_size = entry_size_ * num_entries_;
    const uint32_t num_all_cores = all_cores_.num_cores();
    TT_FATAL(
        data_buffer.page_size() == ring_size,
        "CrossNodeDFB data buffer page_size {} must equal entry_size * num_entries ({})",
        data_buffer.page_size(),
        ring_size);

    const auto shard_spec = data_buffer.shard_spec();
    TT_FATAL(
        shard_spec.grid() == all_cores_,
        "CrossNodeDFB data buffer shard grid {} must match CrossNode all_cores {}",
        shard_spec.grid().str(),
        all_cores_.str());
    TT_FATAL(
        data_buffer.size() == static_cast<DeviceAddr>(ring_size) * num_all_cores,
        "CrossNodeDFB data buffer size {} must equal page_size * num_all_cores ({})",
        data_buffer.size(),
        static_cast<DeviceAddr>(ring_size) * num_all_cores);
}

void CrossNodeDFB::set_data_address(uint32_t data_address) {
    // Drop any CrossNode-owned ring before pointing at an external address.
    owned_dfb_buffer_ = {};
    TT_FATAL(data_address != 0, "CrossNodeDFB data address must be non-zero");
    data_address_ = data_address;
}

void CrossNodeDFB::setup_buffers_with_borrowed_data(Buffer& data_buffer) {
    validate_data_buffer(data_buffer);
    set_data_address(static_cast<uint32_t>(data_buffer.address()));
    // Config uses the same L1 bank class as the borrowed data buffer.
    allocate_config_and_write_pages(data_buffer.buffer_type());
}

void CrossNodeDFB::retarget_data_buffer(Buffer& data_buffer) {
    validate_data_buffer(data_buffer);
    TT_FATAL(config_buffer_.get_buffer() != nullptr, "CrossNodeDFB config buffer must already exist for retarget");
    set_data_address(static_cast<uint32_t>(data_buffer.address()));
    // Keep the existing config allocation; rewrite pages with the new data base address.
    write_config_pages();
}

// Accessors -------------------------------------------------------------------

const Buffer& CrossNodeDFB::config_buffer() const { return *config_buffer_.get_buffer(); }
uint32_t CrossNodeDFB::buffer_address() const { return data_address_; }
uint32_t CrossNodeDFB::config_address() const { return static_cast<uint32_t>(config_buffer().address()); }
uint32_t CrossNodeDFB::credit_reset_address() const { return config_address() + credit_reset_offset_; }
uint32_t CrossNodeDFB::credit_reset_size() const { return credit_reset_size_; }
uint32_t CrossNodeDFB::entry_size() const { return entry_size_; }
uint32_t CrossNodeDFB::num_entries() const { return num_entries_; }
const CoreRangeSet& CrossNodeDFB::sender_cores() const { return sender_cores_; }
const CoreRangeSet& CrossNodeDFB::receiver_cores() const { return receiver_cores_; }
const CoreRangeSet& CrossNodeDFB::all_cores() const { return all_cores_; }
const std::vector<std::pair<CoreCoord, CoreRangeSet>>& CrossNodeDFB::sender_receiver_core_mapping() const {
    return sender_receiver_mapping_;
}

// Free functions --------------------------------------------------------------

uint8_t CreateCrossNodeDFB(
    Program& program,
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type) {
    return program.impl().add_cross_node_dfb(
        CrossNodeDFB(device, sender_receiver_mapping, entry_size, num_entries, buffer_type));
}

uint8_t CreateCrossNodeDFB(
    Program& program,
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_mapping,
    uint32_t entry_size,
    uint32_t num_entries,
    Buffer& data_buffer) {
    return program.impl().add_cross_node_dfb(
        CrossNodeDFB(device, sender_receiver_mapping, entry_size, num_entries, data_buffer));
}

uint32_t CreateCrossNodeRelayDataflowBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& receiver_core_spec,
    const dfb::DataflowBufferConfig& config,
    uint8_t remote_dfb_id) {
    const CrossNodeDFB& gdfb = program.impl().get_cross_node_dfb(remote_dfb_id);

    CoreRangeSet receiver_cores;
    if (std::holds_alternative<CoreCoord>(receiver_core_spec)) {
        receiver_cores = CoreRangeSet({CoreRange(std::get<CoreCoord>(receiver_core_spec))});
    } else if (std::holds_alternative<CoreRange>(receiver_core_spec)) {
        receiver_cores = CoreRangeSet({std::get<CoreRange>(receiver_core_spec)});
    } else {
        receiver_cores = std::get<CoreRangeSet>(receiver_core_spec);
    }

    TT_FATAL(
        gdfb.receiver_cores().contains(receiver_cores),
        "CreateCrossNodeRelayDataflowBuffer: relay cores {} must be a subset of receiver cores {}",
        receiver_cores.str(),
        gdfb.receiver_cores().str());
    TT_FATAL(
        config.entry_size == gdfb.entry_size(),
        "CreateCrossNodeRelayDataflowBuffer: entry size {} must match CrossNodeDFB entry size {}",
        config.entry_size,
        gdfb.entry_size());
    TT_FATAL(
        config.num_entries == gdfb.num_entries(),
        "CreateCrossNodeRelayDataflowBuffer: depth {} must match CrossNodeDFB depth {}",
        config.num_entries,
        gdfb.num_entries());

    auto relay_config = config;
    relay_config.borrows_memory = true;
    const uint32_t relay_dfb_id = dfb::CreateDataflowBuffer(program, receiver_cores, relay_config);
    program.impl().register_cross_node_relay_dfb(receiver_cores, remote_dfb_id, relay_dfb_id);
    return relay_dfb_id;
}

void UpdateDynamicCrossNodeDFBAddress(Program& program, uint8_t remote_dfb_id, Buffer& buffer) {
    program.impl().update_dynamic_cross_node_dfb_address(remote_dfb_id, buffer);
}

}  // namespace tt::tt_metal::experimental
