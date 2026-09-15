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
// fanout in the topology so all pages share one size.
// Fresh zero credits on each launch (via a fresh ringbuffer page image) rewind every
// receiver's derived write position to the start of the ring.
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

void CrossNodeDFB::allocate_config_buffer(BufferType config_buffer_type) {
    TT_FATAL(
        config_buffer_type == BufferType::L1 || config_buffer_type == BufferType::L1_SMALL,
        "CrossNodeDFB can only use L1 buffer types");

    const auto context_id = extract_context_id(device_);
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    const uint32_t num_all_cores = all_cores_.num_cores();
    config_page_size_ = compute_config_page_layout(max_num_receivers_per_sender_, l1_alignment).page_size;

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

void CrossNodeDFB::rebuild_config_pages() {
    TT_FATAL(config_buffer_.get_buffer() != nullptr, "CrossNodeDFB config buffer must exist before building pages");
    TT_FATAL(data_address_ != 0, "CrossNodeDFB data address must be set before building config pages");

    const auto context_id = extract_context_id(device_);
    const auto& hal = MetalContext::instance(context_id).hal();
    const uint32_t l1_alignment = hal.get_alignment(HalMemType::L1);
    const uint32_t ring_size = entry_size_ * num_entries_;

    // Host-only config page images. Materialized into the dedicated sharded config Buffer
    // on program launch. Words 5–7 are page-relative offsets; firmware resolves them as
    // config_page_ptr + offset. Every shard has the same numeric L1 base, so page-relative
    // remote credit offsets target the peer's corresponding config page via NOC.
    const auto layout = compute_config_page_layout(max_num_receivers_per_sender_, l1_alignment);
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
        sender_page[si++] = 1;               // is_sender
        sender_page[si++] = num_recv;        // num_receivers
        sender_page[si++] = data_base_addr;  // fifo_start_addr
        sender_page[si++] = ring_size;       // fifo_size
        sender_page[si++] = data_base_addr;  // word[4]: reserved checkpoint
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
            receiver_page[rci++] = 0;  // is_sender
            receiver_page[rci++] = num_recv;
            receiver_page[rci++] = data_base_addr;
            receiver_page[rci++] = ring_size;
            receiver_page[rci++] = data_base_addr;
            receiver_page[rci++] = layout.noc_xy_offset;
            receiver_page[rci++] = layout.counters_offset + 2 * ri * l1_alignment;
            receiver_page[rci++] = layout.counters_offset + 2 * ri * l1_alignment + l1_alignment;
            receiver_page[rci++] = static_cast<uint32_t>(sender_phys.x);
            receiver_page[rci++] = static_cast<uint32_t>(sender_phys.y);
            config_pages_[receiver_vec[ri]] = std::move(receiver_page);
        }
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
    allocate_config_buffer(buffer_type);
    rebuild_config_pages();
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
    allocate_config_buffer(data_buffer.buffer_type());
    rebuild_config_pages();
}

void CrossNodeDFB::retarget_data_buffer(Buffer& data_buffer) {
    validate_data_buffer(data_buffer);
    TT_FATAL(config_buffer_.get_buffer() != nullptr, "CrossNodeDFB config buffer must already exist for retarget");
    set_data_address(static_cast<uint32_t>(data_buffer.address()));
    // Host-only rebuild; device L1 is unchanged until the next program launch.
    rebuild_config_pages();
}

// Accessors -------------------------------------------------------------------

const Buffer& CrossNodeDFB::config_buffer() const { return *config_buffer_.get_buffer(); }
uint32_t CrossNodeDFB::buffer_address() const { return data_address_; }
uint32_t CrossNodeDFB::config_address() const { return static_cast<uint32_t>(config_buffer().address()); }
const std::vector<uint32_t>& CrossNodeDFB::config_page(const CoreCoord& core) const {
    auto it = config_pages_.find(core);
    TT_FATAL(it != config_pages_.end(), "CrossNodeDFB has no host config page for core {}", core.str());
    return it->second;
}
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
