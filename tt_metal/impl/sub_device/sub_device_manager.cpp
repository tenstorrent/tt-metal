// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <allocator.hpp>
#include <assert.hpp>
#include <device.hpp>
#include <host_api.hpp>
#include <sub_device.hpp>
#include <sub_device_types.hpp>
#include <tt_align.hpp>
#include <tt_stl/span.hpp>
#include <functional>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "allocator_types.hpp"
#include "buffer_types.hpp"
#include "core_coord.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "hal.hpp"
#include <tt_stl/strong_type.hpp>
#include "sub_device_manager.hpp"
#include "impl/context/metal_context.hpp"
#include "trace/trace.hpp"
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/types/xy_pair.h>
#include "vector_aligned.hpp"

namespace tt {
namespace tt_metal {
enum NOC : uint8_t;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

static_assert(
    DispatchSettings::DISPATCH_MESSAGE_ENTRIES <= std::numeric_limits<SubDeviceId::value_type>::max(),
    "Max number of sub-devices must be less than or equal to the max value of SubDeviceId::Id");

std::atomic<uint64_t> SubDeviceManager::next_sub_device_manager_id_ = 0;

SubDeviceManager::SubDeviceManager(
    tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size, IDevice* device) :
    id_(next_sub_device_manager_id_++),
    sub_devices_(sub_devices.begin(), sub_devices.end()),
    local_l1_size_(tt::align(local_l1_size, MetalContext::instance().hal().get_alignment(HalMemType::L1))),
    device_(device) {
    TT_ASSERT(device != nullptr, "Device must not be null");
    this->validate_sub_devices();
    this->populate_sub_device_ids();
    this->populate_num_cores();
    this->populate_sub_allocators();
    this->populate_noc_data();
}

SubDeviceManager::SubDeviceManager(
    IDevice* device, std::unique_ptr<Allocator>&& global_allocator, tt::stl::Span<const SubDevice> sub_devices) :
    id_(next_sub_device_manager_id_++),
    device_(device),
    sub_devices_(sub_devices.begin(), sub_devices.end()),
    local_l1_size_(0) {
    TT_ASSERT(device != nullptr, "Device must not be null");

    this->populate_sub_device_ids();
    // No need to validate sub-devices since this constructs a sub-device of the entire grid
    this->populate_num_cores();
    this->sub_device_allocators_.push_back(std::move(global_allocator));
    this->populate_noc_data();
}

SubDeviceManager::~SubDeviceManager() {
    for (const auto& allocator : sub_device_allocators_) {
        if (allocator) {
            // Clear the bank managers, this makes subsequent buffer deallocations fast
            allocator->clear();
            // Deallocate all buffers
            // This is done to set buffer object status to Deallocated
            const auto& allocated_buffers = allocator->get_allocated_buffers();
            for (auto buf = allocated_buffers.begin(); buf != allocated_buffers.end();) {
                tt::tt_metal::DeallocateBuffer(*(*(buf++)));
            }
        }
    }
}

SubDeviceManagerId SubDeviceManager::id() const { return id_; }

uint8_t SubDeviceManager::num_sub_devices() const { return sub_devices_.size(); }

const std::vector<SubDeviceId>& SubDeviceManager::get_sub_device_ids() const { return sub_device_ids_; }

const SubDevice& SubDeviceManager::sub_device(SubDeviceId sub_device_id) const {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    return sub_devices_[sub_device_index];
}

const vector_aligned<uint32_t>& SubDeviceManager::noc_mcast_unicast_data() const { return noc_mcast_unicast_data_; }

uint8_t SubDeviceManager::num_noc_mcast_txns(SubDeviceId sub_device_id) const {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    return num_noc_mcast_txns_[sub_device_index];
}

uint8_t SubDeviceManager::num_noc_unicast_txns(SubDeviceId sub_device_id) const {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    return num_noc_unicast_txns_[sub_device_index];
}

uint8_t SubDeviceManager::noc_mcast_data_start_index(SubDeviceId sub_device_id) const {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    return noc_mcast_data_start_index_[sub_device_index];
}

uint8_t SubDeviceManager::noc_unicast_data_start_index(SubDeviceId sub_device_id) const {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    return noc_unicast_data_start_index_[sub_device_index];
}

const std::unique_ptr<Allocator>& SubDeviceManager::allocator(SubDeviceId sub_device_id) const {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    TT_FATAL(sub_device_allocators_[sub_device_index], "SubDevice allocator not initialized");
    return sub_device_allocators_[sub_device_index];
}

std::unique_ptr<Allocator>& SubDeviceManager::sub_device_allocator(SubDeviceId sub_device_id) {
    auto sub_device_index = this->get_sub_device_index(sub_device_id);
    return sub_device_allocators_[sub_device_index];
}

std::shared_ptr<TraceBuffer>& SubDeviceManager::create_trace(uint32_t tid) {
    auto [trace, emplaced] = trace_buffer_pool_.emplace(tid, Trace::create_empty_trace_buffer());
    TT_ASSERT(emplaced, "Trace buffer with tid {} already exists", tid);
    return trace->second;
}

void SubDeviceManager::release_trace(uint32_t tid) { trace_buffer_pool_.erase(tid); }

std::shared_ptr<TraceBuffer> SubDeviceManager::get_trace(uint32_t tid) {
    auto trace = trace_buffer_pool_.find(tid);
    if (trace != trace_buffer_pool_.end()) {
        return trace->second;
    }
    return nullptr;
}

bool SubDeviceManager::has_allocations() const {
    for (const auto& allocator : sub_device_allocators_) {
        if (allocator && allocator->get_allocated_buffers().size() > 0) {
            return true;
        }
    }
    return false;
}

DeviceAddr SubDeviceManager::local_l1_size() const { return local_l1_size_; }

const std::vector<SubDeviceId>& SubDeviceManager::get_sub_device_stall_group() const { return sub_device_stall_group_; }

void SubDeviceManager::set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    TT_FATAL(!sub_device_ids.empty(), "sub_device_ids to stall must not be empty");
    for (const auto& sub_device_id : sub_device_ids) {
        TT_FATAL(
            *sub_device_id < sub_devices_.size(),
            "SubDevice index {} out of bounds {}",
            *sub_device_id,
            sub_devices_.size());
    }
    sub_device_stall_group_ = std::vector<SubDeviceId>(sub_device_ids.begin(), sub_device_ids.end());
}

void SubDeviceManager::reset_sub_device_stall_group() { this->set_sub_device_stall_group(sub_device_ids_); }

uint8_t SubDeviceManager::get_sub_device_index(SubDeviceId sub_device_id) const {
    auto sub_device_index = *sub_device_id;
    TT_FATAL(
        sub_device_index < sub_devices_.size(),
        "SubDevice index {} out of bounds {}",
        sub_device_index,
        sub_devices_.size());
    return sub_device_index;
}

void SubDeviceManager::validate_sub_devices() const {
    TT_FATAL(
        sub_devices_.size() <= DispatchSettings::DISPATCH_MESSAGE_ENTRIES,
        "Number of sub-devices specified {} is larger than the max number of sub-devices {}",
        sub_devices_.size(),
        DispatchSettings::DISPATCH_MESSAGE_ENTRIES);
    // Validate sub device cores fit inside the device grid
    const auto& compute_grid_size = device_->compute_with_storage_grid_size();
    CoreRange device_worker_cores = CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1});

    for (uint8_t sub_device_id = 0; sub_device_id < this->num_sub_devices(); ++sub_device_id) {
        const auto& sub_device = this->sub_device(SubDeviceId(sub_device_id));
        const auto& worker_cores = sub_device.cores(HalProgrammableCoreType::TENSIX);
        TT_FATAL(
            device_worker_cores.contains(worker_cores),
            "Tensix cores {} specified in sub device must be within device grid {}",
            worker_cores,
            device_worker_cores);

        if (sub_device.has_core_type(HalProgrammableCoreType::ACTIVE_ETH)) {
            const auto& eth_cores = sub_device.cores(HalProgrammableCoreType::ACTIVE_ETH);
            uint32_t num_eth_cores = 0;
            const auto& device_eth_cores =
                tt::tt_metal::MetalContext::instance().get_cluster().get_active_ethernet_cores(device_->id());
            for (const auto& dev_eth_core : device_eth_cores) {
                if (eth_cores.contains(dev_eth_core)) {
                    num_eth_cores++;
                }
            }
            TT_FATAL(
                num_eth_cores == eth_cores.num_cores(),
                "Ethernet cores {} specified in sub device must be within device grid",
                eth_cores);
        }
    }
    if (sub_devices_.size() < 2) {
        return;
    }
    // Validate no overlap of sub devices
    for (uint32_t i = 0; i < sub_devices_.size(); ++i) {
        for (uint32_t j = i + 1; j < sub_devices_.size(); ++j) {
            for (uint32_t k = 0; k < NumHalProgrammableCoreTypes; ++k) {
                TT_FATAL(
                    !(sub_devices_[i].cores()[k].intersects(sub_devices_[j].cores()[k])),
                    "SubDevices specified for SubDeviceManager intersect");
            }
        }
    }
}

void SubDeviceManager::populate_sub_device_ids() {
    sub_device_ids_.reserve(this->num_sub_devices());
    for (uint8_t i = 0; i < this->num_sub_devices(); ++i) {
        sub_device_ids_.push_back(SubDeviceId{i});
    }
    this->reset_sub_device_stall_group();
}

void SubDeviceManager::populate_num_cores() {
    for (const auto& sub_device : sub_devices_) {
        for (uint32_t i = 0; i < NumHalProgrammableCoreTypes; ++i) {
            num_cores_[i] += sub_device.num_cores(static_cast<HalProgrammableCoreType>(i));
        }
    }
}

void SubDeviceManager::populate_sub_allocators() {
    sub_device_allocators_.resize(this->num_sub_devices());
    if (local_l1_size_ == 0) {
        return;
    }
    const auto& global_allocator_config = device_->allocator()->get_config();
    // Construct allocator config from soc_desc
    // Take max alignment to satisfy NoC rd/wr constraints
    // Tensix/Eth -> PCIe/DRAM src and dst addrs must be L1_ALIGNMENT aligned
    // PCIe/DRAM -> Tensix/Eth src and dst addrs must be DRAM_ALIGNMENT aligned
    // Tensix/Eth <-> Tensix/Eth src and dst addrs must be L1_ALIGNMENT aligned
    for (uint32_t i = 0; i < this->num_sub_devices(); ++i) {
        const auto& compute_cores = sub_devices_[i].cores(HalProgrammableCoreType::TENSIX);
        if (compute_cores.empty()) {
            continue;
        }
        auto compute_cores_vec = corerange_to_cores(compute_cores, std::nullopt, true);
        // Make sub-device cores have the same bank id as global allocator
        std::vector<uint32_t> l1_bank_remap;
        l1_bank_remap.reserve(compute_cores_vec.size());
        for (const auto& core : compute_cores_vec) {
            // These are compute cores, so they should have a single bank
            l1_bank_remap.push_back(device_->allocator()->get_bank_ids_from_logical_core(BufferType::L1, core)[0]);
        }
        AllocatorConfig config(
            {.num_dram_channels = global_allocator_config.num_dram_channels,
             .dram_bank_size = 0,
             .dram_bank_offsets = global_allocator_config.dram_bank_offsets,
             .dram_unreserved_base = global_allocator_config.dram_unreserved_base,
             .dram_alignment = global_allocator_config.dram_alignment,
             .l1_unreserved_base = global_allocator_config.l1_unreserved_base,
             .worker_grid = compute_cores,
             .worker_l1_size = global_allocator_config.l1_unreserved_base + local_l1_size_,
             .storage_core_bank_size = std::nullopt,
             .l1_small_size = 0,
             .trace_region_size = 0,
             .core_type_from_noc_coord_table = {},  // Populated later
             .worker_log_to_virtual_routing_x = global_allocator_config.worker_log_to_virtual_routing_x,
             .worker_log_to_virtual_routing_y = global_allocator_config.worker_log_to_virtual_routing_y,
             .l1_bank_remap = std::move(l1_bank_remap),
             .compute_grid = compute_cores,
             .l1_alignment = global_allocator_config.l1_alignment,
             .disable_interleaved = true});
        TT_FATAL(
            config.l1_small_size < (config.storage_core_bank_size.has_value()
                                        ? config.storage_core_bank_size.value()
                                        : config.worker_l1_size - config.l1_unreserved_base),
            "Reserved size must be less than bank size");
        TT_FATAL(
            config.l1_small_size % config.l1_alignment == 0,
            "Reserved size must be aligned to allocator L1 alignment {}",
            config.l1_alignment);

        // sub_devices only have compute cores for allocation
        for (const CoreCoord& core : corerange_to_cores(compute_cores)) {
            const auto noc_coord = device_->worker_core_from_logical_core(core);
            config.core_type_from_noc_coord_table.insert({noc_coord, AllocCoreType::ComputeAndStore});
        }

        // L1_BANKING scheme creates 1 bank per DRAM core and splits up L1 such that there are power 2 num L1 banks
        // This is the only allocator scheme supported because kernel APIs assume num L1 banks are power of 2
        TT_ASSERT(device_->allocator_scheme_ == MemoryAllocator::L1_BANKING);
        sub_device_allocators_[i] = std::make_unique<L1BankingAllocator>(config);
    }
}

void SubDeviceManager::populate_noc_data() {
    uint32_t num_sub_devices = this->num_sub_devices();
    num_noc_mcast_txns_.resize(num_sub_devices);
    num_noc_unicast_txns_.resize(num_sub_devices);
    noc_mcast_data_start_index_.resize(num_sub_devices);
    noc_unicast_data_start_index_.resize(num_sub_devices);

    NOC noc_index = MetalContext::instance().get_dispatch_query_manager().go_signal_noc();
    uint32_t idx = 0;
    for (uint32_t i = 0; i < num_sub_devices; ++i) {
        const auto& tensix_cores = sub_devices_[i].cores(HalProgrammableCoreType::TENSIX).merge_ranges();
        const auto& eth_cores = sub_devices_[i].cores(HalProgrammableCoreType::ACTIVE_ETH);

        noc_mcast_data_start_index_[i] = idx;
        num_noc_mcast_txns_[i] = tensix_cores.size();
        noc_mcast_unicast_data_.resize(idx + num_noc_mcast_txns_[i] * 2);
        for (const auto& core_range : tensix_cores.ranges()) {
            auto virtual_start = device_->virtual_core_from_logical_core(core_range.start_coord, CoreType::WORKER);
            auto virtual_end = device_->virtual_core_from_logical_core(core_range.end_coord, CoreType::WORKER);
            auto virtual_core_range = CoreRange(virtual_start, virtual_end);
            noc_mcast_unicast_data_[idx++] = device_->get_noc_multicast_encoding(noc_index, virtual_core_range);
            noc_mcast_unicast_data_[idx++] = core_range.size();
        }
        noc_unicast_data_start_index_[i] = idx;

        // TODO: Precompute number of eth cores and resize once
        for (const auto& core_range : eth_cores.ranges()) {
            noc_mcast_unicast_data_.resize(idx + core_range.size());
            for (const auto& core : core_range) {
                auto virtual_core = device_->virtual_core_from_logical_core(core, CoreType::ETH);
                noc_mcast_unicast_data_[idx++] = device_->get_noc_unicast_encoding(noc_index, virtual_core);
            }
        }
        num_noc_unicast_txns_[i] = idx - noc_unicast_data_start_index_[i];

        TT_FATAL(
            idx <= DispatchSettings::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES,
            "NOC data entries {} exceeds maximum supported size {}",
            idx,
            DispatchSettings::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES);
    }
}

}  // namespace tt::tt_metal
