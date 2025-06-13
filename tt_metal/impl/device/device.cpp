// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device_impl.hpp"

#include <core_descriptor.hpp>
#include "dev_msgs.h"
#include <device_pool.hpp>
#include <host_api.hpp>
#include <magic_enum/magic_enum.hpp>
#include <persistent_kernel_cache.hpp>
#include <sub_device.hpp>
#include <sub_device_types.hpp>
#include <tt-metalium/program_cache.hpp>
#include <tt-metalium/hal.hpp>
#include <tt_align.hpp>
#include <tt_metal.hpp>
#include <tt_stl/span.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "allocator.hpp"
#include "allocator_types.hpp"
#include "assert.hpp"
#include "buffer_types.hpp"
#include "command_queue.hpp"
#include "dispatch/command_queue_common.hpp"
#include "common/core_assignment.hpp"
#include "program/program_impl.hpp"
#include "core_coord.hpp"
#include "device.hpp"
#include "impl/context/metal_context.hpp"
#include "trace/trace.hpp"
#include "dispatch_core_common.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "dprint_server.hpp"
#include "hal_types.hpp"
#include "jit_build/build.hpp"
#include "dispatch/launch_message_ring_buffer_state.hpp"
#include "lightmetal/lightmetal_capture.hpp"
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include "metal_soc_descriptor.h"
#include "multi_producer_single_consumer_queue.hpp"
#include "profiler_types.hpp"
#include "tt-metalium/program.hpp"
#include <tt_stl/strong_type.hpp>
#include "dispatch/system_memory_manager.hpp"
#include "trace/trace_buffer.hpp"
#include "tracy/Tracy.hpp"
#include "tt_memory.h"
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/dispatch/hardware_command_queue.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tt_metal/impl/sub_device/sub_device_manager.hpp"
#include "sub_device/sub_device_manager_tracker.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "tt_metal/tools/profiler/tt_metal_tracy.hpp"
#include <umd/device/coordinate_manager.h>
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/tt_silicon_driver_common.hpp>
#include <umd/device/tt_xy_pair.h>
#include <umd/device/types/xy_pair.h>

namespace tt {
enum class ARCH;

namespace tt_metal {

uint64_t IDevice::get_dev_addr(CoreCoord virtual_core, HalL1MemAddrType addr_type) const {
    return MetalContext::instance().hal().get_dev_addr(this->get_programmable_core_type(virtual_core), addr_type);
}

uint64_t IDevice::get_dev_size(CoreCoord virtual_core, HalL1MemAddrType addr_type) const {
    return MetalContext::instance().hal().get_dev_size(this->get_programmable_core_type(virtual_core), addr_type);
}

void IDevice::set_program_cache_misses_allowed(bool allowed) {
    this->get_program_cache().set_cache_misses_allowed(allowed);
}

Device::Device(Device&& other) = default;
Device& Device::operator=(Device&& other) = default;

Device::Device(
    chip_id_t device_id,
    const uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    bool minimal,
    uint32_t worker_thread_core,
    uint32_t completion_queue_reader_core,
    size_t worker_l1_size) :
    id_(device_id),
    worker_thread_core_(worker_thread_core),
    completion_queue_reader_core_(completion_queue_reader_core) {
    ZoneScoped;
    this->initialize(num_hw_cqs, l1_small_size, trace_region_size, worker_l1_size, l1_bank_remap, minimal);
}

std::unordered_set<CoreCoord> Device::get_active_ethernet_cores(bool skip_reserved_tunnel_cores) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_active_ethernet_cores(
        this->id_, skip_reserved_tunnel_cores);
}

bool Device::is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores) const {
    auto active_ethernet_cores = this->get_active_ethernet_cores(skip_reserved_tunnel_cores);
    return active_ethernet_cores.find(logical_core) != active_ethernet_cores.end();
}

std::unordered_set<CoreCoord> Device::get_inactive_ethernet_cores() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_inactive_ethernet_cores(this->id_);
}

bool Device::is_inactive_ethernet_core(CoreCoord logical_core) const {
    auto inactive_ethernet_cores =
        tt::tt_metal::MetalContext::instance().get_cluster().get_inactive_ethernet_cores(this->id_);
    return inactive_ethernet_cores.find(logical_core) != inactive_ethernet_cores.end();
}

uint32_t Device::num_virtual_eth_cores(SubDeviceId sub_device_id) {
    return this->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
}

std::tuple<chip_id_t, CoreCoord> Device::get_connected_ethernet_core(CoreCoord eth_core) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_connected_ethernet_core(
        std::make_tuple(this->id_, eth_core));
}

std::vector<CoreCoord> Device::get_ethernet_sockets(chip_id_t connected_chip_id) const {
    if (tt::tt_metal::MetalContext::instance().get_fabric_config() !=
        tt::tt_metal::FabricConfig::DISABLED) {
        return tt::tt_metal::MetalContext::instance().get_cluster().get_fabric_ethernet_routers_between_src_and_dest(
            this->id_, connected_chip_id);
    } else {
        return tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_sockets(this->id_, connected_chip_id);
    }
}

bool Device::is_mmio_capable() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(this->id_) == this->id_;
}

CoreRangeSet Device::worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->sub_device(sub_device_id).cores(core_type);
}

uint32_t Device::num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->sub_device(sub_device_id).num_cores(core_type);
}

void Device::initialize_cluster() {
    ZoneScoped;
    if (tt_metal::MetalContext::instance().rtoptions().get_clear_l1()) {
        this->clear_l1_state();
    }
    if (tt_metal::MetalContext::instance().rtoptions().get_clear_dram()) {
        this->clear_dram_state();
    }
    int ai_clk = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(this->id_);
    log_info(tt::LogMetal, "AI CLK for device {} is:   {} MHz", this->id_, ai_clk);
}

void Device::initialize_default_sub_device_state(
    size_t l1_small_size,
    size_t trace_region_size,
    size_t worker_l1_unreserved_start,
    tt::stl::Span<const std::uint32_t> l1_bank_remap) {
    // Create the default sub-device manager representing the entire chip
    const auto& compute_grid_size = this->compute_with_storage_grid_size();
    const auto& active_eth_cores = this->get_active_ethernet_cores(true);
    std::vector<CoreRange> active_eth_core_ranges;
    active_eth_core_ranges.reserve(active_eth_cores.size());
    for (const auto& core : active_eth_cores) {
        active_eth_core_ranges.emplace_back(core, core);
    }

    auto sub_devices = {SubDevice(std::array{
        CoreRangeSet(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1})),
        CoreRangeSet(std::move(active_eth_core_ranges))})};

    sub_device_manager_tracker_ = std::make_unique<SubDeviceManagerTracker>(
        this,
        this->initialize_allocator(l1_small_size, trace_region_size, worker_l1_unreserved_start, l1_bank_remap),
        sub_devices);
}

std::unique_ptr<Allocator> Device::initialize_allocator(
    size_t l1_small_size,
    size_t trace_region_size,
    size_t worker_l1_unreserved_start,
    tt::stl::Span<const std::uint32_t> l1_bank_remap) {
    ZoneScoped;
    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id_);
    const auto& dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    CoreType dispatch_core_type = dispatch_core_config.get_core_type();
    // Construct allocator config from soc_desc
    // Take max alignment to satisfy NoC rd/wr constraints
    // Tensix/Eth -> PCIe/DRAM src and dst addrs must be L1_ALIGNMENT aligned
    // PCIe/DRAM -> Tensix/Eth src and dst addrs must be DRAM_ALIGNMENT aligned
    // Tensix/Eth <-> Tensix/Eth src and dst addrs must be L1_ALIGNMENT aligned
    const auto &logical_size = this->logical_grid_size();
    const auto& compute_size = this->compute_with_storage_grid_size();
    const auto& hal = MetalContext::instance().hal();
    AllocatorConfig config(
        {.num_dram_channels = static_cast<size_t>(soc_desc.get_num_dram_views()),
         .dram_bank_size = soc_desc.dram_view_size,
         .dram_bank_offsets = {},
         .dram_unreserved_base = hal.get_dev_addr(HalDramMemAddrType::UNRESERVED),
         .dram_alignment = hal.get_alignment(HalMemType::DRAM),
         .l1_unreserved_base = align(worker_l1_unreserved_start, hal.get_alignment(HalMemType::DRAM)),
         .worker_grid = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(logical_size.x - 1, logical_size.y - 1))),
         .worker_l1_size = static_cast<size_t>(soc_desc.worker_l1_size),
         .storage_core_bank_size = get_storage_core_bank_size(id_, num_hw_cqs_, dispatch_core_config),
         .l1_small_size = align(l1_small_size, hal.get_alignment(HalMemType::DRAM)),
         .trace_region_size = align(trace_region_size, hal.get_alignment(HalMemType::DRAM)),
         .core_type_from_noc_coord_table = {},  // Populated later
         .worker_log_to_virtual_routing_x =
             tt::tt_metal::MetalContext::instance().get_cluster().get_worker_logical_to_virtual_x(this->id()),
         .worker_log_to_virtual_routing_y =
             tt::tt_metal::MetalContext::instance().get_cluster().get_worker_logical_to_virtual_y(this->id()),
         .l1_bank_remap = {l1_bank_remap.begin(), l1_bank_remap.end()},
         .compute_grid = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(compute_size.x - 1, compute_size.y - 1))),
         .l1_alignment = hal.get_alignment(HalMemType::L1),
         .disable_interleaved = false});
    TT_FATAL(config.l1_small_size < (config.storage_core_bank_size.has_value() ? config.storage_core_bank_size.value() : config.worker_l1_size - config.l1_unreserved_base),
            "Reserved size must be less than bank size");
    TT_FATAL(
        config.l1_small_size % config.l1_alignment == 0,
        "Reserved size must be aligned to L1 allocator alignment {}",
        config.l1_alignment);
    // Initialize dram_offsets from soc_descriptor
    for (auto channel = 0; channel < soc_desc.get_num_dram_views(); channel++) {
        config.dram_bank_offsets.push_back(soc_desc.get_address_offset(channel));
    }
    // Initialize core_type_from_noc_coord_table table
    for (const CoreCoord& core : soc_desc.get_all_cores(CoordSystem::PHYSICAL)) {
        config.core_type_from_noc_coord_table.insert(
            {this->virtual_core_from_physical_core({core.x, core.y}), AllocCoreType::Invalid});
    }
    for (const CoreCoord& core : soc_desc.get_all_harvested_cores(CoordSystem::PHYSICAL)) {
        config.core_type_from_noc_coord_table.insert(
            {this->virtual_core_from_physical_core({core.x, core.y}), AllocCoreType::Invalid});
    }

    for (const CoreCoord& core : tt::get_logical_compute_cores(id_, num_hw_cqs_, dispatch_core_config)) {
        this->compute_cores_.insert(core);
        const auto noc_coord = this->worker_core_from_logical_core(core);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::ComputeAndStore;
    }
    for (const CoreCoord& core : tt::get_logical_storage_cores(id_, num_hw_cqs_, dispatch_core_config)) {
        this->storage_only_cores_.insert(core);
        const auto noc_coord = this->worker_core_from_logical_core(core);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::StorageOnly;
    }
    for (const CoreCoord &core : tt::get_logical_dispatch_cores(id_, num_hw_cqs_, dispatch_core_config)) {
        const auto noc_coord = this->virtual_core_from_logical_core(core, dispatch_core_type);
        config.core_type_from_noc_coord_table[noc_coord] = AllocCoreType::Dispatch;
    }
    for (const tt::umd::CoreCoord& core : soc_desc.get_cores(CoreType::ETH, CoordSystem::LOGICAL)) {
        this->ethernet_cores_.insert({core.x, core.y});
    }

    // L1_BANKING scheme creates 1 bank per DRAM core and splits up L1 such that there are power 2 num L1 banks
    // This is the only allocator scheme supported because kernel APIs assume num L1 banks are power of 2
    TT_ASSERT(this->allocator_scheme_ == MemoryAllocator::L1_BANKING);
    return std::make_unique<L1BankingAllocator>(config);
}

void Device::initialize_device_bank_to_noc_tables(const HalProgrammableCoreType &core_type, CoreCoord virtual_core)
{
    const uint32_t dram_to_noc_sz_in_bytes = dram_bank_to_noc_xy_.size() * sizeof(uint16_t);
    const uint32_t l1_to_noc_sz_in_bytes = l1_bank_to_noc_xy_.size() * sizeof(uint16_t);
    const uint32_t dram_offset_sz_in_bytes = dram_bank_offset_map_.size() * sizeof(int32_t);
    const uint32_t l1_offset_sz_in_bytes = l1_bank_offset_map_.size() * sizeof(int32_t);

    const uint64_t mem_bank_to_noc_addr =
        MetalContext::instance().hal().get_dev_addr(core_type, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);
    const uint32_t mem_bank_to_noc_size =
        MetalContext::instance().hal().get_dev_size(core_type, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);

    TT_ASSERT((dram_to_noc_sz_in_bytes + l1_to_noc_sz_in_bytes + dram_offset_sz_in_bytes + l1_offset_sz_in_bytes) <= mem_bank_to_noc_size,
        "Size of bank_to_noc table is greater than available space");

    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        &dram_bank_to_noc_xy_[0], dram_to_noc_sz_in_bytes, tt_cxy_pair(this->id(), virtual_core), mem_bank_to_noc_addr);
    uint64_t l1_noc_addr = mem_bank_to_noc_addr + dram_to_noc_sz_in_bytes;
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        &l1_bank_to_noc_xy_[0], l1_to_noc_sz_in_bytes, tt_cxy_pair(this->id(), virtual_core), l1_noc_addr);

    uint64_t dram_offset_addr = l1_noc_addr + l1_to_noc_sz_in_bytes;
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        &dram_bank_offset_map_[0], dram_offset_sz_in_bytes, tt_cxy_pair(this->id(), virtual_core), dram_offset_addr);
    uint64_t l1_offset_addr = dram_offset_addr + dram_offset_sz_in_bytes;
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        &l1_bank_offset_map_[0], l1_offset_sz_in_bytes, tt_cxy_pair(this->id(), virtual_core), l1_offset_addr);
}

void Device::initialize_firmware(const HalProgrammableCoreType &core_type, CoreCoord virtual_core, launch_msg_t *launch_msg, go_msg_t* go_msg) {
    ZoneScoped;

    this->initialize_device_bank_to_noc_tables(core_type, virtual_core);
    const auto& hal = MetalContext::instance().hal();
    uint32_t core_type_idx = hal.get_programmable_core_type_index(core_type);
    uint32_t processor_class_count = hal.get_processor_classes_count(core_type);
    auto jit_build_config =
        hal.get_jit_build_config(core_type_idx, 0, 0);  // Only the first risc needs to be programmed
    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();

    switch (core_type) {
        case HalProgrammableCoreType::TENSIX: {
            for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                auto [build_idx, num_build_states] =
                    BuildEnvManager::get_instance().get_build_index_and_state_count(core_type_idx, processor_class);
                for (uint32_t riscv_id = 0; riscv_id < num_build_states; riscv_id++) {
                    auto fw_path = BuildEnvManager::get_instance()
                                       .get_firmware_build_state(id_, core_type_idx, processor_class, riscv_id)
                                       .get_target_out_path("");
                    const ll_api::memory& binary_mem = llrt::get_risc_binary(fw_path);
                    uint32_t fw_size = binary_mem.get_text_size();
                    if (riscv_id + build_idx == 1) {  // TODO: clean up how brisc/ncrisc are handled
                        // In this context, ncrisc_kernel_size16 is the size of the fw
                        launch_msg->kernel_config.ncrisc_kernel_size16 = (fw_size + 15) >> 4;
                    }
                    log_debug(tt::LogMetal, "RISC {} fw binary size: {} in bytes", riscv_id, fw_size);

                    if (not rtoptions.get_skip_loading_fw()) {
                        llrt::test_load_write_read_risc_binary(
                            binary_mem, this->id(), virtual_core, core_type_idx, processor_class, riscv_id);
                    }
                }
            }

            if (this->using_slow_dispatch()) {
                // Host always writes launch messages
                launch_msg->kernel_config.mode = DISPATCH_MODE_HOST;
            } else {
                std::vector<CoreCoord> physical_dispatch_cores = {};
                if (MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type() == CoreType::WORKER) {
                    physical_dispatch_cores = this->worker_cores_from_logical_cores(
                        MetalContext::instance().get_dispatch_core_manager().get_all_logical_dispatch_cores(
                            this->id()));
                }
                if (std::find(physical_dispatch_cores.begin(), physical_dispatch_cores.end(), virtual_core) != physical_dispatch_cores.end()) {
                    // Dispatch cores - Host writes launch messages
                    launch_msg->kernel_config.mode = DISPATCH_MODE_HOST;
                } else {
                    // Worker cores - Dispatcher will write launch messages
                    launch_msg->kernel_config.mode = DISPATCH_MODE_DEV;
                }
            }

            break;
        }
        case HalProgrammableCoreType::ACTIVE_ETH:
        case HalProgrammableCoreType::IDLE_ETH: {
            bool is_idle_eth = core_type == HalProgrammableCoreType::IDLE_ETH;
            TensixSoftResetOptions reset_val = TENSIX_ASSERT_SOFT_RESET;
            if (not is_idle_eth) {
                reset_val =
                    reset_val & static_cast<TensixSoftResetOptions>(
                                    ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::BRISC));
            }
            if (is_idle_eth or !hal.get_eth_fw_is_cooperative()) {
                tt::tt_metal::MetalContext::instance().get_cluster().assert_risc_reset_at_core(
                    tt_cxy_pair(this->id(), virtual_core), reset_val);
            }
            if (not rtoptions.get_skip_loading_fw()) {
                for (uint32_t processor_class = 0; processor_class < processor_class_count; processor_class++) {
                    auto num_build_states = hal.get_processor_types_count(core_type_idx, processor_class);
                    for (uint32_t eriscv_id = 0; eriscv_id < num_build_states; eriscv_id++) {
                        auto fw_path = BuildEnvManager::get_instance()
                                           .get_firmware_build_state(id_, core_type_idx, processor_class, eriscv_id)
                                           .get_target_out_path("");
                        const ll_api::memory& binary_mem = llrt::get_risc_binary(fw_path);
                        uint32_t fw_size = binary_mem.get_text_size();
                        log_debug(tt::LogMetal, "ERISC fw binary size: {} in bytes", fw_size);
                        llrt::test_load_write_read_risc_binary(
                            binary_mem, this->id(), virtual_core, core_type_idx, processor_class, eriscv_id);
                    }
                }
            }
            // Ethernet worker core. Launch messages will be sent by FD infra if it's enabled
            // Idle ethernet core. Used by FD infra. Host will write launch messages during init.
            launch_msg->kernel_config.mode = (this->using_slow_dispatch() or is_idle_eth) ? DISPATCH_MODE_HOST :  DISPATCH_MODE_DEV;
            break;
        }
        default:
            TT_THROW("Unsupported programable core type {} to initialize build states", magic_enum::enum_name(core_type));
    }

    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        &jit_build_config.fw_launch_addr_value,
        sizeof(uint32_t),
        tt_cxy_pair(this->id_, virtual_core),
        jit_build_config.fw_launch_addr);

    // Initialize each entry in the launch_msg ring buffer with the correct dispatch mode - Cores that don't get a valid
    // launch_message during program execution need to at least have the correct dispatch mode.
    // When using Fast Dispatch on Tensix:
        // dispatch cores (Tensix) configured with DISPATCH_MODE_HOST
        // worker cores (Tensix and active eth) configured with DISPATCH_MODE_DEV
        // Idle Eth cores configured with DISPATCH_MODE_HOST but not used
    // When using Fast Dispatch on Idle Eth:
        // dispatch cores (Idle Eth) configured with DISPATCH_MODE_HOST
        // worker cores (Tensix and active eth) configured with DISPATCH_MODE_DEV
    // When using Slow Dispatch, all cores initialized with DISPATCH_MODE_HOST
    std::vector<launch_msg_t> init_launch_msg_data(launch_msg_buffer_num_entries, *launch_msg);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        init_launch_msg_data.data(),
        launch_msg_buffer_num_entries * sizeof(launch_msg_t),
        tt_cxy_pair(this->id(), virtual_core),
        this->get_dev_addr(virtual_core, HalL1MemAddrType::LAUNCH));
    uint32_t go_addr = this->get_dev_addr(virtual_core, HalL1MemAddrType::GO_MSG);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        go_msg, sizeof(go_msg_t), tt_cxy_pair(this->id(), virtual_core), go_addr);
    uint64_t launch_msg_buffer_read_ptr_addr = this->get_dev_addr(virtual_core, HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
    uint32_t zero = 0;
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        &zero, sizeof(uint32_t), tt_cxy_pair(this->id(), virtual_core), launch_msg_buffer_read_ptr_addr);
}

void Device::clear_launch_messages_on_eth_cores() {
    launch_msg_t launch_msg;
    go_msg_t go_msg;
    go_msg.signal = RUN_MSG_INIT;
    std::memset(&launch_msg, 0, sizeof(launch_msg_t));
    std::vector<launch_msg_t> init_launch_msg_data(launch_msg_buffer_num_entries, launch_msg);

    for (const auto& eth_core : this->get_active_ethernet_cores()) {
        CoreCoord phys_eth_core = this->ethernet_core_from_logical_core(eth_core);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            init_launch_msg_data.data(),
            launch_msg_buffer_num_entries * sizeof(launch_msg_t),
            tt_cxy_pair(this->id(), phys_eth_core),
            this->get_dev_addr(phys_eth_core, HalL1MemAddrType::LAUNCH));
        uint32_t go_addr = this->get_dev_addr(phys_eth_core, HalL1MemAddrType::GO_MSG);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            &go_msg, sizeof(go_msg_t), tt_cxy_pair(this->id(), phys_eth_core), go_addr);
    }
    for (const auto& eth_core : this->get_inactive_ethernet_cores()) {
        CoreCoord phys_eth_core = this->ethernet_core_from_logical_core(eth_core);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            init_launch_msg_data.data(),
            launch_msg_buffer_num_entries * sizeof(launch_msg_t),
            tt_cxy_pair(this->id(), phys_eth_core),
            this->get_dev_addr(phys_eth_core, HalL1MemAddrType::LAUNCH));
        uint32_t go_addr = this->get_dev_addr(phys_eth_core, HalL1MemAddrType::GO_MSG);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            &go_msg, sizeof(go_msg_t), tt_cxy_pair(this->id(), phys_eth_core), go_addr);
    }
}

void Device::reset_cores() {
    ZoneScoped;

    const auto& hal = MetalContext::instance().hal();
    auto get_active_erisc_launch_flag_addr = [&]() {
        auto core_type_idx =
            MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
        std::uint32_t launch_erisc_addr =
            tt::tt_metal::MetalContext::instance().hal().get_jit_build_config(core_type_idx, 0, 0).fw_launch_addr;
        return launch_erisc_addr;
    };

    auto erisc_app_still_running = [&](CoreCoord virtual_core) {
        // Check if the kernel/erisc_app is still running on a ethernet core with context switching enabled
        // The LAUNCH_ERISC_APP_FLAG is reset to 0 after reset/reboot, and set to 1 when Metal runtime launches erisc
        // app FW Only applicable to WORMHOLE ethernet cores today, but could in theory extend to other cores, remove
        // assert if so
        if (this->arch() != ARCH::WORMHOLE_B0) {
            return false;
        }
        TT_ASSERT(
            tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_core(virtual_core, this->id()),
            "Invalid core {} for context switch check",
            virtual_core.str());
        std::uint32_t launch_erisc_addr = get_active_erisc_launch_flag_addr();
        auto data =
            tt::llrt::read_hex_vec_from_core(this->id(), virtual_core, launch_erisc_addr, sizeof(std::uint32_t));
        return (data[0] != 0);
    };

    // Send exit_erisc_kernel to the launch message
    auto erisc_send_exit_signal = [&](CoreCoord virtual_core, bool is_idle_eth) {
        go_msg_t go_msg;
        std::memset(&go_msg, 0, sizeof(go_msg_t));
        log_info(
            tt::LogMetal,
            "While initializing device {}, {} ethernet dispatch core {} detected as still "
            "running, issuing exit signal.",
            this->id(),
            is_idle_eth ? "idle" : "active",
            virtual_core.str());

        DeviceAddr launch_addr = hal.get_dev_addr(
            is_idle_eth ? HalProgrammableCoreType::IDLE_ETH : HalProgrammableCoreType::ACTIVE_ETH,
            HalL1MemAddrType::LAUNCH);

        std::vector<uint32_t> data(sizeof(launch_msg_t) / sizeof(uint32_t));
        data = tt::llrt::read_hex_vec_from_core(this->id(), virtual_core, launch_addr, sizeof(launch_msg_t));

        launch_msg_t* launch_msg = (launch_msg_t*)(&data[0]);
        launch_msg->kernel_config.exit_erisc_kernel = 1;
        llrt::write_launch_msg_to_core(this->id(), virtual_core, launch_msg, &go_msg, launch_addr, false);

        if (!is_idle_eth) {
            // Active
            std::vector<uint32_t> clear_flag_data = {0};
            tt::llrt::write_hex_vec_to_core(
                this->id(), virtual_core, clear_flag_data, get_active_erisc_launch_flag_addr());
        }
    };

    auto mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(this->id_);
    // Assert worker cores + dispatch cores, in case they were in a bad state from before.
    std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> device_to_early_exit_cores;

    if (hal.get_eth_fw_is_cooperative()) {
        // Active ethernet
        for (const auto& eth_core : this->get_active_ethernet_cores()) {
            CoreCoord virtual_core = this->ethernet_core_from_logical_core(eth_core);
            if (erisc_app_still_running(virtual_core)) {
                erisc_send_exit_signal(virtual_core, false /* is_idle_eth */);
                device_to_early_exit_cores[this->id()].insert(virtual_core);
            }
        }

        // Idle ethernet
        for (const auto& eth_core : this->get_inactive_ethernet_cores()) {
            CoreCoord virtual_core = this->ethernet_core_from_logical_core(eth_core);
            if (erisc_app_still_running(virtual_core)) {
                tt::tt_metal::MetalContext::instance().get_cluster().assert_risc_reset_at_core(
                    tt_cxy_pair(this->id(), virtual_core));
            }
        }
    }

    // Early exiting dispatch cores should show RUN_MSG_DONE when they exit.
    for (auto &id_and_cores : device_to_early_exit_cores) {
        const int timeout_ms = 10000; // 10 seconds for now
        if (!id_and_cores.second.empty()) {
            try {
                llrt::internal_::wait_until_cores_done(id_and_cores.first, RUN_MSG_GO, id_and_cores.second, timeout_ms);
            } catch (std::runtime_error &e) {
                log_warning(
                    tt::LogMetal,
                    "Detected dispatch kernels still running but failed to complete an early exit. This may happen "
                    "from time to time following a reset, continuing to FW intialization...");
            }
        }
    }

    // Reset Tensix cores
    // TODO: reset BH eth cores as well
    CoreCoord grid_size = this->logical_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);
            if (!this->storage_only_cores_.contains(logical_core)) {
                tt::tt_metal::MetalContext::instance().get_cluster().assert_risc_reset_at_core(
                    tt_cxy_pair(this->id(), worker_core));
            }
        }
    }
}

void Device::initialize_and_launch_firmware() {
    ZoneScoped;

    launch_msg_t launch_msg;
    go_msg_t go_msg;
    std::memset(&launch_msg, 0, sizeof(launch_msg_t));
    go_msg.signal = RUN_MSG_INIT;
    const auto& hal = MetalContext::instance().hal();

    // Populate core info, which will be written to device
    std::vector<uint32_t> core_info_vec(sizeof(core_info_msg_t) / sizeof(uint32_t));
    core_info_msg_t *core_info = (core_info_msg_t *) core_info_vec.data();

    const metal_SocDescriptor& soc_d = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id());
    uint64_t pcie_chan_base_addr =
        tt::tt_metal::MetalContext::instance().get_cluster().get_pcie_base_addr_from_device(this->id());
    uint32_t num_host_channels = tt::tt_metal::MetalContext::instance().get_cluster().get_num_host_channels(this->id());
    uint64_t pcie_chan_end_addr = pcie_chan_base_addr;
    for (int pcie_chan = 0; pcie_chan < num_host_channels; pcie_chan++) {
        pcie_chan_end_addr +=
            tt::tt_metal::MetalContext::instance().get_cluster().get_host_channel_size(this->id(), pcie_chan);
    }
    core_info->noc_pcie_addr_base = pcie_chan_base_addr;
    core_info->noc_pcie_addr_end = pcie_chan_end_addr;
    core_info->noc_dram_addr_base = 0;
    core_info->noc_dram_addr_end = soc_d.dram_core_size;
    core_info->l1_unreserved_start = this->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    const std::vector<tt::umd::CoreCoord>& pcie_cores = soc_d.get_cores(CoreType::PCIE, soc_d.get_umd_coord_system());
    // There are multiple NoC endpoints for DRAM, but not all are exposed through the API. Watcher will flag endpoints
    // that are not exposed as invalid transactions. This helps to avoid BH issue highlighted by SYS-592 where writing
    // to multiple DRAM endpoints can hang the card.
    std::unordered_set<tt::umd::CoreCoord> dram_cores;
    for (uint32_t dram_channel = 0; dram_channel < this->num_dram_channels(); dram_channel++) {
        auto worker_dram_ep = soc_d.get_preferred_worker_core_for_dram_view(dram_channel);
        auto eth_dram_ep = soc_d.get_preferred_eth_core_for_dram_view(dram_channel);
        auto physical_worker_dram_ep =
            soc_d.translate_coord_to(worker_dram_ep, CoordSystem::TRANSLATED, CoordSystem::PHYSICAL);
        auto physical_eth_dram_ep =
            soc_d.translate_coord_to(eth_dram_ep, CoordSystem::TRANSLATED, CoordSystem::PHYSICAL);
        dram_cores.insert(physical_worker_dram_ep);
        dram_cores.insert(physical_eth_dram_ep);
    }

    const std::vector<tt::umd::CoreCoord>& eth_cores =
        soc_d.get_cores(CoreType::ETH, CoordSystem::PHYSICAL);  // make these translated and then convert to physical

    TT_ASSERT(
        pcie_cores.size() + dram_cores.size() + eth_cores.size() <= MAX_PHYSICAL_NON_WORKER_CORES,
        "Detected more pcie/dram/eth cores than fit in the device mailbox.");
    TT_ASSERT(
        eth_cores.size() <= MAX_VIRTUAL_NON_WORKER_CORES,
        "Detected more eth cores (virtual non-workers) than can fit in device mailbox.");
    for (int idx = 0; idx < MAX_PHYSICAL_NON_WORKER_CORES; idx++) {
        core_info->non_worker_cores[idx] = {CORE_COORD_INVALID, CORE_COORD_INVALID, AddressableCoreType::UNKNOWN};
    }
    for (int idx = 0; idx < MAX_VIRTUAL_NON_WORKER_CORES; idx++) {
        core_info->virtual_non_worker_cores[idx] = {CORE_COORD_INVALID, CORE_COORD_INVALID, AddressableCoreType::UNKNOWN};
    }

    // On Blackhole, virtualized Tensix coordinates overlap with NoC1 physical DRAM and PCIe coordinates beause
    // virtualized Tensix coordinates == NoC0 Tensix physical coordinates. This causes false negative Watcher
    // sanitization errors because it appears as a mixed use of physical and virtual To workaround this, skip over
    // populating `non_worker_cores` for BH DRAM when virtualization is enabled
    int non_worker_cores_idx = 0;
    bool skip_physical = this->arch() == ARCH::BLACKHOLE and hal.is_coordinate_virtualization_enabled();
    if (not skip_physical) {
        for (tt::umd::CoreCoord core : pcie_cores) {
            tt::umd::CoreCoord translated_coord =
                soc_d.translate_coord_to(tt_xy_pair(core.x, core.y), CoordSystem::PHYSICAL, CoordSystem::VIRTUAL);
            core_info->non_worker_cores[non_worker_cores_idx++] = {core.x, core.y, AddressableCoreType::PCIE};
        }
        for (tt::umd::CoreCoord core : dram_cores) {
            core_info->non_worker_cores[non_worker_cores_idx++] = {core.x, core.y, AddressableCoreType::DRAM};
        }
        for (tt::umd::CoreCoord core : eth_cores) {
            core_info->non_worker_cores[non_worker_cores_idx++] = {core.x, core.y, AddressableCoreType::ETH};
        }
    }

    if (hal.is_coordinate_virtualization_enabled()) {
        // Track Virtual Non Worker Cores (In this case only Eth) separately
        uint32_t virtual_non_worker_cores_idx = 0;
        for (tt::umd::CoreCoord core : eth_cores) {
            auto virtual_core = this->virtual_core_from_physical_core({core.x, core.y});
            core_info->virtual_non_worker_cores[virtual_non_worker_cores_idx++] = {virtual_core.x, virtual_core.y, AddressableCoreType::ETH};
        }

        if (this->arch() == ARCH::BLACKHOLE) {
            for (const CoreCoord& core : pcie_cores) {
                auto virtual_core = this->virtual_core_from_physical_core({core.x, core.y});
                core_info->virtual_non_worker_cores[virtual_non_worker_cores_idx++] = {
                    virtual_core.x, virtual_core.y, AddressableCoreType::PCIE};
            }

            for (const CoreCoord& core : dram_cores) {
                auto virtual_core = this->virtual_core_from_physical_core({core.x, core.y});
                core_info->virtual_non_worker_cores[virtual_non_worker_cores_idx++] = {
                    virtual_core.x, virtual_core.y, AddressableCoreType::DRAM};
            }
        }
    }

    // Determine which noc-coords are harvested
    std::vector<uint32_t> harvested_axis_coord;
    uint32_t harvested_noc_coords = CoordinateManager::shuffle_tensix_harvesting_mask_to_noc0_coords(
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id()).arch, tt::tt_metal::MetalContext::instance().get_cluster().get_harvesting_mask(this->id()));
    uint32_t max_along_axis =
        hal.get_tensix_harvest_axis() == HalTensixHarvestAxis::ROW ? soc_d.grid_size.y : soc_d.grid_size.x;
    for (uint32_t idx = 0; idx < max_along_axis; idx++) {
        bool harvested_axis = (harvested_noc_coords >> idx) & 0x1;
        if (harvested_axis) {
            harvested_axis_coord.push_back(idx);
        }
    }
    TT_ASSERT(
        harvested_axis_coord.size() <= MAX_HARVESTED_ON_AXIS, "Detected more harvested rows than fit in mailbox.");
    for (int idx = 0; idx < MAX_HARVESTED_ON_AXIS; idx++) {
        core_info->harvested_coords[idx] =
            (idx < harvested_axis_coord.size()) ? harvested_axis_coord[idx] : CORE_COORD_INVALID;
        // Populate harvested rows/cols in virtual coordinate space if virtualization is supported by HW.
        // Harvested rows/cols in the virtual space are placed at the end of the worker grid,
        if (hal.is_coordinate_virtualization_enabled() and idx < harvested_axis_coord.size()) {
            // On BH virtual coordinates are not contiguous
            uint32_t end_virtual_grid = hal.get_tensix_harvest_axis() == HalTensixHarvestAxis::ROW
                                            ? hal.get_virtual_worker_start_y() + this->logical_grid_size().y
                                        : (this->arch() == ARCH::BLACKHOLE)
                                            ? max_along_axis - 1
                                            : hal.get_virtual_worker_start_x() + this->logical_grid_size().x;

            // BH translated tensix cores are same as noc0 physical
            core_info->virtual_harvested_coords[idx] = end_virtual_grid + harvested_axis_coord.size() - (idx + 1);
        } else {
            core_info->virtual_harvested_coords[idx] = CORE_COORD_INVALID;
        }
    }

    core_info->noc_size_x = soc_d.grid_size.x;
    core_info->noc_size_y = soc_d.grid_size.y;
    core_info->worker_grid_size_x = this->logical_grid_size().x;  // Grid size as virtual coords see it (workers only)
    core_info->worker_grid_size_y = this->logical_grid_size().y;

    // Download to worker cores
    log_debug(tt::LogMetal, "Initializing firmware");
    CoreCoord grid_size = this->logical_grid_size();
    std::unordered_set<CoreCoord> not_done_cores;

    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            if (!this->storage_only_cores_.count(logical_core)) {
                CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);
                // Setup the absolute logical coordinates of this worker which are relative to true origin. not the sub
                // device. When running the user kernel, which potentially is on a sub device, send that info using the
                // launch message using dispatch.
                core_info->absolute_logical_x = logical_core.x;
                core_info->absolute_logical_y = logical_core.y;
                // Must write to core before starting it
                tt::llrt::write_hex_vec_to_core(
                    this->id(), worker_core, core_info_vec, this->get_dev_addr(worker_core, HalL1MemAddrType::CORE_INFO));
                this->initialize_firmware(HalProgrammableCoreType::TENSIX, worker_core, &launch_msg, &go_msg);
                not_done_cores.insert(worker_core);
            }
        }
    }

    // Clear erisc sync info
    for (const auto &eth_core : this->get_active_ethernet_cores()) {
        static std::vector<uint32_t> zero_vec_erisc_init(
            hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO) / sizeof(uint32_t),
            0);

        CoreCoord virtual_core = this->ethernet_core_from_logical_core(eth_core);

        llrt::write_hex_vec_to_core(
            this->id(),
            virtual_core,
            zero_vec_erisc_init,
            hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::APP_SYNC_INFO));
    }

    // Load erisc app base FW to eth cores on WH and active_erisc FW on second risc of BH active eth cores
    std::unordered_set<CoreCoord> active_eth_cores;
    for (const auto &eth_core : this->get_active_ethernet_cores()) {
        CoreCoord phys_eth_core = this->ethernet_core_from_logical_core(eth_core);
        core_info->absolute_logical_x = eth_core.x;
        core_info->absolute_logical_y = eth_core.y;
        tt::llrt::write_hex_vec_to_core(
            this->id(), phys_eth_core, core_info_vec, this->get_dev_addr(phys_eth_core, HalL1MemAddrType::CORE_INFO));
        this->initialize_firmware(HalProgrammableCoreType::ACTIVE_ETH, phys_eth_core, &launch_msg, &go_msg);
        if (!hal.get_eth_fw_is_cooperative()) {
            active_eth_cores.insert(phys_eth_core);
            not_done_cores.insert(phys_eth_core);
        }
    }

    for (const auto &eth_core : this->get_inactive_ethernet_cores()) {
        CoreCoord phys_eth_core = this->ethernet_core_from_logical_core(eth_core);
        core_info->absolute_logical_x = eth_core.x;
        core_info->absolute_logical_y = eth_core.y;
        tt::llrt::write_hex_vec_to_core(
            this->id(), phys_eth_core, core_info_vec, this->get_dev_addr(phys_eth_core, HalL1MemAddrType::CORE_INFO));
        this->initialize_firmware(HalProgrammableCoreType::IDLE_ETH, phys_eth_core, &launch_msg, &go_msg);
        not_done_cores.insert(phys_eth_core);
    }

    // Barrier between L1 writes above and deassert below
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(this->id());

    // Deassert worker cores
    TensixSoftResetOptions reset_val;
    for (const auto& worker_core : not_done_cores) {
        if (active_eth_cores.find(worker_core) != active_eth_cores.end()) {
            // bit 12 needs to be deasserted to run second erisc on BH
            reset_val = TENSIX_DEASSERT_SOFT_RESET &
                        static_cast<TensixSoftResetOptions>(
                            ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::TRISC0));
        } else {
            reset_val = TENSIX_DEASSERT_SOFT_RESET;
        }
        tt::tt_metal::MetalContext::instance().get_cluster().deassert_risc_reset_at_core(
            tt_cxy_pair(this->id(), worker_core), reset_val);
    }

    // Wait until fw init is done, ensures the next launch msg doesn't get
    // written while fw is still in init
    log_debug(tt::LogMetal, "Waiting for firmware init complete");
    const int timeout_ms = 10000; // 10 seconds for now
    try {
        llrt::internal_::wait_until_cores_done(this->id(), RUN_MSG_INIT, not_done_cores, timeout_ms);
    } catch (std::runtime_error &e) {
        TT_THROW("Device {} init: failed to initialize FW! Try resetting the board.", this->id());
    }
    log_debug(tt::LogMetal, "Firmware init complete");
}

void Device::clear_l1_state() {
    log_debug(tt::LogMetal, "Clearing L1 for device {}", this->id_);
    // Clear all clearable Tensix and Eth L1
    CoreCoord logical_grid_size = this->logical_grid_size();
    TT_ASSERT(this->l1_size_per_core() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> zero_vec(this->l1_size_per_core() / sizeof(uint32_t), 0);
    constexpr uint32_t start_address = 0;
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            CoreCoord logical_core(x, y);
            detail::WriteToDeviceL1(this, logical_core, start_address, zero_vec);
        }
    }

    // Clear erisc unreserved L1
    for (const auto& eth_core : this->get_active_ethernet_cores()) {
        static uint32_t zero_vec_size = tt::tt_metal::hal::get_erisc_l1_unreserved_size();
        auto zero_vec_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

        static std::vector<uint32_t> zero_vec(zero_vec_size / sizeof(uint32_t), 0);

        CoreCoord virtual_core = this->ethernet_core_from_logical_core(eth_core);

        llrt::write_hex_vec_to_core(this->id(), virtual_core, zero_vec, zero_vec_addr);
    }
    // TODO: clear idle eriscs as well
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(this->id());
}

void Device::clear_dram_state() {
    log_debug(tt::LogMetal, "Clearing DRAM for device {}", this->id_);

    TT_ASSERT(this->dram_size_per_channel() % sizeof(uint32_t) == 0);
    constexpr uint32_t start_address = 0;
    const int num_dram_channels = this->num_dram_channels();
    std::vector<uint32_t> zero_vec(this->dram_size_per_channel() / sizeof(uint32_t), 0);
    for (int channel = 0; channel < num_dram_channels; ++channel) {
        detail::WriteToDeviceDRAMChannel(this, channel, start_address, zero_vec);
    }

    tt::tt_metal::MetalContext::instance().get_cluster().dram_barrier(this->id());
}

void Device::compile_command_queue_programs() {
    ZoneScoped;
    if (this->is_mmio_capable() && !tt::tt_metal::MetalContext::instance().rtoptions().get_fd_fabric()) {
        auto command_queue_program_ptr = create_and_compile_cq_program(this);
        this->command_queue_programs_.push_back(std::move(command_queue_program_ptr));
        // Since devices could be set up in any order, on mmio device do a pass and populate cores for tunnelers.
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_mmio_device_tunnel_count(this->id_) > 0) {
            tunnels_from_mmio_ =
                tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(this->id_);
            for (auto& tunnel : tunnels_from_mmio_) {
                for (uint32_t tunnel_stop = 0; tunnel_stop < tunnel.size() - 1; tunnel_stop++) {
                    chip_id_t device_id = tunnel[tunnel_stop];
                    chip_id_t ds_device_id = tunnel[tunnel_stop + 1];
                    uint16_t channel =
                        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(
                            ds_device_id);
                    // Only one tunneler per connection, use CQ ID 0
                    MetalContext::instance().get_dispatch_core_manager().tunneler_core(
                        device_id, ds_device_id, channel, 0);
                }
            }
        }
    } else {
        auto command_queue_program_ptr = create_and_compile_cq_program(this);
        this->command_queue_programs_.push_back(std::move(command_queue_program_ptr));
    }
}

// Writes issue and completion queue pointers to device and in sysmem and loads fast dispatch program onto dispatch cores
void Device::configure_command_queue_programs() {
    chip_id_t device_id = this->id();
    chip_id_t mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    IDevice* mmio_device = tt::DevicePool::instance().get_active_device(mmio_device_id);

    std::vector<uint32_t> zero = {0x0}; // Reset state in case L1 Clear is disabled.
    std::vector<uint32_t> pointers;
    uint32_t cq_size = this->sysmem_manager().get_cq_size();
    TT_ASSERT(this->command_queue_programs_.size() == 1);

    Program& command_queue_program = *this->command_queue_programs_[0];
    uint8_t num_hw_cqs = this->num_hw_cqs();

    // Reset host-side command queue pointers for all channels controlled by this mmio device
    if (this->is_mmio_capable()) {
        for (chip_id_t serviced_device_id :
             tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(device_id)) {
            uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(
                serviced_device_id);
            uint32_t host_issue_q_rd_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
                CommandQueueHostAddrType::ISSUE_Q_RD);
            uint32_t host_issue_q_wr_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
                CommandQueueHostAddrType::ISSUE_Q_WR);
            uint32_t host_completion_q_wr_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
                CommandQueueHostAddrType::COMPLETION_Q_WR);
            uint32_t host_completion_q_rd_ptr = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
                CommandQueueHostAddrType::COMPLETION_Q_RD);
            uint32_t cq_start = MetalContext::instance().dispatch_mem_map().get_host_command_queue_addr(
                CommandQueueHostAddrType::UNRESERVED);
            pointers.resize(cq_start/sizeof(uint32_t));
            for (uint8_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
                // Reset the host manager's pointer for this command queue
                this->sysmem_manager_->reset(cq_id);

                pointers[host_issue_q_rd_ptr / sizeof(uint32_t)] = (cq_start + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
                pointers[host_issue_q_wr_ptr / sizeof(uint32_t)] = (cq_start + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
                pointers[host_completion_q_wr_ptr / sizeof(uint32_t)] = (cq_start + this->sysmem_manager_->get_issue_queue_size(cq_id) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;
                pointers[host_completion_q_rd_ptr / sizeof(uint32_t)] = (cq_start + this->sysmem_manager_->get_issue_queue_size(cq_id) + get_absolute_cq_offset(channel, cq_id, cq_size)) >> 4;

                tt::tt_metal::MetalContext::instance().get_cluster().write_sysmem(
                    pointers.data(),
                    pointers.size() * sizeof(uint32_t),
                    get_absolute_cq_offset(channel, cq_id, cq_size),
                    mmio_device_id,
                    get_umd_channel(channel));
            }
        }
    }

    // Write device-side cq pointers
    configure_dispatch_cores(this);

    // Run the cq program
    command_queue_program.finalize_offsets(this);
    detail::ConfigureDeviceWithProgram(this, command_queue_program, true);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(this->id());
}

void Device::init_command_queue_host() {
    using_fast_dispatch_ = true;
    sysmem_manager_ = std::make_unique<SystemMemoryManager>(this->id_, this->num_hw_cqs());

    auto cq_shared_state = std::make_shared<CQSharedState>();
    cq_shared_state->sub_device_cq_owner.resize(1);
    command_queues_.reserve(num_hw_cqs());
    for (size_t cq_id = 0; cq_id < num_hw_cqs(); cq_id++) {
        command_queues_.push_back(std::make_unique<HWCommandQueue>(
            this, cq_shared_state, cq_id, k_dispatch_downstream_noc, completion_queue_reader_core_));
    }
}

void Device::init_command_queue_device() {
    if (tt_metal::MetalContext::instance().rtoptions().get_skip_loading_fw()) {
        detail::EnablePersistentKernelCache();
        this->compile_command_queue_programs();
        detail::DisablePersistentKernelCache();
    } else {
        this->compile_command_queue_programs();
    }

    TT_ASSERT(this->command_queue_programs_.size() == 1);
    this->configure_command_queue_programs();
    Program& command_queue_program = *this->command_queue_programs_[0];

    // TODO: should get a const ref
    std::vector<std::vector<CoreCoord>>logical_cores = command_queue_program.logical_cores();
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        const auto& logical_dispatch_cores = logical_cores[index];
        CoreType core_type = hal.get_core_type(index);
        for (const CoreCoord &logical_dispatch_core : logical_dispatch_cores) {
            launch_msg_t msg = command_queue_program.impl().kernels_on_core(logical_dispatch_core, index)->launch_msg;
            go_msg_t go_msg = command_queue_program.impl().kernels_on_core(logical_dispatch_core, index)->go_msg;
            CoreCoord virtual_core = this->virtual_core_from_logical_core(logical_dispatch_core, core_type);
            tt::llrt::write_launch_msg_to_core(this->id(), virtual_core, &msg, &go_msg, this->get_dev_addr(virtual_core, HalL1MemAddrType::LAUNCH));
        }
    }
    // Set num_worker_sems and go_signal_noc_data on dispatch for the default sub device config
    for (auto& hw_cq : this->command_queues_) {
        hw_cq->set_go_signal_noc_data_and_dispatch_sems(
            sub_device_manager_tracker_->get_active_sub_device_manager()->num_sub_devices(),
            sub_device_manager_tracker_->get_active_sub_device_manager()->noc_mcast_unicast_data());
    }
}

void Device::init_fabric() {
    fabric_program_ = create_and_compile_fabric_program(this);
    if (fabric_program_ == nullptr) {
        return;
    }

    configure_fabric_cores(this);

    fabric_program_->finalize_offsets(this);

    detail::WriteRuntimeArgsToDevice(this, *fabric_program_, this->using_fast_dispatch());
    detail::ConfigureDeviceWithProgram(this, *fabric_program_, this->using_fast_dispatch());

    // Note: the l1_barrier below is needed to be sure writes to cores that
    // don't get the GO mailbox (eg, storage cores) have all landed
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(this->id());
    std::vector<std::vector<CoreCoord>> logical_cores_used_in_program = fabric_program_->logical_cores();
    const auto& hal = MetalContext::instance().hal();
    for (uint32_t programmable_core_type_index = 0; programmable_core_type_index < logical_cores_used_in_program.size();
         programmable_core_type_index++) {
        CoreType core_type = hal.get_core_type(programmable_core_type_index);
        for (const auto& logical_core : logical_cores_used_in_program[programmable_core_type_index]) {
            launch_msg_t* msg =
                &fabric_program_->impl().kernels_on_core(logical_core, programmable_core_type_index)->launch_msg;
            go_msg_t* go_msg =
                &fabric_program_->impl().kernels_on_core(logical_core, programmable_core_type_index)->go_msg;
            msg->kernel_config.host_assigned_id = fabric_program_->get_runtime_id();

            auto physical_core = this->virtual_core_from_logical_core(logical_core, core_type);
            tt::llrt::write_launch_msg_to_core(
                this->id(), physical_core, msg, go_msg, this->get_dev_addr(physical_core, HalL1MemAddrType::LAUNCH));
        }
    }
    log_info(tt::LogMetal, "Fabric initialized on Device {}", this->id_);
}

bool Device::initialize(
    const uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t worker_l1_size,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    bool minimal) {
    ZoneScoped;
    log_info(tt::LogMetal, "Initializing device {}. Program cache is {}enabled", this->id_, this->program_cache_.is_enabled() ? "": "NOT ");
    log_debug(tt::LogMetal, "Running with {} cqs ", num_hw_cqs);
    TT_FATAL(num_hw_cqs > 0 and num_hw_cqs <= dispatch_core_manager::MAX_NUM_HW_CQS, "num_hw_cqs can be between 1 and {}", dispatch_core_manager::MAX_NUM_HW_CQS);
    this->using_fast_dispatch_ = false;
    // Trying to preserve logic that was in device_pool.cpp
    // However, I honestly don't understand it
    this->num_hw_cqs_ = num_hw_cqs;
    const auto& hal = MetalContext::instance().hal();
    if (worker_l1_size == 0) {
        worker_l1_size = hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    }

    size_t max_worker_l1_size = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
                                hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) -
                                hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);

    TT_FATAL(
        worker_l1_size <= max_worker_l1_size,
        "Worker L1 size {} is larger than max size {}",
        worker_l1_size,
        max_worker_l1_size);
    log_debug(tt::LogMetal, "Worker L1 size: {} Max: {}", worker_l1_size, max_worker_l1_size);
    std::uint32_t max_alignment = std::max(hal.get_alignment(HalMemType::DRAM), hal.get_alignment(HalMemType::L1));
    uint32_t worker_l1_unreserved_start = tt::align(
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
            hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) - worker_l1_size,
        max_alignment);
    this->initialize_default_sub_device_state(
        l1_small_size, trace_region_size, worker_l1_unreserved_start, l1_bank_remap);
    this->generate_device_bank_to_noc_tables();

    // For minimal setup, don't initialize FW, watcher, dprint. They won't work if we're attaching to a hung chip.
    if (minimal)
        return true;

    this->initialized_ = true;
    // Clear the entire launch message ring buffer on ethernet cores before application firmware is activated.
    // This is required since ethernet cores context switch between application and routing firmware.
    // If ERISC application firmware is activated before the launch messages are cleared, it can enter an undefined
    // state by reading a corrupted launch message. Routing firmware will never run in this case, causing UMD issued
    // transactions to hang.
    this->clear_launch_messages_on_eth_cores();

    return true;
}

bool Device::close() {
    log_info(tt::LogMetal, "Closing device {}", this->id_);
    if (not this->initialized_) {
        TT_THROW("Cannot close device {} that has not been initialized!", this->id_);
    }

    tt_metal::detail::DumpDeviceProfileResults(this, ProfilerDumpState::LAST_CLOSE_DEVICE);

    this->disable_and_clear_program_cache();
    this->set_program_cache_misses_allowed(true);

    sub_device_manager_tracker_.reset(nullptr);

    DprintServerDetach(this->id());
    watcher_detach(this->id());

    // Assert worker cores only for this device
    auto dispatch_cores = tt::tt_metal::get_virtual_dispatch_cores(this->id());
    auto routing_cores = tt::tt_metal::get_virtual_dispatch_routing_cores(this->id());

    CoreCoord grid_size = this->logical_grid_size();
    for (uint32_t y = 0; y < grid_size.y; y++) {
        for (uint32_t x = 0; x < grid_size.x; x++) {
            CoreCoord logical_core(x, y);
            CoreCoord worker_core = this->worker_core_from_logical_core(logical_core);

            if (!dispatch_cores.contains(worker_core) && !routing_cores.contains(worker_core)) {
                if (!this->storage_only_cores_.contains(logical_core)) {
                    tt::tt_metal::MetalContext::instance().get_cluster().assert_risc_reset_at_core(
                        tt_cxy_pair(this->id(), worker_core));
                }
            } else {
                log_debug(tt::LogMetal, "{} will not be Reset when closing Device {}", worker_core.str(), this->id());
            }
        }
    }

    if (!MetalContext::instance().hal().get_eth_fw_is_cooperative()) {
        for (const auto& eth_core : this->get_active_ethernet_cores()) {
            CoreCoord virtual_eth_core = this->ethernet_core_from_logical_core(eth_core);
            TensixSoftResetOptions reset_val =
                TENSIX_ASSERT_SOFT_RESET &
                static_cast<TensixSoftResetOptions>(
                    ~std::underlying_type<TensixSoftResetOptions>::type(TensixSoftResetOptions::BRISC));
            tt::tt_metal::MetalContext::instance().get_cluster().assert_risc_reset_at_core(
                tt_cxy_pair(this->id(), virtual_eth_core), reset_val);
        }
    }

    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(id_);

    this->compute_cores_.clear();
    this->storage_only_cores_.clear();
    this->ethernet_cores_.clear();
    this->command_queue_programs_.clear();
    this->command_queues_.clear();
    this->sysmem_manager_.reset();
    this->initialized_ = false;

    return true;
}

Device::~Device() {
    log_debug(tt::LogMetal, "Device {} destructor", this->id_);
    if (this->initialized_) {
        this->close();
    }
}

tt::ARCH Device::arch() const { return tt::tt_metal::MetalContext::instance().get_cluster().arch(); }

int Device::num_dram_channels() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).get_num_dram_views();
}

uint32_t Device::l1_size_per_core() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).worker_l1_size;
}
uint32_t Device::dram_size_per_channel() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).dram_view_size;
}

CoreCoord Device::grid_size() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).grid_size;
}

CoreCoord Device::logical_grid_size() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).get_grid_size(CoreType::TENSIX);
}

CoreCoord Device::dram_grid_size() const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).get_dram_grid_size();
}

CoreCoord Device::compute_with_storage_grid_size() const {
    const auto& dispatch_core_config = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config();
    return tt::get_compute_grid_size(id_, num_hw_cqs_, dispatch_core_config);
}

CoreCoord Device::virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const {
    if (coord.x >= this->grid_size().x || coord.y >= this->grid_size().y || this->arch() == ARCH::BLACKHOLE) {
        // Coordinate already in virtual space: NOC0 and NOC1 are the same
        return coord;
    } else {
        const auto& grid_size = this->grid_size();
        // Coordinate in Physical NOC0 Space. Convert to Virtual.
        coord = this->virtual_core_from_physical_core(coord);
        // Derive virtual coord in noc_index space.
        CoreCoord virtual_coord = {
            MetalContext::instance().hal().noc_coordinate(noc_index, grid_size.x, coord.x),
            MetalContext::instance().hal().noc_coordinate(noc_index, grid_size.y, coord.y)};
        return virtual_coord;
    }
}

CoreCoord Device::physical_worker_core_from_logical_core(const CoreCoord &logical_core) const {
    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id_);
    return soc_desc.get_physical_tensix_core_from_logical(logical_core);
}

std::vector<CoreCoord> Device::worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> worker_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++)
        worker_cores[idx] = this->worker_core_from_logical_core(logical_cores[idx]);

    return worker_cores;
}

std::vector<CoreCoord> Device::ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const {
    std::vector<CoreCoord> eth_cores(logical_cores.size());
    for (std::size_t idx = 0; idx < logical_cores.size(); idx++) {
        eth_cores[idx] = this->ethernet_core_from_logical_core(logical_cores[idx]);
    }
    return eth_cores;
}

CoreCoord Device::virtual_core_from_logical_core(const CoreCoord &logical_coord, const CoreType& core_type) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
        this->id_, logical_coord, core_type);
}

CoreCoord Device::virtual_core_from_physical_core(const CoreCoord& physical_coord) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_physical_coordinates(
        this->id_, physical_coord);
}

CoreCoord Device::worker_core_from_logical_core(const CoreCoord &logical_core) const {
    return this->virtual_core_from_logical_core(logical_core, CoreType::WORKER);
}

CoreCoord Device::ethernet_core_from_logical_core(const CoreCoord &logical_core) const {
    return this->virtual_core_from_logical_core(logical_core, CoreType::ETH);
}

CoreCoord Device::logical_core_from_ethernet_core(const CoreCoord &ethernet_core) const {
    return tt::tt_metal::MetalContext::instance().get_cluster().get_logical_ethernet_core_from_virtual(
        this->id(), ethernet_core);
}

uint32_t Device::get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const {
    auto virtual_noc_coord = this->virtual_noc0_coordinate(noc_index, core);
    return tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(virtual_noc_coord.x, virtual_noc_coord.y);
}

uint32_t Device::get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const {
    auto virtual_noc_start = this->virtual_noc0_coordinate(noc_index, cores.start_coord);
    auto virtual_noc_end = this->virtual_noc0_coordinate(noc_index, cores.end_coord);

    // NOC 1 mcasts from bottom left to top right, so we need to reverse the coords
    if (noc_index == 0) {
        return tt::tt_metal::MetalContext::instance().hal().noc_multicast_encoding(
            virtual_noc_start.x, virtual_noc_start.y, virtual_noc_end.x, virtual_noc_end.y);
    } else {
        return tt::tt_metal::MetalContext::instance().hal().noc_multicast_encoding(
            virtual_noc_end.x, virtual_noc_end.y, virtual_noc_start.x, virtual_noc_start.y);
    }
}

const std::unique_ptr<Allocator>& Device::allocator() const {
    return sub_device_manager_tracker_->get_default_sub_device_manager()->allocator(SubDeviceId{0});
}

const std::unique_ptr<Allocator>& Device::allocator(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->allocator(sub_device_id);
}

uint32_t Device::num_sub_devices() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->num_sub_devices();
}

CoreCoord Device::dram_core_from_dram_channel(uint32_t dram_channel) const {
    return tt::tt_metal::MetalContext::instance()
        .get_cluster()
        .get_soc_desc(id_)
        .get_preferred_worker_core_for_dram_view(dram_channel);
}

CoreCoord Device::logical_core_from_dram_channel(uint32_t dram_channel) const {
    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id_);
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).get_logical_core_for_dram_view(
        dram_channel);
}

uint32_t Device::dram_channel_from_logical_core(const CoreCoord& logical_core) const {
    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id_);
    return tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(id_).get_dram_channel_from_logical_core(
        logical_core);
}

uint32_t Device::dram_channel_from_virtual_core(const CoreCoord& virtual_core) const {
    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id_);
    for (uint32_t channel = 0; channel < this->num_dram_channels(); ++channel) {
        if (soc_desc.get_preferred_worker_core_for_dram_view(channel) == virtual_core) {
            return channel;
        }
    }
    TT_THROW("Virtual core {} is not a DRAM core", virtual_core.str());
}

std::optional<DeviceAddr> Device::lowest_occupied_compute_l1_address() const {
    return sub_device_manager_tracker_->lowest_occupied_compute_l1_address();
}

std::optional<DeviceAddr> Device::lowest_occupied_compute_l1_address(tt::stl::Span<const SubDeviceId> sub_device_ids) const {
    return sub_device_manager_tracker_->lowest_occupied_compute_l1_address(sub_device_ids);
}

CommandQueue& Device::command_queue(size_t cq_id) {
    detail::DispatchStateCheck(using_fast_dispatch_);
    TT_FATAL(cq_id < command_queues_.size(), "cq_id {} is out of range", cq_id);
    TT_FATAL(this->is_initialized(), "Device has not been initialized, did you forget to call InitializeDevice?");
    return *command_queues_[cq_id];
}

bool Device::using_slow_dispatch() const {
    return !using_fast_dispatch();
}

bool Device::using_fast_dispatch() const {
    return using_fast_dispatch_;
}

void Device::begin_trace(const uint8_t cq_id, const uint32_t tid) {
    ZoneScoped;

    TracyTTMetalBeginTrace(this->id(), tid);
    TT_FATAL(
        !this->command_queues_[cq_id]->tid().has_value(),
        "CQ {} is already being used for tracing tid {}",
        (uint32_t)cq_id,
        tid);
    this->mark_allocations_safe();
    // Create an empty trace buffer here. This will get initialized in end_trace
    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    TT_FATAL(
        active_sub_device_manager->get_trace(tid) == nullptr,
        "Trace already exists for tid {} on device {}'s active sub-device manager {}",
        tid,
        this->id_,
        active_sub_device_manager->id());
    auto& trace_buffer = active_sub_device_manager->create_trace(tid);
    this->command_queues_[cq_id]->record_begin(tid, trace_buffer->desc);
}

void Device::end_trace(const uint8_t cq_id, const uint32_t tid) {
    ZoneScoped;
    TracyTTMetalEndTrace(this->id(), tid);
    TT_FATAL(
        this->command_queues_[cq_id]->tid() == tid, "CQ {} is not being used for tracing tid {}", (uint32_t)cq_id, tid);
    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    auto trace_buffer = active_sub_device_manager->get_trace(tid);
    TT_FATAL(
        trace_buffer != nullptr,
        "Trace instance {} must exist on device {}'s active sub-device manager {}",
        tid,
        this->id_,
        active_sub_device_manager->id());
    this->command_queues_[cq_id]->record_end();

    // Capture Trace if light metal trace capturing is enabled.
    auto& lm_capture_ctx = LightMetalCaptureContext::get();
    if (lm_capture_ctx.is_tracing()) {
        lm_capture_ctx.capture_trace_descriptor(*trace_buffer->desc, tid);
    }

    Trace::initialize_buffer(this->command_queue(cq_id), trace_buffer);
    this->mark_allocations_unsafe();
}

// Load the TraceDescriptor for a given trace_id to the device. A combination of logic from begin/end_trace.
void Device::load_trace(const uint8_t cq_id, const uint32_t trace_id, const TraceDescriptor& trace_desc) {
    this->mark_allocations_safe();

    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    TT_FATAL(
        active_sub_device_manager->get_trace(trace_id) == nullptr,
        "Trace already exists for trace_id {} on device {}'s active sub-device manager {}",
        trace_id,
        this->id_,
        active_sub_device_manager->id());

    auto& trace_buffer = active_sub_device_manager->create_trace(trace_id);
    *trace_buffer->desc = trace_desc;
    Trace::initialize_buffer(this->command_queue(cq_id), trace_buffer);
    this->mark_allocations_unsafe();
}

void Device::replay_trace(
    const uint8_t cq_id, const uint32_t tid, const bool block_on_device, const bool block_on_worker_thread) {
    // If blocking, ensure that worker thread blocks until trace is completed
    ZoneScoped;
    TracyTTMetalReplayTrace(this->id(), tid);
    constexpr bool check = false;
    auto* active_sub_device_manager = sub_device_manager_tracker_->get_active_sub_device_manager();
    const auto& trace_buffer = active_sub_device_manager->get_trace(tid);
    TT_FATAL(
        trace_buffer != nullptr,
        "Trace instance {} must exist on device {}'s active sub-device manager {}",
        tid,
        this->id_,
        active_sub_device_manager->id());
    if constexpr (check) {
        trace_buffer->validate();
    }
    EnqueueTrace(this->command_queue(cq_id), tid, block_on_device);
}

void Device::release_trace(const uint32_t tid) {
    ZoneScoped;
    TracyTTMetalReleaseTrace(this->id(), tid);

    sub_device_manager_tracker_->get_active_sub_device_manager()->release_trace(tid);

    // Only enable allocations once all captured traces are released
    if (this->trace_buffers_size_ == 0) {
        this->mark_allocations_safe();
    }
}

std::shared_ptr<TraceBuffer> Device::get_trace(uint32_t tid) {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_trace(tid);
}

void Device::enable_program_cache() {
    log_info(tt::LogMetal, "Enabling program cache on device {}", this->id_);
    program_cache_.enable();
}
void Device::disable_and_clear_program_cache() {
    log_info(tt::LogMetal, "Disabling and clearing program cache on device {}", this->id_);
    if (this->program_cache_.is_enabled()) {
        program_cache_.disable();
    }
    program_cache_.clear();
}
std::size_t Device::num_program_cache_entries() { return program_cache_.num_entries(); }

void Device::mark_allocations_unsafe() { this->allocator()->mark_allocations_unsafe(); }

void Device::mark_allocations_safe() { this->allocator()->mark_allocations_safe(); }

void Device::generate_device_bank_to_noc_tables()
{
    const auto& allocator = this->allocator();
    const size_t num_dram_banks = allocator->get_num_banks(BufferType::DRAM);
    std::vector<CoreCoord> dram_noc_coord_per_bank(num_dram_banks);
    dram_bank_offset_map_.clear();
    dram_bank_offset_map_.resize(num_dram_banks);
    for (unsigned bank_id = 0; bank_id < num_dram_banks; bank_id++) {
        auto physical_dram_core = this->dram_core_from_dram_channel(allocator->get_dram_channel_from_bank_id(bank_id));
        dram_noc_coord_per_bank[bank_id] = physical_dram_core;
        dram_bank_offset_map_[bank_id] = allocator->get_bank_offset(BufferType::DRAM, bank_id);
    }
    const size_t num_l1_banks = allocator->get_num_banks(BufferType::L1);
    std::vector<CoreCoord> l1_noc_coord_per_bank(num_l1_banks);
    l1_bank_offset_map_.clear();
    l1_bank_offset_map_.resize(num_l1_banks);
    for (unsigned bank_id = 0; bank_id < num_l1_banks; bank_id++) {
        l1_noc_coord_per_bank[bank_id] =
            this->worker_core_from_logical_core(allocator->get_logical_core_from_bank_id(bank_id));
        l1_bank_offset_map_[bank_id] = allocator->get_bank_offset(BufferType::L1, bank_id);
    }

    const metal_SocDescriptor& soc_d = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id());

    const auto& hal = MetalContext::instance().hal();
    dram_bank_to_noc_xy_.clear();
    dram_bank_to_noc_xy_.reserve(hal.get_num_nocs() * dram_noc_coord_per_bank.size());
    bool dram_is_virtualized = hal.get_virtualized_core_types().find(AddressableCoreType::DRAM) !=
                               hal.get_virtualized_core_types().end();
    for (unsigned int noc = 0; noc < hal.get_num_nocs(); noc++) {
        for (unsigned int bank_id = 0; bank_id < dram_noc_coord_per_bank.size(); bank_id++) {
            uint16_t noc_x, noc_y;
            if (dram_is_virtualized) {
                noc_x = dram_noc_coord_per_bank[bank_id].x;
                noc_y = dram_noc_coord_per_bank[bank_id].y;
            } else {
                noc_x = hal.noc_coordinate(noc, soc_d.grid_size.x, dram_noc_coord_per_bank[bank_id].x);
                noc_y = hal.noc_coordinate(noc, soc_d.grid_size.y, dram_noc_coord_per_bank[bank_id].y);
            }
            uint16_t xy = ((noc_y << hal.get_noc_addr_node_id_bits()) | noc_x) << hal.get_noc_coord_reg_offset();
            dram_bank_to_noc_xy_.push_back(xy);
        }
    }

    l1_bank_to_noc_xy_.clear();
    l1_bank_to_noc_xy_.reserve(hal.get_num_nocs() * l1_noc_coord_per_bank.size());
    for (unsigned int noc = 0; noc < hal.get_num_nocs(); noc++) {
        for (unsigned int bank_id = 0; bank_id < l1_noc_coord_per_bank.size(); bank_id++) {
            auto l1_noc_coords = this->virtual_noc0_coordinate(noc, l1_noc_coord_per_bank[bank_id]);
            uint16_t noc_x = l1_noc_coords.x;
            uint16_t noc_y = l1_noc_coords.y;
            uint16_t xy = ((noc_y << hal.get_noc_addr_node_id_bits()) | noc_x) << hal.get_noc_coord_reg_offset();
            l1_bank_to_noc_xy_.push_back(xy);
        }
    }
}

uint8_t Device::num_noc_mcast_txns(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->num_noc_mcast_txns(sub_device_id);
}

uint8_t Device::num_noc_unicast_txns(SubDeviceId sub_device_id) const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->num_noc_unicast_txns(sub_device_id);
}

uint8_t Device::noc_data_start_index(SubDeviceId sub_device_id, bool mcast_data, bool unicast_data) const {
    if (mcast_data) {
        return sub_device_manager_tracker_->get_active_sub_device_manager()->noc_mcast_data_start_index(sub_device_id);
    } else if (unicast_data) {
        return sub_device_manager_tracker_->get_active_sub_device_manager()->noc_unicast_data_start_index(
            sub_device_id);
    } else {
        return 0;
    }
}

CoreCoord Device::virtual_program_dispatch_core(uint8_t cq_id) const {
    return this->command_queues_[cq_id]->virtual_enqueue_program_dispatch_core();
}

SubDeviceManagerId Device::get_active_sub_device_manager_id() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->id();
}

SubDeviceManagerId Device::get_default_sub_device_manager_id() const {
    return sub_device_manager_tracker_->get_default_sub_device_manager()->id();
}

SubDeviceManagerId Device::create_sub_device_manager(tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    return sub_device_manager_tracker_->create_sub_device_manager(sub_devices, local_l1_size);
}

std::tuple<SubDeviceManagerId, SubDeviceId> Device::create_sub_device_manager_with_fabric(tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    return sub_device_manager_tracker_->create_sub_device_manager_with_fabric(sub_devices, local_l1_size);
}

void Device::load_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    sub_device_manager_tracker_->load_sub_device_manager(sub_device_manager_id);
}

void Device::clear_loaded_sub_device_manager() { sub_device_manager_tracker_->clear_loaded_sub_device_manager(); }

void Device::remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id) {
    sub_device_manager_tracker_->remove_sub_device_manager(sub_device_manager_id);
}

const std::vector<SubDeviceId> &Device::get_sub_device_ids() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_sub_device_ids();
}

const std::vector<SubDeviceId> &Device::get_sub_device_stall_group() const {
    return sub_device_manager_tracker_->get_active_sub_device_manager()->get_sub_device_stall_group();
}

void Device::set_sub_device_stall_group(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    sub_device_manager_tracker_->get_active_sub_device_manager()->set_sub_device_stall_group(sub_device_ids);
}

void Device::reset_sub_device_stall_group() {
    sub_device_manager_tracker_->get_active_sub_device_manager()->reset_sub_device_stall_group();
}

std::vector<CoreCoord> Device::get_optimal_dram_bank_to_logical_worker_assignment() {
    // Top level function that users (ex: Op Writers) can use to assign Tensix Worker cores
    // as DRAM readers or writers. Returns logical coordinates of optimally placed workers.
    // This function queries Physical Coordinates (only exposed directly to the Device class)
    // and passes them to logic in core_assignment.cpp to derive the most optimal core placement
    // based on architecture specific logic and Physical Grid configuration.
    if (not this->optimal_dram_bank_to_logical_worker_assignment_.size()) {
        uint32_t full_grid_size_x = this->grid_size().x;
        uint32_t full_grid_size_y = this->grid_size().y;

        auto compute_with_storage_grid_size = this->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        // Get physical coordinates of DRAM Controller NOC end-points
        uint32_t num_dram_banks = this->num_dram_channels();

        const auto& hal = MetalContext::instance().hal();
        bool dram_is_virtualized =
            hal.get_virtualized_core_types().find(AddressableCoreType::DRAM) != hal.get_virtualized_core_types().end();
        const metal_SocDescriptor& soc_d =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(this->id());
        std::vector<CoreCoord> dram_phy_coords;
        for (int i = 0; i < num_dram_banks; ++i) {
            auto dram_core = dram_core_from_dram_channel(i);
            if (dram_is_virtualized) {
                tt::umd::CoreCoord umd_dram_coord = soc_d.translate_coord_to(
                    tt_xy_pair(dram_core.x, dram_core.y), CoordSystem::TRANSLATED, CoordSystem::PHYSICAL);
                dram_core = CoreCoord(umd_dram_coord.x, umd_dram_coord.y);
            }
            dram_phy_coords.push_back(dram_core);
        }
        // Get all logical cores in the worker grid
        std::vector<CoreCoord> all_worker_cores_logical;
        for (int i = 0; i < num_cores_x; ++i) {
            for (int j = 0; j < num_cores_y; ++j) {
                all_worker_cores_logical.push_back(CoreCoord(i, j));
            }
        }
        // Get the physical rows and cols  (y, x) in the worker grid
        std::vector<uint32_t> worker_phy_y = std::vector<uint32_t>(num_cores_y);
        for (int i = 0; i < num_cores_y; ++i) {
            auto core_phy = this->physical_worker_core_from_logical_core(CoreCoord(0, i));
            worker_phy_y.at(i) = core_phy.y;
        }
        std::vector<uint32_t> worker_phy_x = std::vector<uint32_t>(num_cores_x);
        for (int i = 0; i < num_cores_x; ++i) {
            auto core_phy = this->physical_worker_core_from_logical_core(CoreCoord(i, 0));
            worker_phy_x.push_back(core_phy.x);
        }
        // Get optimal placement of worker cores interfacing with DRAM Controllers in physical coordinate space
        auto physical_worker_cores = get_optimal_dram_to_physical_worker_assignment(this->arch(), dram_phy_coords, full_grid_size_x, full_grid_size_y, worker_phy_x, worker_phy_y);
        // Convert to physical worker coordinates to logical. This gets returned to the user.
        for (int i = 0; i < physical_worker_cores.size(); ++i) {
            for (int j = 0; j < all_worker_cores_logical.size(); ++j) {
                auto core = this->physical_worker_core_from_logical_core(all_worker_cores_logical[j]);
                if (physical_worker_cores[i] == core) {
                    this->optimal_dram_bank_to_logical_worker_assignment_.push_back(all_worker_cores_logical[j]);
                }
            }
        }
    }
    return this->optimal_dram_bank_to_logical_worker_assignment_;
}

HalProgrammableCoreType Device::get_programmable_core_type(CoreCoord virtual_core) const {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_core(virtual_core, this->id_)) {
        return HalProgrammableCoreType::TENSIX;
    }

    // Eth pcores have a different address, but only active ones.
    CoreCoord logical_core = this->logical_core_from_ethernet_core(virtual_core);
    if (this->is_active_ethernet_core(logical_core)) {
        return HalProgrammableCoreType::ACTIVE_ETH;
    }

    return HalProgrammableCoreType::IDLE_ETH;
}

HalMemType Device::get_mem_type_of_core(CoreCoord virtual_core) const {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_core(virtual_core, this->id_) &&
        !tt::tt_metal::MetalContext::instance().get_cluster().is_worker_core(virtual_core, this->id_)) {
        return HalMemType::DRAM;
    } else {
        return HalMemType::L1;
    }
}

std::shared_ptr<distributed::MeshDevice> Device::get_mesh_device() { return mesh_device.lock(); }

}  // namespace tt_metal

}  // namespace tt
