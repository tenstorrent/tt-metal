// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "dispatch_s.hpp"

#include <span>
#include <tt_metal.hpp>
#include "impl/buffers/semaphore.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include "dispatch/command_queue_common.hpp"
#include "device.hpp"
#include "dispatch.hpp"
#include "fd_kernel.hpp"
#include "dispatch_core_common.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "hal_types.hpp"
#include "prefetch.hpp"
#include "context/context_descriptor.hpp"
#include "debug/inspector/inspector.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "device/device_manager.hpp"
#include <dispatch/dispatch_query_manager.hpp>
#include <dispatch/dispatch_mem_map.hpp>
#include "impl/dispatch/dispatch_engine_cores.hpp"
#include "hostdev/realtime_profiler_msgs.h"

#include "impl/context/metal_context.hpp"
#include "impl/debug/dprint_server.hpp"
#include "impl/debug/debug_helpers.hpp"
#include "hostdev/device_print_structures.h"
#include "hostdevcommon/dispatch_telemetry_types.hpp"
#include "hostdevcommon/dprint_common.h"

using namespace tt::tt_metal;

namespace {

// Zeros selected realtime_profiler_msg_t fields on dispatch_s L1 (REALTIME_PROFILER_MSG carve); order matches
// former cq_dispatch_subordinate kernel_main init (signalling fields before FIFO, then timestamp .id words).
void zero_dispatch_s_realtime_profiler_msg_fields(
    IDevice* device, const CoreCoord& logical_core, tt::CoreType core_type, const Hal& hal, uint32_t msg_base_l1_addr) {
    const auto& factory = hal.get_realtime_profiler_msgs_factory(HalProgrammableCoreType::TENSIX);
    // WriteToDeviceL1(..., vector<uint32_t>&) requires a mutable vector (non-const ref overload).
    std::vector<uint32_t> zero_word = {0u};
    const uint32_t ts_id_byte_off = offsetof(realtime_profiler_timestamp_t, id);

    auto write_u32 = [&](uint32_t addr) {
        tt::tt_metal::detail::WriteToDeviceL1(device, logical_core, addr, zero_word, core_type);
    };

    const uint32_t base = msg_base_l1_addr;
    write_u32(
        base + factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                   realtime_profiler_msgs::realtime_profiler_msg_t::Field::realtime_profiler_core_noc_xy));
    write_u32(
        base + factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                   realtime_profiler_msgs::realtime_profiler_msg_t::Field::realtime_profiler_remote_state_addr));
    write_u32(
        base + factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                   realtime_profiler_msgs::realtime_profiler_msg_t::Field::realtime_profiler_state));
    write_u32(
        base + factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                   realtime_profiler_msgs::realtime_profiler_msg_t::Field::program_id_fifo_start));
    write_u32(
        base + factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
                   realtime_profiler_msgs::realtime_profiler_msg_t::Field::program_id_fifo_end));

    const uint32_t ksa = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
        realtime_profiler_msgs::realtime_profiler_msg_t::Field::kernel_start_a);
    const uint32_t kea = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
        realtime_profiler_msgs::realtime_profiler_msg_t::Field::kernel_end_a);
    const uint32_t ksb = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
        realtime_profiler_msgs::realtime_profiler_msg_t::Field::kernel_start_b);
    const uint32_t keb = factory.offset_of<realtime_profiler_msgs::realtime_profiler_msg_t>(
        realtime_profiler_msgs::realtime_profiler_msg_t::Field::kernel_end_b);
    write_u32(base + ksa + ts_id_byte_off);
    write_u32(base + kea + ts_id_byte_off);
    write_u32(base + ksb + ts_id_byte_off);
    write_u32(base + keb + ts_id_byte_off);
}

}  // namespace

DispatchSKernel::DispatchSKernel(
    int node_id,
    ChipId device_id,
    ChipId servicing_device_id,
    uint8_t cq_id,
    noc_selection_t noc_selection,
    const ContextDescriptor& descriptor,
    dispatch_core_manager& dispatch_core_manager,
    const GetControlPlaneFn& get_control_plane,
    const GetDispatchQueryManagerFn& get_dispatch_query_manager,
    const GetMaxNumEthCoresFn& get_max_num_eth_cores,
    const GetReadsDispatchCoresFn& get_reads_dispatch_cores) :
    FDKernel(
        node_id,
        device_id,
        servicing_device_id,
        cq_id,
        noc_selection,
        descriptor,
        dispatch_core_manager,
        get_control_plane,
        get_dispatch_query_manager,
        get_max_num_eth_cores,
        get_reads_dispatch_cores) {
    uint16_t channel = descriptor.cluster().get_assigned_channel_for_device(device_id);
    this->logical_core_ = dispatch_core_manager.dispatcher_s_core(device_id, channel, cq_id_);
    this->kernel_type_ = FDKernelType::DISPATCH;
    // Log dispatch_s core info based on virtual core to inspector
    Inspector::set_dispatch_s_core_info(this->GetVirtualCore(), DISPATCH_S, cq_id, device_id, servicing_device_id);
}

void DispatchSKernel::GenerateStaticConfigs() {
    const auto& my_dispatch_constants = get_dispatch_mem_map();

    std::set<uint8_t> upstream_cq_ids;
    std::set<uint8_t> downstream_cq_ids;
    for (const auto* kernel : upstream_kernels_) {
        TT_FATAL(dynamic_cast<const PrefetchKernel*>(kernel) != nullptr, "DISPATCH_S upstream must be PREFETCH");
        upstream_cq_ids.insert(kernel->GetCQId());
    }
    for (const auto* kernel : downstream_kernels_) {
        TT_FATAL(dynamic_cast<const DispatchKernel*>(kernel) != nullptr, "DISPATCH_S downstream must be DISPATCH");
        downstream_cq_ids.insert(kernel->GetCQId());
    }
    TT_FATAL(
        upstream_cq_ids.size() == upstream_kernels_.size() && downstream_cq_ids.size() == downstream_kernels_.size(),
        "DISPATCH_S requires one peer of each type per CQ");
    TT_FATAL(upstream_cq_ids == downstream_cq_ids, "DISPATCH_S must have matching PREFETCH and DISPATCH peers by CQ");
    served_cq_ids_.assign(upstream_cq_ids.begin(), upstream_cq_ids.end());

    const bool serves_one_cq = served_cq_ids_.size() == 1;
    const bool serves_cq0_and_cq1 = served_cq_ids_.size() == 2 && served_cq_ids_[0] == 0 && served_cq_ids_[1] == 1;
    TT_FATAL(serves_one_cq || serves_cq0_and_cq1, "DISPATCH_S must serve exactly one CQ, or CQ0 and CQ1 when shared");
    if (!serves_one_cq) {
        for (const auto* kernel : upstream_kernels_) {
            TT_FATAL(kernel->GetLogicalCore() == logical_core_, "Shared DISPATCH_S peers must be colocated");
        }
        for (const auto* kernel : downstream_kernels_) {
            TT_FATAL(kernel->GetLogicalCore() == logical_core_, "Shared DISPATCH_S peers must be colocated");
        }
        const uint16_t channel = descriptor_.cluster().get_assigned_channel_for_device(device_id_);
        for (const uint8_t served_cq_id : served_cq_ids_) {
            TT_FATAL(
                dispatch_core_manager_.dispatcher_s_core(device_id_, channel, served_cq_id) == logical_core_,
                "All CQs served by one DISPATCH_S must resolve to the same core");
        }
    }

    const bool dispatch_s_enabled = get_dispatch_query_manager_ref().dispatch_s_enabled();
    // dispatch_s shares the core with dispatch_d (Tensix WORKER or Quasar DE): start its CB past dispatch_d's.
    const uint32_t cb_base_offset =
        (GetCoreType() == CoreType::WORKER || GetCoreType() == CoreType::DISPATCH)
            ? (1 << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE) * my_dispatch_constants.dispatch_buffer_pages()
            : 0;
    for (const uint8_t served_cq_id : served_cq_ids_) {
        auto& channel_config = static_config_.channels[served_cq_id];
        channel_config.cb_base =
            dispatch_s_enabled ? my_dispatch_constants.dispatch_buffer_base(served_cq_id) + cb_base_offset : 0xff;
        channel_config.my_dispatch_cb_sem_id = CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
        channel_config.dispatch_d_shutdown_sem_id = CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
        // cq_dispatch.cpp derives dispatch_s_enabled from a non-zero shutdown sem id.
        TT_FATAL(
            channel_config.dispatch_d_shutdown_sem_id.value() != 0,
            "DISPATCH_S shutdown semaphore for CQ {} must be non-zero",
            served_cq_id);
        channel_config.dispatch_s_sync_sem_base_addr = my_dispatch_constants.get_device_command_queue_addr(
            CommandQueueDeviceAddrType::DISPATCH_S_SYNC_SEM, served_cq_id);
        channel_config.completion_counter_offset = my_dispatch_constants.get_completion_counter_offset(served_cq_id);
        channel_config.realtime_profiler_msg_addr = my_dispatch_constants.get_device_command_queue_addr(
            CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG, served_cq_id);
        channel_config.dispatch_telemetry_addr = my_dispatch_constants.get_device_command_queue_addr(
            CommandQueueDeviceAddrType::DISPATCH_TELEMETRY, served_cq_id);
        channel_config.dispatch_telemetry_control_addr = my_dispatch_constants.get_device_command_queue_addr(
            CommandQueueDeviceAddrType::DISPATCH_TELEMETRY_CONTROL, served_cq_id);
    }

    static_config_.cb_log_page_size = DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE;
    static_config_.cb_size = my_dispatch_constants.dispatch_s_buffer_size();
    static_config_.mcast_go_signal_addr =
        descriptor_.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::GO_MSG);
    static_config_.unicast_go_signal_addr =
        (descriptor_.hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH) != -1)
            ? descriptor_.hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::GO_MSG)
            : 0;
    static_config_.distributed_dispatcher = get_dispatch_query_manager_ref().distributed_dispatcher();
    static_config_.first_stream_used = my_dispatch_constants.get_dispatch_stream_index(0);
    static_config_.max_num_worker_sems = DispatchSettings::DISPATCH_MESSAGE_ENTRIES;
    static_config_.max_num_go_signal_noc_data_entries = DispatchSettings::DISPATCH_GO_SIGNAL_NOC_DATA_ENTRIES;
    static_config_.dispatch_telemetry_disabled = descriptor_.rtoptions().get_dispatch_telemetry_disabled();

    // Configuration for DEVICE_PRINT dispatch.
    static_config_.device_print_dispatch_enabled = 0;
    // With multiple CQs there is one dispatch_s per CQ, but they all read the same per-core
    // DEVICE_PRINT L1 buffers. Only enable the DRAM-aggregation work on cq_id 0 so the buffers
    // aren't drained twice (which would race the host's rpos updates and reorder/drop messages).
    if (cq_id_ == 0 && get_dispatch_query_manager_ref().dispatch_s_enabled() &&
        descriptor_.metal_context().dprint_server() && device_->arch() != tt::ARCH::QUASAR) {
        auto print_cores = descriptor_.metal_context().dprint_server()->get_print_cores(device_->id());
        if (!print_cores.empty()) {
            const auto& hal = descriptor_.hal();
            const uint32_t num_print_cores = static_cast<uint32_t>(print_cores.size());

            // Overlay constraint: noc_locations and l1_cache share the same L1 bytes (the kernel
            // copies noc_locations into LDM at init then reuses the L1 region as l1_cache). The
            // initial noc_locations data must fit within the overlaid l1_cache region.
            const uint32_t noc_locations_size =
                static_cast<uint32_t>(sizeof(device_print_dispatch::NocLocationInputInfo)) * num_print_cores;
            const uint32_t l1_cache_size = my_dispatch_constants.dispatch_s_device_print_l1_cache_size();

            // Check if there is enough space in the buffer
            if (noc_locations_size > l1_cache_size) {
                log_warning(
                    tt::LogMetal,
                    "DPRINT dispatch_s DRAM aggregation disabled on device {}: l1_cache ({} bytes) is too "
                    "small to hold noc_locations for {} print cores ({} bytes). Falling back to per-core L1 "
                    "polling; raise TT_METAL_DEVICE_PRINT_DISPATCH_L1_CACHE_BYTES to re-enable.",
                    device_->id(),
                    l1_cache_size,
                    num_print_cores,
                    noc_locations_size);
            } else {
                const uint32_t dram_alignment = hal.get_alignment(HalMemType::DRAM);
                const uint64_t dram_base = hal.get_dev_addr(HalDramMemAddrType::DEVICE_PRINT_DISPATCH);
                const uint32_t dram_total_size = hal.get_dev_size(HalDramMemAddrType::DEVICE_PRINT_DISPATCH);

                // dram_view 0's preferred worker NOC coords for the dispatch_s NOC.
                auto dram_logical = device_->logical_core_from_dram_channel(0);
                auto dram_virtual = device_->virtual_core_from_logical_core(dram_logical, CoreType::DRAM);
                auto dram_noc = device_->virtual_noc0_coordinate(noc_selection_.non_dispatch_noc, dram_virtual);

                const auto& rtoptions = descriptor_.rtoptions();
                const uint64_t clock_mhz = static_cast<uint64_t>(device_->get_clock_rate_mhz());

                static_config_.device_print_dispatch_enabled = 1;
                static_config_.device_print_noc_locations_addr =
                    my_dispatch_constants.device_print_dispatch_noc_locations_addr(cq_id_);
                static_config_.device_print_noc_locations_count = num_print_cores;
                static_config_.device_print_l1_cache_addr =
                    my_dispatch_constants.device_print_dispatch_l1_cache_addr(cq_id_);
                static_config_.device_print_l1_cache_size = l1_cache_size;
                static_config_.device_print_dram_x = dram_noc.x;
                static_config_.device_print_dram_y = dram_noc.y;
                static_config_.device_print_dram_rw_ptrs = dram_base;
                static_config_.device_print_dram_buf_addr = dram_base + dram_alignment;
                static_config_.device_print_dram_buf_size = dram_total_size - dram_alignment;
                static_config_.device_print_cycles_for_stall =
                    clock_mhz * rtoptions.get_device_print_dispatch_stall_us();
                static_config_.device_print_cycles_for_full = clock_mhz * rtoptions.get_device_print_dispatch_full_us();
            }
        }
    }
}

void DispatchSKernel::GenerateDependentConfigs() {
    for (const uint8_t served_cq_id : served_cq_ids_) {
        auto& channel = dependent_config_.channels[served_cq_id];
        PrefetchKernel* prefetch_kernel = nullptr;
        DispatchKernel* dispatch_kernel = nullptr;
        for (auto* upstream : upstream_kernels_) {
            if (upstream->GetCQId() == served_cq_id) {
                TT_ASSERT(prefetch_kernel == nullptr);
                prefetch_kernel = dynamic_cast<PrefetchKernel*>(upstream);
            }
        }
        for (auto* downstream : downstream_kernels_) {
            if (downstream->GetCQId() == served_cq_id) {
                TT_ASSERT(dispatch_kernel == nullptr);
                dispatch_kernel = dynamic_cast<DispatchKernel*>(downstream);
            }
        }
        TT_ASSERT(prefetch_kernel != nullptr && dispatch_kernel != nullptr);
        channel.upstream_logical_core = prefetch_kernel->GetLogicalCore();
        channel.upstream_dispatch_cb_sem_id = prefetch_kernel->GetStaticConfig().my_dispatch_s_cb_sem_id;
        channel.downstream_logical_core = dispatch_kernel->GetLogicalCore();
    }
}

void DispatchSKernel::CreateKernel() {
    // Issue #19729: Workaround to allow TT-Mesh Workload dispatch to target active ethernet cores.
    // num_virtual_active_eth_cores is set if the user application requested virtualizing the
    // number of ethernet cores across devices (to essentially fake uniformity). This value is the
    // max number of ethernet cores across all chips in the opened cluster.
    // num_physical_ethernet_cores is the number of actual available ethernet cores on the current device.
    // virtualize_num_eth_cores is set if the number of virtual cores is greater than the number of actual
    // ethernet cores in the chip.
    uint32_t num_virtual_active_eth_cores = get_max_num_eth_cores();
    uint32_t num_physical_active_eth_cores =
        get_control_plane_ref().get_active_ethernet_cores(device_->id(), /*skip_reserved_tunnel_cores*/ true).size();
    bool virtualize_num_eth_cores = num_virtual_active_eth_cores > num_physical_active_eth_cores;

    const auto& compute_grid_size = device_->compute_with_storage_grid_size();
    CoreRange device_worker_cores = CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1});
    auto virtual_start = device_->virtual_core_from_logical_core(device_worker_cores.start_coord, CoreType::WORKER);
    auto virtual_end = device_->virtual_core_from_logical_core(device_worker_cores.end_coord, CoreType::WORKER);
    auto virtual_core_range = CoreRange(virtual_start, virtual_end);

    auto my_virtual_core = device_->virtual_core_from_logical_core(logical_core_, GetCoreType());
    auto downstream_s_virtual_core = device_->virtual_core_from_logical_core(UNUSED_LOGICAL_CORE, GetCoreType());

    auto my_virtual_noc_coords = device_->virtual_noc0_coordinate(noc_selection_.non_dispatch_noc, my_virtual_core);
    auto downstream_s_virtual_noc_coords =
        device_->virtual_noc0_coordinate(noc_selection_.downstream_noc, downstream_s_virtual_core);

    std::map<std::string, std::string> defines = {
        {"NUM_DISPATCH_S_CHANNELS", std::to_string(served_cq_ids_.size())},
        {"MY_NOC_X", std::to_string(my_virtual_noc_coords.x)},
        {"MY_NOC_Y", std::to_string(my_virtual_noc_coords.y)},
        {"UPSTREAM_NOC_INDEX", std::to_string(noc_selection_.upstream_noc)},                  // Unused, remove later
        {"DOWNSTREAM_SUBORDINATE_NOC_X", std::to_string(downstream_s_virtual_noc_coords.x)},  // Unused, remove later
        {"DOWNSTREAM_SUBORDINATE_NOC_Y", std::to_string(downstream_s_virtual_noc_coords.y)},  // Unused, remove later
        {"CB_LOG_PAGE_SIZE", std::to_string(static_config_.cb_log_page_size.value())},
        {"CB_SIZE", std::to_string(static_config_.cb_size.value())},
        {"MCAST_GO_SIGNAL_ADDR", std::to_string(static_config_.mcast_go_signal_addr.value())},
        {"UNICAST_GO_SIGNAL_ADDR", std::to_string(static_config_.unicast_go_signal_addr.value())},
        {"DISTRIBUTED_DISPATCHER", std::to_string(static_config_.distributed_dispatcher.value())},
        {"FIRST_STREAM_USED", std::to_string(static_config_.first_stream_used.value())},
        {"MAX_NUM_WORKER_SEMS", std::to_string(static_config_.max_num_worker_sems.value())},
        {"MAX_NUM_GO_SIGNAL_NOC_DATA_ENTRIES",
         std::to_string(static_config_.max_num_go_signal_noc_data_entries.value())},
        {"VIRTUALIZE_UNICAST_CORES", std::to_string(virtualize_num_eth_cores)},
        {"NUM_VIRTUAL_UNICAST_CORES", std::to_string(num_virtual_active_eth_cores)},
        {"NUM_PHYSICAL_UNICAST_CORES", std::to_string(num_physical_active_eth_cores)},
        {"WORKER_MCAST_GRID",
         std::to_string(device_->get_noc_multicast_encoding(noc_selection_.downstream_noc, virtual_core_range))},
        {"NUM_WORKER_CORES_TO_MCAST", std::to_string(device_worker_cores.size())},
        {"DISPATCH_TELEMETRY_DISABLED", std::to_string(static_config_.dispatch_telemetry_disabled.value_or(false))},
        {"DEVICE_PRINT_DISPATCH_ENABLED", std::to_string(static_config_.device_print_dispatch_enabled.value_or(0))},
        // For each per-device dispatch_s build, MaxNocLocations equals the actual print-core count
        // for that device — passed as a compile-time #define so DevicePrintDispatch<>'s LDM arrays
        // (rw_noc_addresses, cache_buffer_offsets, cache_buffer_sizes, noc_locations_to_process)
        // are sized to actual usage instead of device_print_dispatch::DEFAULT_MAX_NOC_LOCATIONS.
        {"DEVICE_PRINT_MAX_NOC_LOCATIONS", std::to_string(static_config_.device_print_noc_locations_count.value_or(0))},
        {"DEVICE_PRINT_NOC_LOCATIONS_ADDR", std::to_string(static_config_.device_print_noc_locations_addr.value_or(0))},
        {"DEVICE_PRINT_NOC_LOCATIONS_COUNT",
         std::to_string(static_config_.device_print_noc_locations_count.value_or(0))},
        {"DEVICE_PRINT_L1_CACHE_ADDR", std::to_string(static_config_.device_print_l1_cache_addr.value_or(0))},
        {"DEVICE_PRINT_L1_CACHE_SIZE", std::to_string(static_config_.device_print_l1_cache_size.value_or(0))},
        {"DEVICE_PRINT_DRAM_X", std::to_string(static_config_.device_print_dram_x.value_or(0))},
        {"DEVICE_PRINT_DRAM_Y", std::to_string(static_config_.device_print_dram_y.value_or(0))},
        {"DEVICE_PRINT_DRAM_RW_PTRS", std::to_string(static_config_.device_print_dram_rw_ptrs.value_or(0)) + "ULL"},
        {"DEVICE_PRINT_DRAM_BUF_ADDR", std::to_string(static_config_.device_print_dram_buf_addr.value_or(0)) + "ULL"},
        {"DEVICE_PRINT_DRAM_BUF_SIZE", std::to_string(static_config_.device_print_dram_buf_size.value_or(0))},
        {"DEVICE_PRINT_CYCLES_FOR_STALL",
         std::to_string(static_config_.device_print_cycles_for_stall.value_or(0)) + "ULL"},
        {"DEVICE_PRINT_CYCLES_FOR_FULL",
         std::to_string(static_config_.device_print_cycles_for_full.value_or(0)) + "ULL"},
    };
    for (size_t channel_index = 0; channel_index < served_cq_ids_.size(); ++channel_index) {
        const uint8_t served_cq_id = served_cq_ids_[channel_index];
        const auto& channel = static_config_.channels.at(served_cq_id);
        const auto& dependent_channel = dependent_config_.channels.at(served_cq_id);
        const auto upstream_virtual_core =
            device_->virtual_core_from_logical_core(dependent_channel.upstream_logical_core.value(), GetCoreType());
        const auto downstream_virtual_core =
            device_->virtual_core_from_logical_core(dependent_channel.downstream_logical_core.value(), GetCoreType());
        const auto upstream_virtual_noc_coords =
            device_->virtual_noc0_coordinate(noc_selection_.upstream_noc, upstream_virtual_core);
        const auto downstream_virtual_noc_coords =
            device_->virtual_noc0_coordinate(noc_selection_.downstream_noc, downstream_virtual_core);
        const std::string suffix = served_cq_ids_.size() == 1 ? "" : "_" + std::to_string(channel_index);
        defines["CB_BASE" + suffix] = std::to_string(channel.cb_base.value());
        defines["MY_DISPATCH_CB_SEM_ID" + suffix] = std::to_string(channel.my_dispatch_cb_sem_id.value());
        defines["UPSTREAM_DISPATCH_CB_SEM_ID" + suffix] =
            std::to_string(dependent_channel.upstream_dispatch_cb_sem_id.value());
        defines["DISPATCH_D_SHUTDOWN_SEM_ID" + suffix] = std::to_string(channel.dispatch_d_shutdown_sem_id.value());
        defines["DISPATCH_S_SYNC_SEM_BASE_ADDR" + suffix] =
            std::to_string(channel.dispatch_s_sync_sem_base_addr.value());
        defines["COMPLETION_COUNTER_OFFSET" + suffix] = std::to_string(channel.completion_counter_offset.value());
        defines["REALTIME_PROFILER_MSG_ADDR" + suffix] = std::to_string(channel.realtime_profiler_msg_addr.value());
        defines["DISPATCH_TELEMETRY_ADDR" + suffix] = std::to_string(channel.dispatch_telemetry_addr.value());
        defines["DISPATCH_TELEMETRY_CONTROL_ADDR" + suffix] =
            std::to_string(channel.dispatch_telemetry_control_addr.value());
        defines["UPSTREAM_NOC_X" + suffix] = std::to_string(upstream_virtual_noc_coords.x);
        defines["UPSTREAM_NOC_Y" + suffix] = std::to_string(upstream_virtual_noc_coords.y);
        defines["DOWNSTREAM_NOC_X" + suffix] = std::to_string(downstream_virtual_noc_coords.x);
        defines["DOWNSTREAM_NOC_Y" + suffix] = std::to_string(downstream_virtual_noc_coords.y);
    }
    num_threads_ = served_cq_ids_.size();
    configure_kernel_variant(
        dispatch_kernel_file_names[DISPATCH_S], {}, defines, tt::tt_metal::KernelBuildOptLevel::Os);

    if (GetCoreType() == CoreType::WORKER && device_->arch() != tt::ARCH::QUASAR) {
        const auto& channel = static_config_.channels.at(served_cq_ids_.front());
        const std::string compute_kernel_path = "tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate_compute.cpp";
        std::map<std::string, std::string> compute_defines = {
            {"FIRST_STREAM_INDEX", std::to_string(static_config_.first_stream_used.value())},
            {"NUM_STREAMS_TO_MONITOR", std::to_string(static_config_.max_num_worker_sems.value())},
            {"REALTIME_PROFILER_MSG_ADDR", std::to_string(channel.realtime_profiler_msg_addr.value())},
            {"DISPATCH_TELEMETRY_ADDR", std::to_string(channel.dispatch_telemetry_addr.value())},
            {"DISPATCH_TELEMETRY_DISABLED", std::to_string(static_config_.dispatch_telemetry_disabled.value_or(false))},
            {"TOTAL_SUB_DEVICES", std::to_string(static_config_.max_num_worker_sems.value())},
            {"DISPATCH_TELEMETRY_CONTROL_ADDR", std::to_string(channel.dispatch_telemetry_control_addr.value())},
        };
        tt::tt_metal::ComputeConfig compute_config;
        compute_config.defines = compute_defines;
        tt::tt_metal::CreateKernel(*program_, compute_kernel_path, logical_core_, compute_config);
    }
}

uint32_t DispatchSKernel::GetMyDispatchCbSemId(uint8_t cq_id) const {
    return static_config_.channels.at(cq_id).my_dispatch_cb_sem_id.value();
}

uint32_t DispatchSKernel::GetDispatchDShutdownSemId(uint8_t cq_id) const {
    return static_config_.channels.at(cq_id).dispatch_d_shutdown_sem_id.value();
}

void DispatchSKernel::ConfigureCore() {
    for (const uint8_t served_cq_id : served_cq_ids_) {
        const auto& channel = static_config_.channels.at(served_cq_id);
        if (get_dispatch_query_manager_ref().dispatch_s_enabled()) {
            zero_dispatch_s_realtime_profiler_msg_fields(
                device_, logical_core_, GetCoreType(), descriptor_.hal(), channel.realtime_profiler_msg_addr.value());

            dispatch_telemetry_types::DispatchTelemetryControl zero_dispatch_telemetry_control{};
            detail::WriteToDeviceL1(
                device_,
                logical_core_,
                channel.dispatch_telemetry_control_addr.value(),
                std::span<const uint8_t>(
                    reinterpret_cast<const uint8_t*>(&zero_dispatch_telemetry_control),
                    sizeof(zero_dispatch_telemetry_control)),
                GetCoreType());
        }

        if (get_dispatch_query_manager_ref().distributed_dispatcher()) {
            TT_ASSERT(static_config_.dispatch_telemetry_disabled.has_value());
            dispatch_telemetry_types::DispatchCoreTelemetry zero_dispatch_telemetry{};
            if (static_config_.dispatch_telemetry_disabled.value()) {
                zero_dispatch_telemetry.signature = dispatch_telemetry_types::INVALID_TELEMETRY_SIGNATURE;
            }
            detail::WriteToDeviceL1(
                device_,
                logical_core_,
                channel.dispatch_telemetry_addr.value(),
                std::span<const uint8_t>(
                    reinterpret_cast<const uint8_t*>(&zero_dispatch_telemetry), sizeof(zero_dispatch_telemetry)),
                GetCoreType());

            std::vector<uint32_t> zero = {0x0};
            const auto& my_dispatch_constants = get_dispatch_mem_map();
            for (uint32_t i = 0; i < DispatchSettings::DISPATCH_MESSAGE_ENTRIES; i++) {
                const uint32_t dispatch_s_sync_sem_addr =
                    channel.dispatch_s_sync_sem_base_addr.value() + my_dispatch_constants.get_sync_offset(i);
                detail::WriteToDeviceL1(device_, logical_core_, dispatch_s_sync_sem_addr, zero, GetCoreType());
            }
        }
    }

    // Configuration for DEVICE_PRINT dispatch if dispatch_s is enabled.
    if (static_config_.device_print_dispatch_enabled.value_or(0) == 0) {
        return;
    }

    // Build NocLocationInputInfo[] for every print core and write to dispatch_s L1.
    auto print_cores = descriptor_.metal_context().dprint_server()->get_print_cores(device_->id());
    if (print_cores.empty()) {
        return;
    }

    auto& cluster = descriptor_.cluster();
    std::vector<device_print_dispatch::NocLocationInputInfo> entries;
    entries.reserve(print_cores.size());

    for (const auto& core_desc : print_cores) {
        auto virtual_core =
            cluster.get_virtual_coordinate_from_logical_coordinates(device_->id(), core_desc.coord, core_desc.type);
        for (const auto& buffer_info :
             descriptor_.metal_context().dprint_server()->get_core_buffers(device_->id(), core_desc)) {
            device_print_dispatch::NocLocationInputInfo entry{};
            entry.x = virtual_core.x;
            entry.y = virtual_core.y;
            entry.rw_ptr_addr = buffer_info.get_read_write_pointer_address();
            entry.buf_offset = buffer_info.buffer_offset;
            entry.buf_size = buffer_info.buffer_size;
            entries.push_back(entry);
        }
    }

    // Write the packed array bitwise to L1. Each entry is 12 bytes.
    const uint32_t l1_addr = static_config_.device_print_noc_locations_addr.value();
    const size_t total_bytes = entries.size() * sizeof(device_print_dispatch::NocLocationInputInfo);
    detail::WriteToDeviceL1(
        device_,
        logical_core_,
        l1_addr,
        std::span(reinterpret_cast<const std::uint8_t*>(entries.data()), total_bytes),
        GetCoreType());
}
