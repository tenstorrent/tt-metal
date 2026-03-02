// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fd_kernel.hpp"

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <host_api.hpp>
#include <utility>
#include <variant>

#include "data_types.hpp"
#include "device.hpp"
#include "dispatch.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"
#include "dispatch_core_common.hpp"
#include "dispatch_s.hpp"
#include "hal_types.hpp"
#include "kernel_types.hpp"
#include "prefetch.hpp"
#include "impl/context/context_descriptor.hpp"
// #include "impl/context/metal_context.hpp"
#include "kernels/kernel.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include <impl/debug/dprint_server.hpp>

using namespace tt::tt_metal;

ChipId FDKernel::GetUpstreamDeviceId(const ContextDescriptor& descriptor, ChipId device_id) {
    ChipId mmio_device_id = descriptor.cluster().get_associated_mmio_device(device_id);
    for (auto tunnel : descriptor.cluster().get_tunnels_from_mmio_device(mmio_device_id)) {
        for (int idx = 0; idx < tunnel.size(); idx++) {
            if (tunnel[idx] == device_id) {
                // MMIO device doesn't have an upsream, just return itself
                return (idx == 0) ? device_id : tunnel[idx - 1];
            }
        }
    }
    TT_ASSERT(false, "Could not find upstream device of Device {}", device_id);
    return device_id;
}

ChipId FDKernel::GetDownstreamDeviceId(const ContextDescriptor& descriptor, ChipId device_id, int tunnel) {
    ChipId mmio_device_id = descriptor.cluster().get_associated_mmio_device(device_id);
    auto tunnels = descriptor.cluster().get_tunnels_from_mmio_device(mmio_device_id);
    if (tunnel < -1 || tunnel >= static_cast<int>(tunnels.size())) {
        TT_THROW("Tunnel {} is out of range. {} tunnels exist", tunnel, tunnels.size());
    }

    if (tunnel != -1) {
        // Remove all tunnels except the relevant one which will be at the front
        std::swap(tunnels[0], tunnels[tunnel]);
        tunnels.erase(tunnels.begin() + 1, tunnels.end());
    }

    for (auto tunnel : tunnels) {
        for (int idx = 0; idx < tunnel.size(); idx++) {
            if (tunnel[idx] == device_id) {
                // End of tunnel doesn't have downstream, just return itself
                return (idx == tunnel.size() - 1) ? device_id : tunnel[idx + 1];
            }
        }
    }
    TT_FATAL(false, "Could not find downstream device of Device {}", device_id);
    return device_id;
}

uint32_t FDKernel::GetTunnelStop(const ContextDescriptor& descriptor, ChipId device_id) {
    ChipId mmio_device_id = descriptor.cluster().get_associated_mmio_device(device_id);
    for (auto tunnel : descriptor.cluster().get_tunnels_from_mmio_device(mmio_device_id)) {
        for (uint32_t idx = 0; idx < tunnel.size(); idx++) {
            if (tunnel[idx] == device_id) {
                return idx;
            }
        }
    }
    TT_ASSERT(false, "Could not find tunnel stop of Device {}", device_id);
    return 0;
}

tt::tt_fabric::ControlPlane& FDKernel::get_control_plane_ref() const {
    TT_ASSERT(static_cast<bool>(get_control_plane_), "Control plane accessor not set (required when fabric is used)");
    return get_control_plane_();
}

const DispatchQueryManager& FDKernel::get_dispatch_query_manager_ref() const {
    TT_ASSERT(static_cast<bool>(get_dispatch_query_manager_), "Dispatch query manager accessor not set");
    return get_dispatch_query_manager_();
}

uint32_t FDKernel::get_max_num_eth_cores() const {
    TT_ASSERT(static_cast<bool>(get_max_num_eth_cores_), "Max num eth cores accessor not set");
    return get_max_num_eth_cores_();
}

FDKernel* FDKernel::Generate(
    int node_id,
    ChipId device_id,
    ChipId servicing_device_id,
    uint8_t cq_id,
    noc_selection_t noc_selection,
    DispatchWorkerType type,
    const ContextDescriptor& descriptor,
    dispatch_core_manager& dispatch_core_manager,
    int tunnel_index,
    const GetControlPlaneFn& get_control_plane,
    const GetDispatchQueryManagerFn& get_dispatch_query_manager,
    const GetMaxNumEthCoresFn& get_max_num_eth_cores,
    const GetReadsDispatchCoresFn& get_reads_dispatch_cores) {
    switch (type) {
        case PREFETCH_HD:
            return new PrefetchKernel(
                node_id,
                device_id,
                servicing_device_id,
                cq_id,
                noc_selection,
                true,
                true,
                descriptor,
                dispatch_core_manager,
                get_control_plane,
                get_dispatch_query_manager,
                get_max_num_eth_cores,
                get_reads_dispatch_cores);
        case PREFETCH_H:
            return new PrefetchKernel(
                node_id,
                device_id,
                servicing_device_id,
                cq_id,
                noc_selection,
                true,
                false,
                descriptor,
                dispatch_core_manager,
                get_control_plane,
                get_dispatch_query_manager,
                get_max_num_eth_cores,
                get_reads_dispatch_cores);
        case PREFETCH_D:
            return new PrefetchKernel(
                node_id,
                device_id,
                servicing_device_id,
                cq_id,
                noc_selection,
                false,
                true,
                descriptor,
                dispatch_core_manager,
                get_control_plane,
                get_dispatch_query_manager,
                get_max_num_eth_cores,
                get_reads_dispatch_cores);
        case DISPATCH_HD:
            return new DispatchKernel(
                node_id,
                device_id,
                servicing_device_id,
                cq_id,
                noc_selection,
                true,
                true,
                descriptor,
                dispatch_core_manager,
                get_control_plane,
                get_dispatch_query_manager,
                get_max_num_eth_cores,
                get_reads_dispatch_cores);
        case DISPATCH_H:
            return new DispatchKernel(
                node_id,
                device_id,
                servicing_device_id,
                cq_id,
                noc_selection,
                true,
                false,
                descriptor,
                dispatch_core_manager,
                get_control_plane,
                get_dispatch_query_manager,
                get_max_num_eth_cores,
                get_reads_dispatch_cores);
        case DISPATCH_D:
            return new DispatchKernel(
                node_id,
                device_id,
                servicing_device_id,
                cq_id,
                noc_selection,
                false,
                true,
                descriptor,
                dispatch_core_manager,
                get_control_plane,
                get_dispatch_query_manager,
                get_max_num_eth_cores,
                get_reads_dispatch_cores);
        case DISPATCH_S:
            return new DispatchSKernel(
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
                get_reads_dispatch_cores);
        case FABRIC_MUX:
            return new tt::tt_metal::RelayMux(
                node_id,
                device_id,
                servicing_device_id,
                cq_id,
                noc_selection,
                false,
                tunnel_index,
                descriptor,
                dispatch_core_manager,
                get_control_plane,
                get_reads_dispatch_cores);
        case RETURN_FABRIC_MUX:
            return new tt::tt_metal::RelayMux(
                node_id,
                device_id,
                servicing_device_id,
                cq_id,
                noc_selection,
                true,
                tunnel_index,
                descriptor,
                dispatch_core_manager,
                get_control_plane,
                get_reads_dispatch_cores);
        default: TT_FATAL(false, "Unrecognized dispatch kernel type: {}.", type); return nullptr;
    }
}

uint32_t FDKernel::get_programmable_core_type_index(
    const ContextDescriptor& descriptor, CoreType dispatch_core_type, bool is_active_eth_core) {
    // TODO(#22895): Too many core types. Consolidate programmable_core_type_index with ProgrammableCoreType and
    // CoreType
    uint32_t programmable_core_type_index;
    if (dispatch_core_type == CoreType::WORKER) {
        programmable_core_type_index =
            descriptor.hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    } else if (is_active_eth_core) {
        programmable_core_type_index =
            descriptor.hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
    } else {
        programmable_core_type_index =
            descriptor.hal().get_programmable_core_type_index(HalProgrammableCoreType::IDLE_ETH);
    }

    return programmable_core_type_index;
}

CoreCoord FDKernel::get_virtual_core_coord(
    const ContextDescriptor& descriptor, const tt_cxy_pair& logical_cxy, const CoreType& core_type) {
    return descriptor.cluster().get_virtual_coordinate_from_logical_coordinates(logical_cxy, core_type);
}

KernelHandle FDKernel::configure_kernel_variant(
    const std::string& path,
    const std::vector<uint32_t>& compile_args,
    std::map<std::string, std::string> defines_in,
    bool is_active_eth_core,
    bool send_to_brisc,
    bool force_watcher_no_inline,
    KernelBuildOptLevel opt_level) {
    uint32_t programmable_core_type_index =
        get_programmable_core_type_index(descriptor_, GetCoreType(), is_active_eth_core);

    std::map<std::string, std::string> defines = {
        {"DISPATCH_KERNEL", "1"},
        {"FD_CORE_TYPE", std::to_string(programmable_core_type_index)},
    };
    if (force_watcher_no_inline) {
        defines.insert({"WATCHER_NOINLINE", std::to_string(force_watcher_no_inline)});
    }
    const auto& rt_options = descriptor_.rtoptions();
    if (rt_options.watcher_dispatch_disabled()) {
        defines["FORCE_WATCHER_OFF"] = "1";
    }
    if (!(get_reads_dispatch_cores_ && get_reads_dispatch_cores_(device_->id()))) {
        defines["FORCE_DPRINT_OFF"] = "1";
    }
    defines.insert(defines_in.begin(), defines_in.end());
    if (descriptor_.cluster().is_galaxy_cluster()) {
        // TG specific fabric routing
        // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
        defines["GALAXY_CLUSTER"] = "1";
    }

    if (GetCoreType() == CoreType::WORKER) {
        kernel_handle_ = tt::tt_metal::CreateKernel(
            *program_,
            path,
            logical_core_,
            tt::tt_metal::DataMovementConfig{
                .processor = send_to_brisc ? tt::tt_metal::DataMovementProcessor::RISCV_0
                                           : tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = noc_selection_.non_dispatch_noc,
                .compile_args = compile_args,
                .defines = defines,
                .opt_level = opt_level});
    } else {
        kernel_handle_ = tt::tt_metal::CreateKernel(
            *program_,
            path,
            logical_core_,
            tt::tt_metal::EthernetConfig{
                .eth_mode = is_active_eth_core ? Eth::SENDER : Eth::IDLE,
                .noc = noc_selection_.non_dispatch_noc,
                .compile_args = compile_args,
                .defines = defines,
                .opt_level = opt_level});
    }

    return kernel_handle_;
}

void FDKernel::create_edm_connection_sems(FDKernelEdmConnectionAttributes& attributes) {
    attributes.worker_flow_control_sem = tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
    attributes.worker_buffer_index_sem = tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
    attributes.worker_teardown_sem = tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
}

void FDKernel::SetRuntimeArgs() {
    TT_ASSERT(program_ != nullptr, "Program must be set before setting runtime args");
    if (not runtime_args_.empty()) {
        tt_metal::SetRuntimeArgs(*program_, kernel_handle_, logical_core_, runtime_args_);
    }
}
