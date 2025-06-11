// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fd_kernel.hpp"

#include <host_api.hpp>
#include <utility>
#include <variant>

#include "data_types.hpp"
#include "demux.hpp"
#include "device.hpp"
#include "dispatch.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"
#include "dispatch_core_common.hpp"
#include "dispatch_s.hpp"
#include "dprint_server.hpp"
#include "eth_router.hpp"
#include "eth_tunneler.hpp"
#include "fabric_types.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "kernel_types.hpp"
#include "mux.hpp"
#include "prefetch.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/tt_core_coordinates.h>

using namespace tt::tt_metal;

chip_id_t FDKernel::GetUpstreamDeviceId(chip_id_t device_id) {
    chip_id_t mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    for (auto tunnel :
         tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id)) {
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

chip_id_t FDKernel::GetDownstreamDeviceId(chip_id_t device_id, int tunnel) {
    chip_id_t mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    auto tunnels = tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
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
    TT_ASSERT(false, "Could not find downstream device of Device {}", device_id);
    return device_id;
}

uint32_t FDKernel::GetTunnelStop(chip_id_t device_id) {
    chip_id_t mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    for (auto tunnel :
         tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id)) {
        for (uint32_t idx = 0; idx < tunnel.size(); idx++) {
            if (tunnel[idx] == device_id) {
                return idx;
            }
        }
    }
    TT_ASSERT(false, "Could not find tunnel stop of Device {}", device_id);
    return 0;
}

FDKernel* FDKernel::Generate(
    int node_id,
    chip_id_t device_id,
    chip_id_t servicing_device_id,
    uint8_t cq_id,
    noc_selection_t noc_selection,
    DispatchWorkerType type,
    int tunnel_index) {
    switch (type) {
        case PREFETCH_HD:
            return new PrefetchKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection, true, true);
        case PREFETCH_H:
            return new PrefetchKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection, true, false);
        case PREFETCH_D:
            return new PrefetchKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection, false, true);
        case DISPATCH_HD:
            return new DispatchKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection, true, true);
        case DISPATCH_H:
            return new DispatchKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection, true, false);
        case DISPATCH_D:
            return new DispatchKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection, false, true);
        case DISPATCH_S: return new DispatchSKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection);
        case MUX_D: return new MuxKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection);
        case DEMUX: return new DemuxKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection);
        case US_TUNNELER_REMOTE:
            return new EthTunnelerKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection, true);
        case US_TUNNELER_LOCAL:
            return new EthTunnelerKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection, false);
        case PACKET_ROUTER_MUX:
            return new EthRouterKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection, true);
        case PACKET_ROUTER_DEMUX:
            return new EthRouterKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection, false);
        case FABRIC_MUX:
            return new tt::tt_metal::RelayMux(
                node_id, device_id, servicing_device_id, cq_id, noc_selection, false, tunnel_index);
        case RETURN_FABRIC_MUX:
            return new tt::tt_metal::RelayMux(
                node_id, device_id, servicing_device_id, cq_id, noc_selection, true, tunnel_index);
        default: TT_FATAL(false, "Unrecognized dispatch kernel type: {}.", type); return nullptr;
    }
}

uint32_t FDKernel::get_programmable_core_type_index(CoreType dispatch_core_type, bool is_active_eth_core) {
    // TODO(#22895): Too many core types. Consolidate programmable_core_type_index with ProgrammableCoreType and
    // CoreType
    uint32_t programmable_core_type_index =
        (dispatch_core_type == CoreType::WORKER)
            ? MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)
        : is_active_eth_core
            ? MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH)
            : MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::IDLE_ETH);

    return programmable_core_type_index;
}

CoreCoord FDKernel::get_virtual_core_coord(const tt_cxy_pair& logical_cxy, const CoreType& core_type) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    return cluster.get_virtual_coordinate_from_logical_coordinates(logical_cxy, core_type);
}

KernelHandle FDKernel::configure_kernel_variant(
    const string& path,
    const std::vector<uint32_t>& compile_args,
    std::map<string, string> defines_in,
    bool is_active_eth_core,
    bool send_to_brisc,
    bool force_watcher_no_inline,
    KernelBuildOptLevel opt_level) {
    uint32_t programmable_core_type_index = get_programmable_core_type_index(GetCoreType(), is_active_eth_core);

    std::map<string, string> defines = {
        {"DISPATCH_KERNEL", "1"},
        {"FD_CORE_TYPE", std::to_string(programmable_core_type_index)},
    };
    if (force_watcher_no_inline) {
        defines.insert({"WATCHER_NOINLINE", std::to_string(force_watcher_no_inline)});
    }
    auto& rt_options = tt::tt_metal::MetalContext::instance().rtoptions();
    if (rt_options.watcher_dispatch_disabled()) {
        defines["FORCE_WATCHER_OFF"] = "1";
    }
    if (!DPrintServerReadsDispatchCores(device_->id())) {
        defines["FORCE_DPRINT_OFF"] = "1";
    }
    defines.insert(defines_in.begin(), defines_in.end());

    if (GetCoreType() == CoreType::WORKER) {
        return tt::tt_metal::CreateKernel(
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
        return tt::tt_metal::CreateKernel(
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
}

void FDKernel::create_edm_connection_sems(FDKernelEdmConnectionAttributes& attributes) {
    attributes.worker_flow_control_sem = tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
    attributes.worker_buffer_index_sem = tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
    attributes.worker_teardown_sem = tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
}
