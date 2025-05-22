// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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
#include "dispatch/kernel_config/fabric_router_vc.hpp"
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

// Helper function to get upstream device in the tunnel from current device, not valid for mmio
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

// Same thing for downstream, is ambiuous for mmio device though if it drives more than one tunnel
chip_id_t FDKernel::GetDownstreamDeviceId(chip_id_t device_id) {
    chip_id_t mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    for (auto tunnel :
         tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id)) {
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

// Helper function to get the tunnel stop of current device
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
    DispatchWorkerType type) {
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
        case FABRIC_ROUTER_VC: return new tt::tt_metal::FabricRouterVC(node_id, device_id, servicing_device_id, cq_id);
        default: TT_FATAL(false, "Unrecognized dispatch kernel type: {}.", type); return nullptr;
    }
}

void FDKernel::configure_kernel_variant(
    const string& path,
    const std::vector<uint32_t>& compile_args,
    std::map<string, string> defines_in,
    bool is_active_eth_core,
    bool send_to_brisc,
    bool force_watcher_no_inline,
    KernelBuildOptLevel opt_level) {
    // TODO: just pass in the programmable index
    uint32_t programmable_core_type_index =
        (GetCoreType() == CoreType::WORKER)
            ? MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)
        : is_active_eth_core
            ? MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH)
            : MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::IDLE_ETH);

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
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_fabric_config() != FabricConfig::FABRIC_2D) {
        defines["FVC_MODE_PULL"] = "1";
    }
    if (!DPrintServerReadsDispatchCores(device_->id())) {
        defines["FORCE_DPRINT_OFF"] = "1";
    }
    defines.insert(defines_in.begin(), defines_in.end());

    if (GetCoreType() == CoreType::WORKER) {
        tt::tt_metal::CreateKernel(
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
        tt::tt_metal::CreateKernel(
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
