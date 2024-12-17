// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fd_kernel.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "impl/debug/dprint_server.hpp"

#include "prefetch.hpp"
#include "dispatch.hpp"
#include "dispatch_s.hpp"
#include "mux.hpp"
#include "demux.hpp"
#include "eth_router.hpp"
#include "eth_tunneler.hpp"

using namespace tt::tt_metal;

// Helper function to get upstream device in the tunnel from current device, not valid for mmio
chip_id_t FDKernel::GetUpstreamDeviceId(chip_id_t device_id) {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    for (auto tunnel : tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_device_id)) {
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
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    for (auto tunnel : tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_device_id)) {
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
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    for (auto tunnel : tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_device_id)) {
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
        default: TT_FATAL(false, "Unrecognized dispatch kernel type: {}.", type); return nullptr;
    }
}

void FDKernel::configure_kernel_variant(
    const string& path,
    const std::vector<uint32_t>& compile_args,
    std::map<string, string> defines_in,
    bool is_active_eth_core,
    bool send_to_brisc,
    bool force_watcher_no_inline) {
    // TODO: just pass in the programmable index
    uint32_t programmable_core_type_index =
        (GetCoreType() == CoreType::WORKER) ? hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)
        : is_active_eth_core                ? hal.get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH)
                                            : hal.get_programmable_core_type_index(HalProgrammableCoreType::IDLE_ETH);

    std::map<string, string> defines = {
        {"DISPATCH_KERNEL", "1"},
        {"FD_CORE_TYPE", std::to_string(programmable_core_type_index)},
    };
    if (force_watcher_no_inline) {
        defines.insert({"WATCHER_NOINLINE", std::to_string(force_watcher_no_inline)});
    }
    if (tt::llrt::RunTimeOptions::get_instance().watcher_dispatch_disabled()) {
        defines["FORCE_WATCHER_OFF"] = "1";
    }
    // if (!tt::DPrintServerReadsDispatchCores(device_)) {
    //     defines["FORCE_DPRINT_OFF"] = "1";
    // }
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
                .defines = defines});
    } else {
        tt::tt_metal::CreateKernel(
            *program_,
            path,
            logical_core_,
            tt::tt_metal::EthernetConfig{
                .eth_mode = is_active_eth_core ? Eth::SENDER : Eth::IDLE,
                .noc = noc_selection_.non_dispatch_noc,
                .compile_args = compile_args,
                .defines = defines});
    }
}
