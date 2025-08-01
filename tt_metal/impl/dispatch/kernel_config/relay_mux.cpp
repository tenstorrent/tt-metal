// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "relay_mux.hpp"
#include "context/metal_context.hpp"
#include "dispatch/kernel_config/fd_kernel.hpp"
#include "dispatch_core_common.hpp"
#include "fabric/fabric_host_utils.hpp"
#include "fabric/fabric_context.hpp"
#include "hal_types.hpp"
#include <bit>
#include <tt-logger/tt-logger.hpp>
#include "tt_align.hpp"
#include "tt_metal.hpp"
#include "umd/device/tt_core_coordinates.h"
#include <algorithm>
#include <tt-metalium/fabric.hpp>

namespace tt::tt_metal {

void RelayMux::GenerateStaticConfigs() {
    uint32_t l1_base = 0;
    uint32_t l1_size = 0;

    if (GetCoreType() == CoreType::WORKER) {
        l1_base = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);
        l1_size = tt::tt_metal::MetalContext::instance().hal().get_dev_size(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE);
    } else {
        l1_base = MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
        l1_size = MetalContext::instance().hal().get_dev_size(
            tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::BASE);
    }
    static_config_.buffer_base_address =
        tt::align(l1_base, tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1));

    const uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_->id());
    logical_core_ = MetalContext::instance().get_dispatch_core_manager().fabric_mux_core(
        device_->id(), channel, this->cq_id_, tunnel_id_);

    // Count number of value kernels that need the channels
    const auto kernels_requiring_full_size_channel =
        (uint32_t)std::count_if(upstream_kernels_.begin(), upstream_kernels_.end(), [](const FDKernel* kernel) {
            return kernel->GetNodeId() != -1;
        });
    const auto kernels_requiring_header_only_channel =
        (uint32_t)std::count_if(downstream_kernels_.begin(), downstream_kernels_.end(), [](const FDKernel* kernel) {
            return kernel->GetNodeId() != -1;
        });

    // Setup the buffer sizes depending on how many upstream/downstream kernels are connected
    static_config_.num_full_size_channels = kernels_requiring_full_size_channel;
    static_config_.num_header_only_channels = kernels_requiring_header_only_channel;

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();

    // Buffer size for the Mux must matching downstream fabric router size
    // Round down to nearest power of 2
    uint32_t mux_buffer_size = std::bit_floor(tt_fabric::get_tt_fabric_max_payload_size_bytes());
    uint32_t header_size = fabric_context.get_fabric_packet_header_size_bytes();
    static_config_.buffer_size_bytes = header_size + mux_buffer_size;
    uint32_t num_slots = 16;
    if (mux_buffer_size < 3072) {
        // Due to 2D fabric having smaller buffer size more slots can be used
        num_slots = 32;
    }

    // FabricMuxConfig only accepts Worker or Idle Eth. Eth is not accepted.
    CoreType mux_config_core = GetCoreType() == CoreType::WORKER ? CoreType::WORKER : CoreType::IDLE_ETH;
    mux_kernel_config_ = std::make_shared<tt::tt_fabric::FabricMuxConfig>(
        static_config_.num_full_size_channels.value(),
        static_config_.num_header_only_channels.value(),
        num_slots,
        num_slots,
        static_config_.buffer_size_bytes.value(),
        static_config_.buffer_base_address.value(),
        mux_config_core);
    mux_ct_args_ = mux_kernel_config_->get_fabric_mux_compile_time_args();

    uint32_t mux_buffer_end = mux_kernel_config_->get_memory_map_end_address();
    TT_ASSERT(mux_buffer_end < l1_size, "RelayMux Buffer End {} Exceeds Max L1 {}", mux_buffer_end, l1_size);

    int destination_device_id = -1;
    TT_FATAL(!(d2h_ && device_->is_mmio_capable()), "There is no D2H (return path) for MMIO devices");
    if (d2h_) {
        // Get the device which is upstream
        destination_device_id = tt::tt_metal::FDKernel::GetUpstreamDeviceId(device_id_);
    } else {
        // Get the device which is downstream on the specified tunnel
        destination_device_id = tt::tt_metal::FDKernel::GetDownstreamDeviceId(device_id_, tunnel_id_);
    }
    const auto src_fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(device_id_);
    const auto dst_fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(destination_device_id);

    auto get_link_idx = [&](tt::tt_fabric::FabricNodeId src_fabric_node_id,
                            tt::tt_fabric::FabricNodeId dst_fabric_node_id) -> uint32_t {
        if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster() && device_->is_mmio_capable()) {
            const auto forwarding_direction =
                control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
            TT_FATAL(
                forwarding_direction.has_value(),
                "No forwarding directions found from {} to {}",
                src_fabric_node_id,
                dst_fabric_node_id);
            const auto& fabric_channels =
                control_plane.get_active_fabric_eth_channels_in_direction(src_fabric_node_id, *forwarding_direction);

            for (auto i = 0; i < fabric_channels.size(); i++) {
                const auto fabric_route =
                    control_plane.get_fabric_route(src_fabric_node_id, dst_fabric_node_id, fabric_channels[i]);
                if (fabric_route.size() == 1) {
                    return i;
                }
            }
        } else {
            const auto& available_links =
                tt_fabric::get_forwarding_link_indices(src_fabric_node_id, dst_fabric_node_id);
            TT_FATAL(
                !available_links.empty(), "No links available from {} to {}", src_fabric_node_id, dst_fabric_node_id);
            return available_links.back();
        }

        TT_THROW("Unable to find forwarding link from {} to {}", src_fabric_node_id, dst_fabric_node_id);
    };

    auto link_index = get_link_idx(src_fabric_node_id, dst_fabric_node_id);
    log_debug(
        tt::LogMetal,
        "RelayMux Device:{}, HeaderCh:{}, FullCh:{}, FullB:{}, Logical:{}, Virtual: {}, D2H: {} Channel Size: {}, Num "
        "Slots: {}, L1 Size: {}, Src: {}, Dst: {}, Link Index: {}",
        device_->id(),
        kernels_requiring_header_only_channel,
        kernels_requiring_full_size_channel,
        static_config_.buffer_size_bytes.value(),
        logical_core_.str(),
        GetVirtualCore().str(),
        d2h_,
        mux_buffer_size,
        num_slots,
        l1_size,
        src_fabric_node_id,
        dst_fabric_node_id,
        link_index);

    mux_rt_args_ = mux_kernel_config_->get_fabric_mux_run_time_args(
        src_fabric_node_id, dst_fabric_node_id, link_index, *program_, {logical_core_});
}

void RelayMux::GenerateDependentConfigs() {}

void RelayMux::CreateKernel() {
    auto mux_kernel =
        configure_kernel_variant(dispatch_kernel_file_names[FABRIC_MUX], mux_ct_args_, {}, false, false, false);

    tt::tt_metal::SetRuntimeArgs(*program_, mux_kernel, logical_core_, mux_rt_args_);
}

void RelayMux::ConfigureCore() {}

int RelayMux::GetWorkerChannelIndex(int worker_id, tt::tt_fabric::FabricMuxChannelType channel_type) const {
    const auto& kernels = channel_type == tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL ? upstream_kernels_
                                                                                                 : downstream_kernels_;
    for (int i = 0; i < kernels.size(); ++i) {
        if (kernels[i]->GetNodeId() == worker_id) {
            return i;
        }
    }

    TT_THROW("Worker ID {} not found in upstream/downstream kernels", worker_id);
    return -1;
}

void assemble_fabric_mux_client_config_args(
    int node_id,
    tt::tt_fabric::FabricMuxChannelType ch_type,
    const RelayMux* fabric_mux,
    relay_mux_client_config& config) {
    const auto ch_index = fabric_mux->GetWorkerChannelIndex(node_id, ch_type);
    const CoreCoord& fabric_mux_core = fabric_mux->GetVirtualCore();
    config.virtual_x = fabric_mux_core.x;
    config.virtual_y = fabric_mux_core.y;
    config.num_buffers_per_channel = fabric_mux->GetMuxKernelConfig()->get_num_buffers(ch_type);
    config.channel_buffer_size_bytes = fabric_mux->GetMuxKernelConfig()->get_buffer_size_bytes(ch_type);
    config.channel_base_address = fabric_mux->GetMuxKernelConfig()->get_channel_base_address(ch_type, ch_index);
    config.connection_info_address = fabric_mux->GetMuxKernelConfig()->get_connection_info_address(ch_type, ch_index);
    config.connection_handshake_address =
        fabric_mux->GetMuxKernelConfig()->get_connection_handshake_address(ch_type, ch_index);
    config.flow_control_address = fabric_mux->GetMuxKernelConfig()->get_flow_control_address(ch_type, ch_index);
    config.buffer_index_address = fabric_mux->GetMuxKernelConfig()->get_buffer_index_address(ch_type, ch_index);
    config.status_address = fabric_mux->GetMuxKernelConfig()->get_status_address();
    config.termination_signal_address = fabric_mux->GetMuxKernelConfig()->get_termination_signal_address();
    config.worker_credits_stream_id =
        fabric_mux->GetMuxKernelConfig()->get_channel_credits_stream_id(ch_type, ch_index);
}

int get_num_hops(chip_id_t mmio_dev_id, chip_id_t downstream_dev_id) {
    const auto dev_mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(mmio_dev_id);

    if (dev_mmio_device_id != mmio_dev_id) {
        TT_THROW(
            "RelayMux Specified MMIO device ID {} is not an MMIO device. MMIO device is {}",
            mmio_dev_id,
            dev_mmio_device_id);
    }

    auto tunnels_from_mmio =
        tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_dev_id);

    constexpr size_t k_MaxTunnelSize = 5;  // 4 remote + 1 mmio
    for (const auto& tunnel : tunnels_from_mmio) {
        TT_FATAL(
            tunnel.size() <= k_MaxTunnelSize,
            "Unexpected tunnel size {}. Max tunnel size expected {}",
            tunnel.size(),
            k_MaxTunnelSize);
        for (int hop = 0; hop < tunnel.size(); ++hop) {
            if (tunnel[hop] == downstream_dev_id) {
                return hop;
            }
        }
    }
    TT_THROW(
        "RelayMux Downstream device {} is not found in tunnel from MMIO device {}", downstream_dev_id, mmio_dev_id);
    return -1;
}
}  // namespace tt::tt_metal
