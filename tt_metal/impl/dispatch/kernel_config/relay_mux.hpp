// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>
#include "fabric_edm_packet_header.hpp"
#include "fd_kernel.hpp"
#include "tt_metal/impl/dispatch/system_memory_manager.hpp"
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/fabric.hpp>

namespace tt::tt_metal {

struct relay_mux_client_config {
    std::optional<uint32_t> virtual_x;
    std::optional<uint32_t> virtual_y;
    std::optional<uint32_t> num_buffers_per_channel;
    std::optional<uint32_t> channel_buffer_size_bytes;
    std::optional<uint32_t> channel_base_address;
    std::optional<uint32_t> connection_info_address;
    std::optional<uint32_t> connection_handshake_address;
    std::optional<uint32_t> flow_control_address;
    std::optional<uint32_t> buffer_index_address;
    std::optional<uint32_t> status_address;
    std::optional<uint32_t> termination_signal_address;
    std::optional<uint32_t> worker_credits_stream_id;
};

struct relay_mux_static_config {
    // Base address for each buffer
    std::optional<uint32_t> buffer_base_address;

    // Number of full size channels
    std::optional<uint32_t> num_full_size_channels;

    // Number of header only channels
    std::optional<uint32_t> num_header_only_channels;

    // Size of each buffer
    std::optional<uint32_t> buffer_size_bytes;
};

//
// Represents a TT-Fabric MUX to be used for relaying to remote chips
// This FDKernel treats upstream and downstream kernels differently:
//  Upstream kernels: Full size channel
//  Downstreamkernels: Header only channel
//
// Device Id: Which device this kernel is on
// Servicing Device Id: The downstream tunnel index
// d2h: True means this MUX is for returning data from device to host. Servicing device Id is ignored if this is true.
//
class RelayMux : public FDKernel {
private:
    relay_mux_static_config static_config_;
    // This is separate from independent/dependent config as it's managed by fabric
    std::shared_ptr<tt::tt_fabric::FabricMuxConfig> mux_kernel_config_;
    std::vector<uint32_t> mux_ct_args_;
    std::vector<uint32_t> mux_rt_args_;
    bool d2h_ = false;
    int tunnel_id_ = 0;

public:
    RelayMux(
        int node_id,
        chip_id_t device_id,
        chip_id_t servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
        bool d2h,
        int tunnel_index) :
        FDKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection), d2h_{d2h}, tunnel_id_{tunnel_index} {
        TT_FATAL(tunnel_id_ >= 0, "Relay Mux Tunnel Index must be >= 0");
        kernel_type_ = FDKernelType::ROUTING;
    }

    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;
    std::optional<tt::tt_metal::TerminationInfo> GetTerminationInfo() const override {
        return tt::tt_metal::TerminationInfo{
            .logical_core = logical_core_,
            .core_type = GetCoreType(),
            .address = this->mux_kernel_config_->get_termination_signal_address(),
            .val = tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE,
        };
    }

    const relay_mux_static_config& GetStaticConfig() const { return static_config_; }

    // Returns the Mux Kernel config. Populated after generating static configs.
    const std::shared_ptr<tt::tt_fabric::FabricMuxConfig>& GetMuxKernelConfig() const { return mux_kernel_config_; }

    // Returns the channel index for a given worker.
    // Note, workers that need the full size channel are specified in the upstream kernels.
    // and workers that only need the header only channel are specified in the downstream kernels.
    // Throws if not found
    int GetWorkerChannelIndex(int worker_id, tt::tt_fabric::FabricMuxChannelType channel_type) const;
};

// Helper function to assemble the dispatch_fabric_mux_client_config args
void assemble_fabric_mux_client_config_args(
    int node_id,
    tt::tt_fabric::FabricMuxChannelType ch_type,
    const RelayMux* fabric_mux,
    relay_mux_client_config& config);

// Helper function to calculate number of hops from a mmio device to downstream device
// The two devices must be along the same tunnel.
int get_num_hops(chip_id_t mmio_dev_id, chip_id_t downstream_dev_id);

// Helper function to assemble args specific to the 2D fabric header
template <typename Configuration>
void assemble_2d_fabric_packet_header_args(Configuration& config, int my_device_id, int destination_device_id) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(my_device_id);
    const auto& dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(destination_device_id);
    const auto& forwarding_direction = control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
    const auto& mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
    const auto router_direction = control_plane.routing_direction_to_eth_direction(forwarding_direction.value());

    config.my_dev_id = src_fabric_node_id.chip_id;
    config.ew_dim = mesh_shape[1];
    config.to_mesh_id = dst_fabric_node_id.mesh_id.get();
    config.to_dev_id = dst_fabric_node_id.chip_id;
    config.router_direction = router_direction;
}

}  // namespace tt::tt_metal
