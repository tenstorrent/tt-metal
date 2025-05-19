// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>
#include "fd_kernel.hpp"
#include "tt_metal/impl/dispatch/system_memory_manager.hpp"
#include "tt_metal/fabric/fabric_mux_config.hpp"
#include <tt-metalium/control_plane.hpp>

namespace tt::tt_metal {

struct fabric_mux_client_config {
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
};

struct fabric_mux_static_config {
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
// Represents a TT-Fabric MUX
// This FDKernel treats upstream and downstream kernels differently:
//  Upstream kernels: Full size channel
//  Downstreamkernels: Header only channel
//
// Device Id: Which device this kernel is on
// Servicing Device Id: Which device this kernel intends to send data to
//
class FabricMux : public FDKernel {
private:
    fabric_mux_static_config static_config_;
    // This is separate from independent/dependent config as it's managed by fabric
    std::shared_ptr<tt::tt_fabric::FabricMuxConfig> mux_kernel_config_;
    std::vector<uint32_t> mux_ct_args;
    std::vector<uint32_t> mux_rt_args;

public:
    FabricMux(
        int node_id, chip_id_t device_id, chip_id_t servicing_device_id, uint8_t cq_id, noc_selection_t noc_selection) :
        FDKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection) {
        kernel_type_ = FDKernelType::ROUTING;
    }

    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;

    const fabric_mux_static_config& GetStaticConfig() const { return static_config_; }

    // Returns the Mux Kernel config. Populated after generating static configs.
    const std::shared_ptr<tt::tt_fabric::FabricMuxConfig>& GetMuxKernelConfig() const { return mux_kernel_config_; }

    // Returns the channel index for a given worker.
    // Note, workers that need the full size channel are specified in the upstream kernels.
    // and workers that only need the header only channel are specified in the downstream kernels.
    // Throws if not found
    int GetWorkerChannelIndex(int worker_id, tt::tt_fabric::FabricMuxChannelType channel_type) const;
};

// Helper function to assemble the fabric_mux_client_config args
void assemble_fabric_mux_client_config_args(
    int node_id,
    tt::tt_fabric::FabricMuxChannelType ch_type,
    const FabricMux* fabric_mux,
    fabric_mux_client_config& config);

}  // namespace tt::tt_metal
