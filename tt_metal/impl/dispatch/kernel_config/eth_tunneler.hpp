// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#include <array>
#include <optional>

#include "dispatch/kernels/packet_queue_ctrl.hpp"
#include "fd_kernel.hpp"
#include <umd/device/tt_core_coordinates.h>

namespace tt {
namespace tt_metal {

struct eth_tunneler_static_config_t {
    std::optional<uint32_t> endpoint_id_start_index;
    std::optional<uint32_t> vc_count;  // Set from arch level
    std::optional<uint32_t> in_queue_start_addr_words;
    std::optional<uint32_t> in_queue_size_words;

    std::optional<uint32_t> kernel_status_buf_addr_arg;
    std::optional<uint32_t> kernel_status_buf_size_bytes;
    std::optional<uint32_t> timeout_cycles;
};

struct eth_tunneler_dependent_config_t {
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_TUNNEL_LANES> remote_receiver_x;  // [4:13], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_TUNNEL_LANES> remote_receiver_y;  // [4:13], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_TUNNEL_LANES>
        remote_receiver_queue_id;  // [4:13], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_TUNNEL_LANES>
        remote_receiver_network_type;  // [4:13], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_TUNNEL_LANES>
        remote_receiver_queue_start;  // [14:2:32], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_TUNNEL_LANES>
        remote_receiver_queue_size;                                                           // [15:2:33], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_TUNNEL_LANES> remote_sender_x;  // [34:43], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_TUNNEL_LANES> remote_sender_y;  // [34:43], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_TUNNEL_LANES>
        remote_sender_queue_id;  // [34:43], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_TUNNEL_LANES>
        remote_sender_network_type;  // [34:43], dependent

    std::optional<uint32_t> inner_stop_mux_d_bypass;  // Dependent
};

class EthTunnelerKernel : public FDKernel {
public:
    EthTunnelerKernel(
        int node_id,
        chip_id_t device_id,
        chip_id_t servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
        bool is_remote) :
        FDKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection), is_remote_(is_remote) {}
    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    CoreType GetCoreType() const override {
        // Tunneler kernel is the exception in that it's always on ethernet core even if dispatch is on tensix.
        return CoreType::ETH;
    }
    const eth_tunneler_static_config_t& GetStaticConfig() { return static_config_; }
    bool IsRemote() { return is_remote_; }
    void SetVCCount(uint32_t vc_count) { static_config_.vc_count = vc_count; }
    uint32_t GetRouterQueueIdOffset(FDKernel* k, bool upstream);
    uint32_t GetRouterId(FDKernel* k, bool upstream);

private:
    eth_tunneler_static_config_t static_config_;
    eth_tunneler_dependent_config_t dependent_config_;
    bool is_remote_;
};

}  // namespace tt_metal
}  // namespace tt
