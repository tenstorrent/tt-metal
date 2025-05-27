// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#include <array>
#include <optional>

#include "dispatch/kernels/packet_queue_ctrl.hpp"
#include "fd_kernel.hpp"

namespace tt {
namespace tt_metal {

struct eth_router_static_config_t {
    std::optional<uint32_t> vc_count;                   // Set from arch level
    std::optional<uint32_t> fwd_vc_count;               // # of VCs continuing on to the next chip
    std::optional<uint32_t> rx_queue_start_addr_words;  // 1
    std::optional<uint32_t> rx_queue_size_words;

    std::optional<uint32_t> kernel_status_buf_addr_arg;  // 22
    std::optional<uint32_t> kernel_status_buf_size_bytes;
    std::optional<uint32_t> timeout_cycles;

    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT>
        output_depacketize_log_page_size;                                                                    // [26:29]
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT> output_depacketize_local_sem;  // [26:29]
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT>
        output_depacketize_remove_header;                                                                    // [26:29]
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> input_packetize;                // [30:33]
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> input_packetize_log_page_size;  // [30:33]
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> input_packetize_local_sem;      // [30:33]
};

struct eth_router_dependent_config_t {
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT> remote_tx_x;         // [4:7], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT> remote_tx_y;         // [4:7], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT> remote_tx_queue_id;  // [4:7], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT>
        remote_tx_network_type;  // [4:7], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT>
        remote_tx_queue_start_addr_words;  // [8:2:14], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT>
        remote_tx_queue_size_words;                                                               // [9:2:15], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> remote_rx_x;         // [16:19], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> remote_rx_y;         // [16:19], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> remote_rx_queue_id;  // [16:19], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN>
        remote_rx_network_type;  // [17:19], dependent

    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT> output_depacketize;  // 25, dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT>
        output_depacketize_downstream_sem;  // [26:29], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN>
        input_packetize_upstream_sem;  // [30:33], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> input_packetize_src_endpoint;  // Dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> input_packetize_dst_endpoint;  // Dependent
};

class EthRouterKernel : public FDKernel {
public:
    EthRouterKernel(
        int node_id,
        chip_id_t device_id,
        chip_id_t servicing_device_id,
        uint8_t cq_id,
        noc_selection_t noc_selection,
        bool as_mux) :
        FDKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection), as_mux_(as_mux) {}
    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    const eth_router_static_config_t& GetStaticConfig() { return static_config_; }
    void SetVCCount(uint32_t vc_count) { static_config_.vc_count = vc_count; }
    void SetPlacementCQID(int id) { placement_cq_id_ = id; }

private:
    eth_router_static_config_t static_config_;
    eth_router_dependent_config_t dependent_config_;
    int placement_cq_id_;  // TODO: remove channel hard-coding for dispatch core manager
    bool as_mux_;
};

}  // namespace tt_metal
}  // namespace tt
