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

struct mux_static_config_t {
    std::optional<uint32_t> reserved;
    std::optional<uint32_t> rx_queue_start_addr_words;
    std::optional<uint32_t> rx_queue_size_words;
    std::optional<uint32_t> mux_fan_in;

    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> remote_rx_network_type;  // [4:7]
    std::optional<uint32_t> tx_network_type;

    std::optional<uint32_t> test_results_buf_addr_arg;
    std::optional<uint32_t> test_results_buf_size_bytes;
    std::optional<uint32_t> timeout_cycles;
    std::optional<uint32_t> output_depacketize;
    std::optional<uint32_t> output_depacketize_info;  // Packed, pack with above same is input?
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> input_packetize_local_sem;
    std::optional<uint32_t> input_packetize_src_endpoint;   // Packed w/ max 4 assumption
    std::optional<uint32_t> input_packetize_dest_endpoint;  // Same as src
};

struct mux_dependent_config_t {
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> remote_rx_x;         // [4:7], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> remote_rx_y;         // [4:7], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> remote_rx_queue_id;  // [4:7], dependent
    std::optional<uint32_t> remote_tx_queue_start_addr_words;                              // Dependent
    std::optional<uint32_t> remote_tx_queue_size_words;                                    // Dependent
    std::optional<uint32_t> remote_tx_x;                                                   // Dependent
    std::optional<uint32_t> remote_tx_y;                                                   // Dependent
    std::optional<uint32_t> remote_tx_queue_id;                                            // Dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> input_packetize;  // Dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN>
        input_packetize_log_page_size;                                                                      // Dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_IN> input_packetize_upstream_sem;  // Dependent
};

class MuxKernel : public FDKernel {
public:
    MuxKernel(
        int node_id, chip_id_t device_id, chip_id_t servicing_device_id, uint8_t cq_id, noc_selection_t noc_selection) :
        FDKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection) {}
    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    const mux_static_config_t& GetStaticConfig() { return static_config_; }

private:
    mux_static_config_t static_config_;
    mux_dependent_config_t dependent_config_;
};

}  // namespace tt_metal
}  // namespace tt
