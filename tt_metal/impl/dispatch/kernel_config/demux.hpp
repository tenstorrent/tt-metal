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

struct demux_static_config_t {
    std::optional<uint32_t> endpoint_id_start_index;
    std::optional<uint32_t> rx_queue_start_addr_words;
    std::optional<uint32_t> rx_queue_size_words;
    std::optional<uint32_t> demux_fan_out;

    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT> remote_tx_queue_id;      // [4:7]
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT> remote_tx_network_type;  // [4:7]
    std::optional<uint32_t> remote_rx_network_type;

    std::optional<uint32_t> test_results_buf_addr_arg;
    std::optional<uint32_t> test_results_buf_size_bytes;
    std::optional<uint32_t> timeout_cycles;
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT>
        output_depacketize_cb_log_page_size;  // [26:29]
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT>
        output_depacketize_local_sem_id;  // [26:29]
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT>
        output_depacketize_remove_header;  // [26:29]
};

struct demux_dependent_config_t {
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT> remote_tx_x;  // [4:7], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT> remote_tx_y;  // [4:7], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT>
        remote_tx_queue_start_addr_words;  // [8:2:14], dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT>
        remote_tx_queue_size_words;                                                            // [9:2:15], dependent
    std::optional<uint32_t> remote_rx_x;                                                       // Dependent
    std::optional<uint32_t> remote_rx_y;                                                       // Dependent
    std::optional<uint32_t> remote_rx_queue_id;                                                // Dependent

    std::optional<uint32_t> dest_endpoint_output_map_hi;                                           // Dependent
    std::optional<uint32_t> dest_endpoint_output_map_lo;                                           // Dependent
    std::optional<uint32_t> output_depacketize;                                                    // Dependent
    std::array<std::optional<uint32_t>, tt::packet_queue::MAX_SWITCH_FAN_OUT>
        output_depacketize_downstream_sem_id;  // [26:29], dependent
};

class DemuxKernel : public FDKernel {
public:
    DemuxKernel(
        int node_id, chip_id_t device_id, chip_id_t servicing_device_id, uint8_t cq_id, noc_selection_t noc_selection) :
        FDKernel(node_id, device_id, servicing_device_id, cq_id, noc_selection) {}
    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    const demux_static_config_t& GetStaticConfig() { return static_config_; }
    void SetPlacementCQID(int id) { placement_cq_id_ = id; }

private:
    demux_static_config_t static_config_;
    demux_dependent_config_t dependent_config_;
    int placement_cq_id_;  // TODO: remove channel hard-coding for dispatch core manager
};

}  // namespace tt_metal
}  // namespace tt
