// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "mux.hpp"
#include "dispatch.hpp"
#include "eth_router.hpp"
#include "eth_tunneler.hpp"

#include <host_api.hpp>
#include <tt_metal.hpp>

using namespace tt::tt_metal;

void MuxKernel::GenerateStaticConfigs() {
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_->id());
    logical_core_ = dispatch_core_manager::instance().mux_d_core(device_->id(), channel, this->cq_id_);
    auto& my_dispatch_constants = dispatch_constants::get(GetCoreType());
    static_config_.reserved = 0;
    static_config_.rx_queue_start_addr_words = my_dispatch_constants.dispatch_buffer_base() >> 4;
    static_config_.rx_queue_size_words = ((1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) *
                                          my_dispatch_constants.mux_buffer_pages(device_->num_hw_cqs())) >>
                                         4;
    static_config_.mux_fan_in = upstream_kernels_.size();
    for (int idx = 0; idx < upstream_kernels_.size(); idx++) {
        static_config_.remote_rx_network_type[idx] = DispatchRemoteNetworkType::NOC0;
    }

    static_config_.tx_network_type = (uint32_t)DispatchRemoteNetworkType::NOC0;
    static_config_.test_results_buf_addr_arg = 0;
    static_config_.test_results_buf_size_bytes = 0;
    static_config_.timeout_cycles = 0;
    static_config_.output_depacketize = 0x0;
    static_config_.output_depacketize_info = 0x0;

    for (int idx = 0; idx < upstream_kernels_.size(); idx++) {
        static_config_.input_packetize_local_sem[idx] =
            tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
    }
}

void MuxKernel::GenerateDependentConfigs() {
    // Upstream, expect DISPATCH_D or TUNNELER
    TT_ASSERT(upstream_kernels_.size() <= MAX_SWITCH_FAN_IN && upstream_kernels_.size() > 0);
    uint32_t num_upstream_dispatchers = 0;
    for (int idx = 0; idx < upstream_kernels_.size(); idx++) {
        FDKernel* k = upstream_kernels_[idx];
        dependent_config_.remote_rx_x[idx] = k->GetVirtualCore().x;
        dependent_config_.remote_rx_y[idx] = k->GetVirtualCore().y;
        dependent_config_.input_packetize_log_page_size[idx] =
            dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;  // Does this ever change?
        if (auto dispatch_kernel = dynamic_cast<DispatchKernel*>(k)) {
            dependent_config_.input_packetize[idx] = 0x1;
            dependent_config_.input_packetize_upstream_sem[idx] =
                dispatch_kernel->GetStaticConfig().my_downstream_cb_sem_id;
            dependent_config_.remote_rx_queue_id[idx] = 1;
            num_upstream_dispatchers++;
        } else if (auto tunneler_kernel = dynamic_cast<EthTunnelerKernel*>(k)) {
            // Don't need to packetize input from tunneler
            dependent_config_.input_packetize[idx] = 0x0;
            dependent_config_.input_packetize_upstream_sem[idx] = 0;
            dependent_config_.remote_rx_queue_id[idx] = tunneler_kernel->GetStaticConfig().vc_count.value() * 2 - 1;
        } else {
            TT_FATAL(false, "Unexpected kernel type upstream of MUX");
        }
    }
    uint32_t src_id = 0xC1 + (FDKernel::GetTunnelStop(device_id_) - 1) * num_upstream_dispatchers;
    uint32_t dest_id = 0xD1 + (FDKernel::GetTunnelStop(device_id_) - 1) * num_upstream_dispatchers;
    static_config_.input_packetize_src_endpoint = packet_switch_4B_pack(src_id, src_id + 1, src_id + 2, src_id + 3);
    static_config_.input_packetize_dest_endpoint =
        packet_switch_4B_pack(dest_id, dest_id + 1, dest_id + 2, dest_id + 3);

    // Downstream, expect TUNNELER
    TT_ASSERT(downstream_kernels_.size() == 1);
    FDKernel* ds = downstream_kernels_[0];
    auto tunneler_kernel = dynamic_cast<EthTunnelerKernel*>(ds);
    TT_ASSERT(ds);
    dependent_config_.remote_tx_queue_start_addr_words =
        tunneler_kernel->GetStaticConfig().in_queue_start_addr_words.value() +
        (tunneler_kernel->GetStaticConfig().vc_count.value() - 1) *
            tunneler_kernel->GetStaticConfig().in_queue_size_words.value();
    dependent_config_.remote_tx_queue_size_words = tunneler_kernel->GetStaticConfig().in_queue_size_words;
    dependent_config_.remote_tx_x = ds->GetVirtualCore().x;
    dependent_config_.remote_tx_y = ds->GetVirtualCore().y;
    dependent_config_.remote_tx_queue_id = tunneler_kernel->GetStaticConfig().vc_count.value() - 1;
}

void MuxKernel::CreateKernel() {
    std::vector<uint32_t> compile_args = {
        static_config_.reserved.value(),
        static_config_.rx_queue_start_addr_words.value(),
        static_config_.rx_queue_size_words.value(),
        static_config_.mux_fan_in.value(),
        0,
        0,
        0,
        0,  // Populate remote_rx_config after
        dependent_config_.remote_tx_queue_start_addr_words.value(),
        dependent_config_.remote_tx_queue_size_words.value(),
        dependent_config_.remote_tx_x.value(),
        dependent_config_.remote_tx_y.value(),
        dependent_config_.remote_tx_queue_id.value(),
        static_config_.tx_network_type.value(),
        static_config_.test_results_buf_addr_arg.value(),
        static_config_.test_results_buf_size_bytes.value(),
        static_config_.timeout_cycles.value(),
        static_config_.output_depacketize.value(),
        static_config_.output_depacketize_info.value(),
        0,
        0,
        0,
        0,  // Populate input_packetize_config after
        static_config_.input_packetize_src_endpoint.value(),
        static_config_.input_packetize_dest_endpoint.value()};
    for (int idx = 0; idx < MAX_SWITCH_FAN_IN; idx++) {
        if (dependent_config_.remote_rx_x[idx]) {
            compile_args[4 + idx] |= (dependent_config_.remote_rx_x[idx].value() & 0xFF);
            compile_args[4 + idx] |= (dependent_config_.remote_rx_y[idx].value() & 0xFF) << 8;
            compile_args[4 + idx] |= (dependent_config_.remote_rx_queue_id[idx].value() & 0xFF) << 16;
            compile_args[4 + idx] |= (static_config_.remote_rx_network_type[idx].value() & 0xFF) << 24;
        }
        if (dependent_config_.input_packetize[idx]) {
            if (dependent_config_.input_packetize[idx]) {
                compile_args[19 + idx] |= (dependent_config_.input_packetize[idx].value() & 0xFF);
                compile_args[19 + idx] |= (dependent_config_.input_packetize_log_page_size[idx].value() & 0xFF) << 8;
                compile_args[19 + idx] |= (dependent_config_.input_packetize_upstream_sem[idx].value() & 0xFF) << 16;
                compile_args[19 + idx] |= (static_config_.input_packetize_local_sem[idx].value() & 0xFF) << 24;
            }
        }
    }
    TT_ASSERT(compile_args.size() == 25);
    const auto& grid_size = device_->grid_size();
    std::map<string, string> defines = {
        // All of these unused, remove later
        {"MY_NOC_X", std::to_string(tt::tt_metal::hal.noc_coordinate(noc_selection_.non_dispatch_noc, grid_size.x, 0))},
        {"MY_NOC_Y", std::to_string(tt::tt_metal::hal.noc_coordinate(noc_selection_.non_dispatch_noc, grid_size.y, 0))},
        {"UPSTREAM_NOC_INDEX", std::to_string(noc_selection_.upstream_noc)},
        {"UPSTREAM_NOC_X",
         std::to_string(tt::tt_metal::hal.noc_coordinate(noc_selection_.upstream_noc, grid_size.x, 0))},
        {"UPSTREAM_NOC_Y",
         std::to_string(tt::tt_metal::hal.noc_coordinate(noc_selection_.upstream_noc, grid_size.y, 0))},
        {"DOWNSTREAM_NOC_X",
         std::to_string(tt::tt_metal::hal.noc_coordinate(noc_selection_.downstream_noc, grid_size.x, 0))},
        {"DOWNSTREAM_NOC_Y",
         std::to_string(tt::tt_metal::hal.noc_coordinate(noc_selection_.downstream_noc, grid_size.y, 0))},
        {"DOWNSTREAM_SLAVE_NOC_X",
         std::to_string(tt::tt_metal::hal.noc_coordinate(noc_selection_.downstream_noc, grid_size.x, 0))},
        {"DOWNSTREAM_SLAVE_NOC_Y",
         std::to_string(tt::tt_metal::hal.noc_coordinate(noc_selection_.downstream_noc, grid_size.y, 0))},
        {"SKIP_NOC_LOGGING", "1"}};
    configure_kernel_variant(dispatch_kernel_file_names[MUX_D], compile_args, defines, false, false, false);
}
