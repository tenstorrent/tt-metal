// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "mux_kernel.hpp"
#include "dispatch_kernel.hpp"
#include "eth_router_kernel.hpp"
#include "eth_tunneler_kernel.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

void MuxKernel::GenerateStaticConfigs() {
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device->id());
    this->logical_core = dispatch_core_manager::instance().mux_d_core(this->device->id(), channel, this->cq_id);
    auto& my_dispatch_constants = dispatch_constants::get(GetCoreType());
    this->config.reserved = 0;
    this->config.rx_queue_start_addr_words = my_dispatch_constants.dispatch_buffer_base() >> 4;
    this->config.rx_queue_size_words = ((1 << dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE) *
                                        my_dispatch_constants.mux_buffer_pages(device->num_hw_cqs())) >>
                                       4;
    this->config.mux_fan_in = this->upstream_kernels.size();
    for (int idx = 0; idx < this->upstream_kernels.size(); idx++) {
        this->config.remote_rx_network_type[idx] = DispatchRemoteNetworkType::NOC0;
    }

    this->config.tx_network_type = (uint32_t)DispatchRemoteNetworkType::NOC0;
    this->config.test_results_buf_addr_arg = 0;
    this->config.test_results_buf_size_bytes = 0;
    this->config.timeout_cycles = 0;
    this->config.output_depacketize = 0x0;
    this->config.output_depacketize_info = 0x0;

    for (int idx = 0; idx < this->upstream_kernels.size(); idx++) {
        // Only connected dispatchers need a semaphore. TODO: can initialize anyways, but this matches previous
        // implementation
        if (dynamic_cast<DispatchKernel*>(this->upstream_kernels[idx])) {
            this->config.input_packetize_local_sem[idx] =
                tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
        }
    }
}

void MuxKernel::GenerateDependentConfigs() {
    // Upstream, expect DISPATCH_D or TUNNELER
    TT_ASSERT(this->upstream_kernels.size() <= MAX_SWITCH_FAN_IN && this->upstream_kernels.size() > 0);
    uint32_t num_upstream_dispatchers = 0;
    for (int idx = 0; idx < this->upstream_kernels.size(); idx++) {
        FDKernel* k = this->upstream_kernels[idx];
        this->config.remote_rx_x[idx] = k->GetVirtualCore().x;
        this->config.remote_rx_y[idx] = k->GetVirtualCore().y;
        this->config.input_packetize_log_page_size[idx] =
            dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;  // Does this ever change?
        if (auto dispatch_kernel = dynamic_cast<DispatchKernel*>(k)) {
            this->config.input_packetize[idx] = 0x1;
            this->config.input_packetize_upstream_sem[idx] = dispatch_kernel->GetConfig().my_downstream_cb_sem_id;
            this->config.remote_rx_queue_id[idx] = 1;
            num_upstream_dispatchers++;
        } else if (auto tunneler_kernel = dynamic_cast<EthTunnelerKernel*>(k)) {
            // Don't need to packetize input from tunneler
            this->config.input_packetize[idx] = 0x0;
            this->config.input_packetize_upstream_sem[idx] = 0;
            this->config.remote_rx_queue_id[idx] = tunneler_kernel->GetConfig().vc_count.value() * 2 - 1;
        } else {
            TT_FATAL(false, "Unexpected kernel type upstream of MUX");
        }
    }
    uint32_t src_id = 0xC1 + (FDKernel::GetTunnelStop(device_id) - 1) * num_upstream_dispatchers;
    uint32_t dest_id = 0xD1 + (FDKernel::GetTunnelStop(device_id) - 1) * num_upstream_dispatchers;
    this->config.input_packetize_src_endpoint = packet_switch_4B_pack(src_id, src_id + 1, src_id + 2, src_id + 3);
    this->config.input_packetize_dest_endpoint = packet_switch_4B_pack(dest_id, dest_id + 1, dest_id + 2, dest_id + 3);

    // Downstream, expect TUNNELER
    TT_ASSERT(this->downstream_kernels.size() == 1);
    FDKernel* ds = this->downstream_kernels[0];
    auto tunneler_kernel = dynamic_cast<EthTunnelerKernel*>(ds);
    TT_ASSERT(ds);
    this->config.remote_tx_queue_start_addr_words =
        tunneler_kernel->GetConfig().in_queue_start_addr_words.value() +
        (tunneler_kernel->GetConfig().vc_count.value() - 1) * tunneler_kernel->GetConfig().in_queue_size_words.value();
    this->config.remote_tx_queue_size_words = tunneler_kernel->GetConfig().in_queue_size_words;
    this->config.remote_tx_x = ds->GetVirtualCore().x;
    this->config.remote_tx_y = ds->GetVirtualCore().y;
    this->config.remote_tx_queue_id = tunneler_kernel->GetConfig().vc_count.value() - 1;
}

void MuxKernel::CreateKernel() {
    std::vector<uint32_t> compile_args = {
        config.reserved.value(),
        config.rx_queue_start_addr_words.value(),
        config.rx_queue_size_words.value(),
        config.mux_fan_in.value(),
        0,
        0,
        0,
        0,  // Populate remote_rx_config after
        config.remote_tx_queue_start_addr_words.value(),
        config.remote_tx_queue_size_words.value(),
        config.remote_tx_x.value(),
        config.remote_tx_y.value(),
        config.remote_tx_queue_id.value(),
        config.tx_network_type.value(),
        config.test_results_buf_addr_arg.value(),
        config.test_results_buf_size_bytes.value(),
        config.timeout_cycles.value(),
        config.output_depacketize.value(),
        config.output_depacketize_info.value(),
        0,
        0,
        0,
        0,  // Populate input_packetize_config after
        config.input_packetize_src_endpoint.value(),
        config.input_packetize_dest_endpoint.value()};
    for (int idx = 0; idx < MAX_SWITCH_FAN_IN; idx++) {
        if (config.remote_rx_x[idx]) {
            compile_args[4 + idx] |= (config.remote_rx_x[idx].value() & 0xFF);
            compile_args[4 + idx] |= (config.remote_rx_y[idx].value() & 0xFF) << 8;
            compile_args[4 + idx] |= (config.remote_rx_queue_id[idx].value() & 0xFF) << 16;
            compile_args[4 + idx] |= (config.remote_rx_network_type[idx].value() & 0xFF) << 24;
        }
        if (config.input_packetize[idx]) {
            // Zero out if input packetize not set to match previous implementation. TODO: don't have to do this
            if (config.input_packetize[idx].value() != 0) {
                compile_args[19 + idx] |= (config.input_packetize[idx].value() & 0xFF);
                compile_args[19 + idx] |= (config.input_packetize_log_page_size[idx].value() & 0xFF) << 8;
                compile_args[19 + idx] |= (config.input_packetize_upstream_sem[idx].value() & 0xFF) << 16;
                compile_args[19 + idx] |= (config.input_packetize_local_sem[idx].value() & 0xFF) << 24;
            }
        }
    }
    TT_ASSERT(compile_args.size() == 25);
    const auto& grid_size = device->grid_size();
    std::map<string, string> defines = {
        // All of these unused, remove later
        {"MY_NOC_X",
         std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.x, 0))},
        {"MY_NOC_Y",
         std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.non_dispatch_noc, grid_size.y, 0))},
        {"UPSTREAM_NOC_INDEX", std::to_string(this->noc_selection.upstream_noc)},
        {"UPSTREAM_NOC_X",
         std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.x, 0))},
        {"UPSTREAM_NOC_Y",
         std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.upstream_noc, grid_size.y, 0))},
        {"DOWNSTREAM_NOC_X",
         std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, 0))},
        {"DOWNSTREAM_NOC_Y",
         std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, 0))},
        {"DOWNSTREAM_SLAVE_NOC_X",
         std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.x, 0))},
        {"DOWNSTREAM_SLAVE_NOC_Y",
         std::to_string(tt::tt_metal::hal.noc_coordinate(this->noc_selection.downstream_noc, grid_size.y, 0))},
        {"SKIP_NOC_LOGGING", "1"}};
    configure_kernel_variant(dispatch_kernel_file_names[MUX_D], compile_args, defines, false, false, false);
}
