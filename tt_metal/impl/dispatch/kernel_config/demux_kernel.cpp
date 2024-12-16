// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "demux_kernel.hpp"
#include "dispatch_kernel.hpp"
#include "eth_tunneler_kernel.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

void DemuxKernel::GenerateStaticConfigs() {
    uint16_t channel =
        tt::Cluster::instance().get_assigned_channel_for_device(this->servicing_device_id);  // TODO: this can be mmio
    this->logical_core =
        dispatch_core_manager::instance().demux_core(this->servicing_device_id, channel, this->placement_cq_id);
    this->config.endpoint_id_start_index = 0xD1;
    this->config.rx_queue_start_addr_words =
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED) >> 4;
    this->config.rx_queue_size_words = 0x10000 >> 4;

    this->config.remote_rx_network_type = DispatchRemoteNetworkType::NOC0;

    this->config.test_results_buf_addr_arg = 0;
    this->config.test_results_buf_size_bytes = 0;
    this->config.timeout_cycles = 0;

    // TODO: Do we need an upstream sem here?
    for (int idx = 0; idx < this->downstream_kernels.size(); idx++) {
        FDKernel* k = this->downstream_kernels[idx];
        this->config.remote_tx_queue_id[idx] = 0;
        this->config.remote_tx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
        this->config.output_depacketize_cb_log_page_size[idx] = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
        // Only connected dispatchers need a semaphore. TODO: can initialize anyways, but this matches previous
        // implementation
        if (dynamic_cast<DispatchKernel*>(k)) {
            this->config.output_depacketize_local_sem_id[idx] =
                tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
        }
        this->config.output_depacketize_remove_header[idx] = 1;
    }
}

void DemuxKernel::GenerateDependentConfigs() {
    auto& my_dispatch_constants = dispatch_constants::get(GetCoreType());
    // Upstream, expect EthTunneler or DEMUX
    TT_ASSERT(this->upstream_kernels.size() == 1);
    if (auto us = dynamic_cast<EthTunnelerKernel*>(this->upstream_kernels[0])) {
        this->config.remote_rx_x = us->GetVirtualCore().x;
        this->config.remote_rx_y = us->GetVirtualCore().y;
        this->config.remote_rx_queue_id = us->GetConfig().vc_count.value() * 2 - 1;
    } else if (auto us = dynamic_cast<DemuxKernel*>(this->upstream_kernels[0])) {
        this->config.remote_rx_x = us->GetVirtualCore().x;
        this->config.remote_rx_y = us->GetVirtualCore().y;
        this->config.remote_rx_queue_id = us->GetDownstreamPort(this) + 1;  // TODO: can this be cleaned up?
        // TODO: why is just this one different? Just match previous implementation for now
        if (us->GetDownstreamPort(this) == 1) {
            this->config.endpoint_id_start_index =
                this->config.endpoint_id_start_index.value() + this->downstream_kernels.size();
        }
    } else {
        TT_FATAL(false, "Unexpected kernel type upstream of DEMUX");
    }

    // Downstream, expect DISPATCH_H or DEMUX
    TT_ASSERT(this->downstream_kernels.size() <= MAX_SWITCH_FAN_OUT && this->downstream_kernels.size() > 0);
    this->config.demux_fan_out = this->downstream_kernels.size();
    this->config.output_depacketize = 0;  // Populated per downstream kernel
    for (int idx = 0; idx < this->downstream_kernels.size(); idx++) {
        FDKernel* k = this->downstream_kernels[idx];
        this->config.remote_tx_x[idx] = k->GetVirtualCore().x;
        this->config.remote_tx_y[idx] = k->GetVirtualCore().y;
        // Expect downstream to be either a DISPATCH or another DEMUX
        if (auto dispatch_kernel = dynamic_cast<DispatchKernel*>(k)) {
            this->config.remote_tx_queue_start_addr_words[idx] =
                dispatch_kernel->GetConfig().dispatch_cb_base.value() >> 4;
            this->config.remote_tx_queue_size_words[idx] =
                ((1 << dispatch_kernel->GetConfig().dispatch_cb_log_page_size.value()) *
                 dispatch_kernel->GetConfig().dispatch_cb_pages.value()) >>
                4;
            this->config.output_depacketize =
                this->config.output_depacketize.value() | (1 << idx);  // Only depacketize for dispatch downstream
            this->config.output_depacketize_downstream_sem_id[idx] = dispatch_kernel->GetConfig().my_dispatch_cb_sem_id;
            uint32_t dest_map_array[4] = {0, 1, 2, 3};  // TODO: how to set these generically? Currently just matching
                                                        // the hard-coded previous implementation
            uint64_t dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
            this->config.dest_endpoint_output_map_hi = (uint32_t)(dest_endpoint_output_map >> 32);
            this->config.dest_endpoint_output_map_lo = (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF);
        } else if (auto demux_kernel = dynamic_cast<DemuxKernel*>(k)) {
            this->config.remote_tx_queue_start_addr_words[idx] =
                demux_kernel->GetConfig().rx_queue_start_addr_words.value();
            this->config.remote_tx_queue_size_words[idx] = 0x1000;  // TODO: hard-coded on previous implementation
            // Match previous implementation where downstream demux has output_depacketize fields zeroed out. TODO: can
            // remove this later
            this->config.output_depacketize_downstream_sem_id[idx] = 0;
            uint64_t dest_endpoint_output_map;
            if (device->num_hw_cqs() == 1) {
                uint32_t dest_map_array[4] = {0, 0, 1, 1};  // TODO: how to set these generically? Currently just
                                                            // matching the hard-coded previous implementation
                dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
            } else {
                uint32_t dest_map_array[8] = {0, 0, 0, 0, 1, 1, 1, 1};
                dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 8);
            }
            this->config.dest_endpoint_output_map_hi = (uint32_t)(dest_endpoint_output_map >> 32);
            this->config.dest_endpoint_output_map_lo = (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF);
        } else {
            TT_FATAL(false, "Unexpected kernel type downstream of DEMUX");
        }
    }
    // TODO: this is just to match the previous implementation hard-code, remove later
    if (!tt::Cluster::instance().is_galaxy_cluster()) {
        this->config.output_depacketize = 0x3;
    }
}

void DemuxKernel::CreateKernel() {
    std::vector<uint32_t> compile_args = {
        config.endpoint_id_start_index.value(),
        config.rx_queue_start_addr_words.value(),
        config.rx_queue_size_words.value(),
        config.demux_fan_out.value(),
        0,
        0,
        0,
        0,  // Populate remote_tx_config after
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  // Populate remote_tx_queue_start_addr_words & remote_tx_queue_size_words after
        config.remote_rx_x.value(),
        config.remote_rx_y.value(),
        config.remote_rx_queue_id.value(),
        config.remote_rx_network_type.value(),
        config.dest_endpoint_output_map_hi.value(),
        config.dest_endpoint_output_map_lo.value(),
        config.test_results_buf_addr_arg.value(),
        config.test_results_buf_size_bytes.value(),
        config.timeout_cycles.value(),
        config.output_depacketize.value(),
        0,
        0,
        0,
        0  // Populate output_depacketize_config after
    };
    for (int idx = 0; idx < MAX_SWITCH_FAN_OUT; idx++) {
        if (config.remote_tx_x[idx]) {
            compile_args[4 + idx] |= (config.remote_tx_x[idx].value() & 0xFF);
            compile_args[4 + idx] |= (config.remote_tx_y[idx].value() & 0xFF) << 8;
            compile_args[4 + idx] |= (config.remote_tx_queue_id[idx].value() & 0xFF) << 16;
            compile_args[4 + idx] |= (config.remote_tx_network_type[idx].value() & 0xFF) << 24;
        }
        if (config.remote_tx_queue_start_addr_words[idx]) {
            compile_args[8 + idx * 2] = config.remote_tx_queue_start_addr_words[idx].value();
            compile_args[9 + idx * 2] = config.remote_tx_queue_size_words[idx].value();
        }
        if (config.output_depacketize_cb_log_page_size[idx]) {
            // To match previous implementation, zero these out if output_depacketize is not set. TODO: don't have to do
            // this
            if (config.output_depacketize.value() & (1 << idx)) {
                compile_args[26 + idx] |= (config.output_depacketize_cb_log_page_size[idx].value() & 0xFF);
                compile_args[26 + idx] |= (config.output_depacketize_downstream_sem_id[idx].value() & 0xFF) << 8;
                compile_args[26 + idx] |= (config.output_depacketize_local_sem_id[idx].value() & 0xFF) << 16;
                compile_args[26 + idx] |= (config.output_depacketize_remove_header[idx].value() & 0xFF) << 24;
            }
        }
    }
    TT_ASSERT(compile_args.size() == 30);
    const auto& grid_size = device->grid_size();
    tt_cxy_pair my_virtual_core =
        tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(this->logical_core, GetCoreType());
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
    configure_kernel_variant(dispatch_kernel_file_names[DEMUX], compile_args, defines, false, false, false);
}
