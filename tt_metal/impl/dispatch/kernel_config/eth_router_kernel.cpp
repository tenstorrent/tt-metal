// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "eth_router_kernel.hpp"
#include "prefetch_kernel.hpp"
#include "eth_tunneler_kernel.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

void EthRouterKernel::GenerateStaticConfigs() {
    auto& my_dispatch_constants = dispatch_constants::get(GetCoreType());
    if (this->as_mux) {
        uint16_t channel =
            tt::Cluster::instance().get_assigned_channel_for_device(this->servicing_device_id);  // TODO: can be mmio
        this->logical_core =
            dispatch_core_manager::instance().mux_core(this->servicing_device_id, channel, placement_cq_id);
        this->config.rx_queue_start_addr_words = my_dispatch_constants.dispatch_buffer_base() >> 4;
        // TODO: why is this hard-coded NUM_CQS=1 for galaxy?
        if (tt::Cluster::instance().is_galaxy_cluster()) {
            this->config.rx_queue_size_words = my_dispatch_constants.mux_buffer_size(1) >> 4;
        } else {
            this->config.rx_queue_size_words = my_dispatch_constants.mux_buffer_size(device->num_hw_cqs()) >> 4;
        }

        this->config.kernel_status_buf_addr_arg = 0;
        this->config.kernel_status_buf_size_bytes = 0;
        this->config.timeout_cycles = 0;
        this->config.output_depacketize = {0x0};
        this->config.output_depacketize_log_page_size = {0x0};
        this->config.output_depacketize_downstream_sem = {0x0};
        this->config.output_depacketize_local_sem = {0x0};
        this->config.output_depacketize_remove_header = {0x0};

        for (int idx = 0; idx < this->upstream_kernels.size(); idx++) {
            this->config.input_packetize[idx] = 0x1;
            this->config.input_packetize_log_page_size[idx] = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
            this->config.input_packetize_local_sem[idx] =
                tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
            this->config.remote_rx_queue_id[idx] = 1;
        }
        // Mux fowrads all VCs
        this->config.fwd_vc_count = this->config.vc_count;
    } else {
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device->id());
        this->logical_core = dispatch_core_manager::instance().demux_d_core(device->id(), channel, placement_cq_id);
        this->config.rx_queue_start_addr_words =
            hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED) >> 4;
        this->config.rx_queue_size_words = 0x8000 >> 4;

        this->config.kernel_status_buf_addr_arg = 0;
        this->config.kernel_status_buf_size_bytes = 0;
        this->config.timeout_cycles = 0;
        this->config.output_depacketize = {0x0};

        this->config.input_packetize = {0x0};
        this->config.input_packetize_log_page_size = {0x0};
        this->config.input_packetize_upstream_sem = {0x0};
        this->config.input_packetize_local_sem = {0x0};
        this->config.input_packetize_src_endpoint = {0x0};
        this->config.input_packetize_dst_endpoint = {0x0};

        this->config.fwd_vc_count = this->config.vc_count;
        uint32_t created_semaphores = 0;
        for (int idx = 0; idx < this->downstream_kernels.size(); idx++) {
            // Forwward VCs are the ones that don't connect to a prefetch
            if (auto pk = dynamic_cast<PrefetchKernel*>(this->downstream_kernels[idx])) {
                this->config.fwd_vc_count = this->config.fwd_vc_count.value() - 1;
                this->config.output_depacketize_local_sem[idx] =  // TODO: to match for now, init one per vc after
                    tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
                created_semaphores++;
            }
        }
        if (created_semaphores == 0) {  // Just to match previous implementation
            tt::tt_metal::CreateSemaphore(*program, this->logical_core, 0, GetCoreType());
        }

        for (int idx = 0; idx < this->config.vc_count.value(); idx++) {
            this->config.output_depacketize_log_page_size[idx] = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
            this->config.output_depacketize_remove_header[idx] = 0;
        }
    }
}

void EthRouterKernel::GenerateDependentConfigs() {
    if (this->as_mux) {
        // Upstream, expect PRETETCH_Hs
        TT_ASSERT(this->upstream_kernels.size() <= MAX_SWITCH_FAN_IN && this->upstream_kernels.size() > 0);

        // Downstream, expect US_TUNNELER_REMOTE
        TT_ASSERT(this->downstream_kernels.size() == 1);
        auto tunneler_kernel = dynamic_cast<EthTunnelerKernel*>(this->downstream_kernels[0]);
        TT_ASSERT(tunneler_kernel);

        uint32_t router_id = tunneler_kernel->GetRouterId(this, true);
        for (int idx = 0; idx < this->upstream_kernels.size(); idx++) {
            auto prefetch_kernel = dynamic_cast<PrefetchKernel*>(this->upstream_kernels[idx]);
            TT_ASSERT(prefetch_kernel);
            this->config.remote_tx_x[idx] = tunneler_kernel->GetVirtualCore().x;
            this->config.remote_tx_y[idx] = tunneler_kernel->GetVirtualCore().y;
            this->config.remote_tx_queue_id[idx] = idx + MAX_SWITCH_FAN_IN * router_id;
            this->config.remote_tx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
            this->config.remote_tx_queue_start_addr_words[idx] =
                tunneler_kernel->GetConfig().in_queue_start_addr_words.value() +
                (idx + router_id * MAX_SWITCH_FAN_IN) * tunneler_kernel->GetConfig().in_queue_size_words.value();
            this->config.remote_tx_queue_size_words[idx] = tunneler_kernel->GetConfig().in_queue_size_words.value();

            this->config.remote_rx_x[idx] = prefetch_kernel->GetVirtualCore().x;
            this->config.remote_rx_y[idx] = prefetch_kernel->GetVirtualCore().y;
            this->config.remote_rx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;

            this->config.input_packetize_upstream_sem[idx] =
                prefetch_kernel->GetConfig().my_downstream_cb_sem_id.value();
        }

        uint32_t src_id_start = 0xA1 + router_id * MAX_SWITCH_FAN_IN;
        uint32_t dst_id_start = 0xB1 + router_id * MAX_SWITCH_FAN_IN;
        this->config.input_packetize_src_endpoint = {
            src_id_start, src_id_start + 1, src_id_start + 2, src_id_start + 3};
        this->config.input_packetize_dst_endpoint = {
            dst_id_start, dst_id_start + 1, dst_id_start + 2, dst_id_start + 3};
    } else {
        // Upstream, expect US_TUNNELER_LOCAL
        TT_ASSERT(this->upstream_kernels.size() == 1);
        auto us_tunneler_kernel = dynamic_cast<EthTunnelerKernel*>(this->upstream_kernels[0]);
        TT_ASSERT(us_tunneler_kernel);
        // Upstream queues connect to the upstream tunneler, as many queues as we have VCs
        for (int idx = 0; idx < config.vc_count.value(); idx++) {
            this->config.remote_rx_x[idx] = us_tunneler_kernel->GetVirtualCore().x;
            this->config.remote_rx_y[idx] = us_tunneler_kernel->GetVirtualCore().y;
            // Queue id starts counting after the input VCs
            this->config.remote_rx_queue_id[idx] = us_tunneler_kernel->GetRouterQueueIdOffset(this, false) + idx;
            this->config.remote_rx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
        }

        // Downstream, expect PREFETCH_D/US_TUNNELER_REMOTE
        TT_ASSERT(this->downstream_kernels.size() <= MAX_SWITCH_FAN_OUT && this->downstream_kernels.size() > 0);
        std::vector<PrefetchKernel*> prefetch_kernels;
        EthTunnelerKernel* ds_tunneler_kernel = nullptr;
        for (auto k : this->downstream_kernels) {
            if (auto pk = dynamic_cast<PrefetchKernel*>(k)) {
                prefetch_kernels.push_back(pk);
            } else if (auto tk = dynamic_cast<EthTunnelerKernel*>(k)) {
                ds_tunneler_kernel = tk;
            } else {
                TT_FATAL(false, "Unexpected kernel type downstream of ROUTER");
            }
        }

        // Populate remote_tx_* for prefetch kernels, assume they are connected "first"
        uint32_t remote_idx = 0;
        for (auto prefetch_kernel : prefetch_kernels) {
            this->config.remote_tx_x[remote_idx] = prefetch_kernel->GetVirtualCore().x;
            this->config.remote_tx_y[remote_idx] = prefetch_kernel->GetVirtualCore().y;
            this->config.remote_tx_queue_id[remote_idx] = 0;  // Prefetch queue id always 0
            this->config.remote_tx_network_type[remote_idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
            this->config.remote_tx_queue_start_addr_words[remote_idx] =
                prefetch_kernel->GetConfig().cmddat_q_base.value() >> 4;
            this->config.remote_tx_queue_size_words[remote_idx] =
                prefetch_kernel->GetConfig().cmddat_q_size.value() >> 4;
            this->config.output_depacketize[remote_idx] = 1;
            this->config.output_depacketize_downstream_sem[remote_idx] =
                prefetch_kernel->GetConfig().my_upstream_cb_sem_id;
            remote_idx++;
        }

        // Populate remote_tx_* for the downstream tunneler, as many queues as we have fwd VCs
        if (ds_tunneler_kernel) {
            for (int idx = 0; idx < config.fwd_vc_count.value(); idx++) {
                this->config.remote_tx_x[remote_idx] = ds_tunneler_kernel->GetVirtualCore().x;
                this->config.remote_tx_y[remote_idx] = ds_tunneler_kernel->GetVirtualCore().y;
                this->config.remote_tx_queue_id[remote_idx] =
                    ds_tunneler_kernel->GetRouterQueueIdOffset(this, true) + idx;
                this->config.remote_tx_network_type[remote_idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
                this->config.remote_tx_queue_start_addr_words[remote_idx] =
                    ds_tunneler_kernel->GetConfig().in_queue_start_addr_words.value() +
                    ds_tunneler_kernel->GetConfig().in_queue_size_words.value() *
                        (this->config.remote_tx_queue_id[remote_idx].value());
                this->config.remote_tx_queue_size_words[remote_idx] =
                    ds_tunneler_kernel->GetConfig().in_queue_size_words.value();
                // Don't depacketize when sending to tunneler
                this->config.output_depacketize[remote_idx] = 0;
                this->config.output_depacketize_downstream_sem[remote_idx] = 0;
                remote_idx++;
            }
        }
    }
}

void EthRouterKernel::CreateKernel() {
    std::vector<uint32_t> compile_args{
        0,  // Unused
        config.rx_queue_start_addr_words.value(),
        config.rx_queue_size_words.value(),
        config.vc_count.value(),
        0,
        0,
        0,
        0,  // Populate remote_tx_* after
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  // Populate remote_tx_queue_* after
        0,
        0,
        0,
        0,  // Populate remote_rx_* after
        0,
        0,  // Unused
        config.kernel_status_buf_addr_arg.value(),
        config.kernel_status_buf_size_bytes.value(),
        config.timeout_cycles.value(),
        0,  // Populate output_depacketize after
        0,
        0,
        0,
        0,  // Populate output_depacketize_* after
        0,
        0,
        0,
        0,  // Populate input_packetize_* afterA
        0,  // input_packetize_src_endpoint
        0,  // input_packetize_dst_endpoint
    };
    // Some unused values, just hardcode them to match for checking purposes...
    if (!this->as_mux) {
        compile_args[0] = 0xB1;
        // compile_args[21] = 84;
    }
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
        if (config.output_depacketize[idx]) {
            compile_args[25] |= (config.output_depacketize[idx].value() & 0x1) << idx;
            if (config.output_depacketize[idx].value() & 0x1) {  // To match previous implementation
                compile_args[26 + idx] |= (config.output_depacketize_log_page_size[idx].value() & 0xFF);
                compile_args[26 + idx] |= (config.output_depacketize_downstream_sem[idx].value() & 0xFF) << 8;
                compile_args[26 + idx] |= (config.output_depacketize_local_sem[idx].value() & 0xFF) << 16;
                compile_args[26 + idx] |= (config.output_depacketize_remove_header[idx].value() & 0xFF) << 24;
            }
        }
    }
    for (int idx = 0; idx < MAX_SWITCH_FAN_IN; idx++) {
        if (config.remote_rx_x[idx]) {
            compile_args[16 + idx] |= (config.remote_rx_x[idx].value() & 0xFF);
            compile_args[16 + idx] |= (config.remote_rx_y[idx].value() & 0xFF) << 8;
            compile_args[16 + idx] |= (config.remote_rx_queue_id[idx].value() & 0xFF) << 16;
            compile_args[16 + idx] |= (config.remote_rx_network_type[idx].value() & 0xFF) << 24;
        }
        if (config.input_packetize[idx]) {
            compile_args[30 + idx] |= (config.input_packetize[idx].value() & 0xFF);
            compile_args[30 + idx] |= (config.input_packetize_log_page_size[idx].value() & 0xFF) << 8;
            compile_args[30 + idx] |= (config.input_packetize_upstream_sem[idx].value() & 0xFF) << 16;
            compile_args[30 + idx] |= (config.input_packetize_local_sem[idx].value() & 0xFF) << 24;
        }
        if (config.input_packetize_src_endpoint[idx]) {
            compile_args[34] |= (config.input_packetize_src_endpoint[idx].value() & 0xFF) << (8 * idx);
        }
        if (config.input_packetize_dst_endpoint[idx]) {
            compile_args[35] |= (config.input_packetize_dst_endpoint[idx].value() & 0xFF) << (8 * idx);
        }
    }
    TT_ASSERT(compile_args.size() == 36);
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
    configure_kernel_variant(dispatch_kernel_file_names[PACKET_ROUTER_MUX], compile_args, defines, false, false, false);
}
