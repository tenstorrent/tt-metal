// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "eth_router.hpp"
#include "prefetch.hpp"
#include "eth_tunneler.hpp"

#include <host_api.hpp>
#include <tt_metal.hpp>

using namespace tt::tt_metal;

void EthRouterKernel::GenerateStaticConfigs() {
    auto& my_dispatch_constants = dispatch_constants::get(GetCoreType());
    if (as_mux_) {
        uint16_t channel =
            tt::Cluster::instance().get_assigned_channel_for_device(servicing_device_id_);  // TODO: can be mmio
        logical_core_ = dispatch_core_manager::instance().mux_core(servicing_device_id_, channel, placement_cq_id_);
        static_config_.rx_queue_start_addr_words = my_dispatch_constants.dispatch_buffer_base() >> 4;
        // TODO: why is this hard-coded NUM_CQS=1 for galaxy?
        if (tt::Cluster::instance().is_galaxy_cluster()) {
            static_config_.rx_queue_size_words = my_dispatch_constants.mux_buffer_size(1) >> 4;
        } else {
            static_config_.rx_queue_size_words = my_dispatch_constants.mux_buffer_size(device_->num_hw_cqs()) >> 4;
        }

        static_config_.kernel_status_buf_addr_arg = 0;
        static_config_.kernel_status_buf_size_bytes = 0;
        static_config_.timeout_cycles = 0;
        dependent_config_.output_depacketize = {0x0};
        static_config_.output_depacketize_log_page_size = {0x0};
        dependent_config_.output_depacketize_downstream_sem = {0x0};
        static_config_.output_depacketize_local_sem = {0x0};
        static_config_.output_depacketize_remove_header = {0x0};

        for (int idx = 0; idx < upstream_kernels_.size(); idx++) {
            static_config_.input_packetize[idx] = 0x1;
            static_config_.input_packetize_log_page_size[idx] = dispatch_constants::DISPATCH_BUFFER_LOG_PAGE_SIZE;
            static_config_.input_packetize_local_sem[idx] =
                tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
            dependent_config_.remote_rx_queue_id[idx] = 1;
        }
        // Mux fowrads all VCs
        static_config_.fwd_vc_count = this->static_config_.vc_count;
    } else {
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_->id());
        logical_core_ = dispatch_core_manager::instance().demux_d_core(device_->id(), channel, placement_cq_id_);
        static_config_.rx_queue_start_addr_words =
            hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED) >> 4;
        static_config_.rx_queue_size_words = 0x8000 >> 4;

        static_config_.kernel_status_buf_addr_arg = 0;
        static_config_.kernel_status_buf_size_bytes = 0;
        static_config_.timeout_cycles = 0;
        dependent_config_.output_depacketize = {0x0};

        static_config_.input_packetize = {0x0};
        static_config_.input_packetize_log_page_size = {0x0};
        dependent_config_.input_packetize_upstream_sem = {0x0};
        static_config_.input_packetize_local_sem = {0x0};
        dependent_config_.input_packetize_src_endpoint = {0x0};
        dependent_config_.input_packetize_dst_endpoint = {0x0};

        static_config_.fwd_vc_count = this->static_config_.vc_count;
        for (int idx = 0; idx < downstream_kernels_.size(); idx++) {
            static_config_.output_depacketize_local_sem[idx] =
                tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
            // Forwward VCs are the ones that don't connect to a prefetch
            if (auto pk = dynamic_cast<PrefetchKernel*>(downstream_kernels_[idx])) {
                static_config_.fwd_vc_count = this->static_config_.fwd_vc_count.value() - 1;
            }
        }

        for (int idx = 0; idx < static_config_.vc_count.value(); idx++) {
            static_config_.output_depacketize_log_page_size[idx] = dispatch_constants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE;
            static_config_.output_depacketize_remove_header[idx] = 0;
        }
    }
}

void EthRouterKernel::GenerateDependentConfigs() {
    if (as_mux_) {
        // Upstream, expect PRETETCH_Hs
        TT_ASSERT(upstream_kernels_.size() <= MAX_SWITCH_FAN_IN && upstream_kernels_.size() > 0);

        // Downstream, expect US_TUNNELER_REMOTE
        TT_ASSERT(downstream_kernels_.size() == 1);
        auto tunneler_kernel = dynamic_cast<EthTunnelerKernel*>(downstream_kernels_[0]);
        TT_ASSERT(tunneler_kernel);

        uint32_t router_id = tunneler_kernel->GetRouterId(this, true);
        for (int idx = 0; idx < upstream_kernels_.size(); idx++) {
            auto prefetch_kernel = dynamic_cast<PrefetchKernel*>(upstream_kernels_[idx]);
            TT_ASSERT(prefetch_kernel);
            dependent_config_.remote_tx_x[idx] = tunneler_kernel->GetVirtualCore().x;
            dependent_config_.remote_tx_y[idx] = tunneler_kernel->GetVirtualCore().y;
            dependent_config_.remote_tx_queue_id[idx] = idx + MAX_SWITCH_FAN_IN * router_id;
            dependent_config_.remote_tx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
            dependent_config_.remote_tx_queue_start_addr_words[idx] =
                tunneler_kernel->GetStaticConfig().in_queue_start_addr_words.value() +
                (idx + router_id * MAX_SWITCH_FAN_IN) * tunneler_kernel->GetStaticConfig().in_queue_size_words.value();
            dependent_config_.remote_tx_queue_size_words[idx] =
                tunneler_kernel->GetStaticConfig().in_queue_size_words.value();

            dependent_config_.remote_rx_x[idx] = prefetch_kernel->GetVirtualCore().x;
            dependent_config_.remote_rx_y[idx] = prefetch_kernel->GetVirtualCore().y;
            dependent_config_.remote_rx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;

            dependent_config_.input_packetize_upstream_sem[idx] =
                prefetch_kernel->GetStaticConfig().my_downstream_cb_sem_id.value();
        }

        uint32_t src_id_start = 0xA1 + router_id * MAX_SWITCH_FAN_IN;
        uint32_t dst_id_start = 0xB1 + router_id * MAX_SWITCH_FAN_IN;
        dependent_config_.input_packetize_src_endpoint = {
            src_id_start, src_id_start + 1, src_id_start + 2, src_id_start + 3};
        dependent_config_.input_packetize_dst_endpoint = {
            dst_id_start, dst_id_start + 1, dst_id_start + 2, dst_id_start + 3};
    } else {
        // Upstream, expect US_TUNNELER_LOCAL
        TT_ASSERT(upstream_kernels_.size() == 1);
        auto us_tunneler_kernel = dynamic_cast<EthTunnelerKernel*>(upstream_kernels_[0]);
        TT_ASSERT(us_tunneler_kernel);
        // Upstream queues connect to the upstream tunneler, as many queues as we have VCs
        for (int idx = 0; idx < static_config_.vc_count.value(); idx++) {
            dependent_config_.remote_rx_x[idx] = us_tunneler_kernel->GetVirtualCore().x;
            dependent_config_.remote_rx_y[idx] = us_tunneler_kernel->GetVirtualCore().y;
            // Queue id starts counting after the input VCs
            dependent_config_.remote_rx_queue_id[idx] = us_tunneler_kernel->GetRouterQueueIdOffset(this, false) + idx;
            dependent_config_.remote_rx_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
        }

        // Downstream, expect PREFETCH_D/US_TUNNELER_REMOTE
        TT_ASSERT(downstream_kernels_.size() <= MAX_SWITCH_FAN_OUT && downstream_kernels_.size() > 0);
        std::vector<PrefetchKernel*> prefetch_kernels;
        EthTunnelerKernel* ds_tunneler_kernel = nullptr;
        for (auto k : downstream_kernels_) {
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
            dependent_config_.remote_tx_x[remote_idx] = prefetch_kernel->GetVirtualCore().x;
            dependent_config_.remote_tx_y[remote_idx] = prefetch_kernel->GetVirtualCore().y;
            dependent_config_.remote_tx_queue_id[remote_idx] = 0;  // Prefetch queue id always 0
            dependent_config_.remote_tx_network_type[remote_idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
            dependent_config_.remote_tx_queue_start_addr_words[remote_idx] =
                prefetch_kernel->GetStaticConfig().cmddat_q_base.value() >> 4;
            dependent_config_.remote_tx_queue_size_words[remote_idx] =
                prefetch_kernel->GetStaticConfig().cmddat_q_size.value() >> 4;
            dependent_config_.output_depacketize[remote_idx] = 1;
            dependent_config_.output_depacketize_downstream_sem[remote_idx] =
                prefetch_kernel->GetStaticConfig().my_upstream_cb_sem_id;
            remote_idx++;
        }

        // Populate remote_tx_* for the downstream tunneler, as many queues as we have fwd VCs
        if (ds_tunneler_kernel) {
            for (int idx = 0; idx < static_config_.fwd_vc_count.value(); idx++) {
                dependent_config_.remote_tx_x[remote_idx] = ds_tunneler_kernel->GetVirtualCore().x;
                dependent_config_.remote_tx_y[remote_idx] = ds_tunneler_kernel->GetVirtualCore().y;
                dependent_config_.remote_tx_queue_id[remote_idx] =
                    ds_tunneler_kernel->GetRouterQueueIdOffset(this, true) + idx;
                dependent_config_.remote_tx_network_type[remote_idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
                dependent_config_.remote_tx_queue_start_addr_words[remote_idx] =
                    ds_tunneler_kernel->GetStaticConfig().in_queue_start_addr_words.value() +
                    ds_tunneler_kernel->GetStaticConfig().in_queue_size_words.value() *
                        (dependent_config_.remote_tx_queue_id[remote_idx].value());
                dependent_config_.remote_tx_queue_size_words[remote_idx] =
                    ds_tunneler_kernel->GetStaticConfig().in_queue_size_words.value();
                // Don't depacketize when sending to tunneler
                dependent_config_.output_depacketize[remote_idx] = 0;
                dependent_config_.output_depacketize_downstream_sem[remote_idx] = 0;
                remote_idx++;
            }
        }
    }
}

void EthRouterKernel::CreateKernel() {
    std::vector<uint32_t> compile_args{
        0,  // Unused
        static_config_.rx_queue_start_addr_words.value(),
        static_config_.rx_queue_size_words.value(),
        static_config_.vc_count.value(),
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
        static_config_.kernel_status_buf_addr_arg.value(),
        static_config_.kernel_status_buf_size_bytes.value(),
        static_config_.timeout_cycles.value(),
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
    if (!as_mux_) {
        compile_args[0] = 0xB1;
        // compile_args[21] = 84;
    }
    for (int idx = 0; idx < MAX_SWITCH_FAN_OUT; idx++) {
        if (dependent_config_.remote_tx_x[idx]) {
            compile_args[4 + idx] |= (dependent_config_.remote_tx_x[idx].value() & 0xFF);
            compile_args[4 + idx] |= (dependent_config_.remote_tx_y[idx].value() & 0xFF) << 8;
            compile_args[4 + idx] |= (dependent_config_.remote_tx_queue_id[idx].value() & 0xFF) << 16;
            compile_args[4 + idx] |= (dependent_config_.remote_tx_network_type[idx].value() & 0xFF) << 24;
        }
        if (dependent_config_.remote_tx_queue_start_addr_words[idx]) {
            compile_args[8 + idx * 2] = dependent_config_.remote_tx_queue_start_addr_words[idx].value();
            compile_args[9 + idx * 2] = dependent_config_.remote_tx_queue_size_words[idx].value();
        }
        if (dependent_config_.output_depacketize[idx]) {
            compile_args[25] |= (dependent_config_.output_depacketize[idx].value() & 0x1) << idx;
            if (dependent_config_.output_depacketize[idx].value() & 0x1) {  // To match previous implementation
                compile_args[26 + idx] |= (static_config_.output_depacketize_log_page_size[idx].value() & 0xFF);
                compile_args[26 + idx] |= (dependent_config_.output_depacketize_downstream_sem[idx].value() & 0xFF)
                                          << 8;
                compile_args[26 + idx] |= (static_config_.output_depacketize_local_sem[idx].value() & 0xFF) << 16;
                compile_args[26 + idx] |= (static_config_.output_depacketize_remove_header[idx].value() & 0xFF) << 24;
            }
        }
    }
    for (int idx = 0; idx < MAX_SWITCH_FAN_IN; idx++) {
        if (dependent_config_.remote_rx_x[idx]) {
            compile_args[16 + idx] |= (dependent_config_.remote_rx_x[idx].value() & 0xFF);
            compile_args[16 + idx] |= (dependent_config_.remote_rx_y[idx].value() & 0xFF) << 8;
            compile_args[16 + idx] |= (dependent_config_.remote_rx_queue_id[idx].value() & 0xFF) << 16;
            compile_args[16 + idx] |= (dependent_config_.remote_rx_network_type[idx].value() & 0xFF) << 24;
        }
        if (static_config_.input_packetize[idx]) {
            compile_args[30 + idx] |= (static_config_.input_packetize[idx].value() & 0xFF);
            compile_args[30 + idx] |= (static_config_.input_packetize_log_page_size[idx].value() & 0xFF) << 8;
            compile_args[30 + idx] |= (dependent_config_.input_packetize_upstream_sem[idx].value() & 0xFF) << 16;
            compile_args[30 + idx] |= (static_config_.input_packetize_local_sem[idx].value() & 0xFF) << 24;
        }
        if (dependent_config_.input_packetize_src_endpoint[idx]) {
            compile_args[34] |= (dependent_config_.input_packetize_src_endpoint[idx].value() & 0xFF) << (8 * idx);
        }
        if (dependent_config_.input_packetize_dst_endpoint[idx]) {
            compile_args[35] |= (dependent_config_.input_packetize_dst_endpoint[idx].value() & 0xFF) << (8 * idx);
        }
    }
    TT_ASSERT(compile_args.size() == 36);
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
    configure_kernel_variant(dispatch_kernel_file_names[PACKET_ROUTER_MUX], compile_args, defines, false, false, false);
}
