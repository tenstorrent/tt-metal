// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "demux.hpp"

#include <host_api.hpp>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "assert.hpp"
#include "device.hpp"
#include "dispatch.hpp"
#include "impl/context/metal_context.hpp"
#include "dispatch/kernel_config/fd_kernel.hpp"
#include "dispatch_core_common.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "eth_tunneler.hpp"
#include <umd/device/tt_xy_pair.h>

using namespace tt::tt_metal;

void DemuxKernel::GenerateStaticConfigs() {
    auto& my_dispatch_constants = MetalContext::instance().dispatch_mem_map(GetCoreType());
    uint16_t channel = tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(
        servicing_device_id_);  // TODO: this can be mmio
    logical_core_ = MetalContext::instance().get_dispatch_core_manager().demux_core(
        servicing_device_id_, channel, placement_cq_id_);
    kernel_type_ = FDKernelType::ROUTING;

    static_config_.endpoint_id_start_index = 0xD1;
    static_config_.rx_queue_start_addr_words = my_dispatch_constants.dispatch_buffer_base() >> 4;
    static_config_.rx_queue_size_words = 0x10000 >> 4;
    static_config_.demux_fan_out = downstream_kernels_.size();

    static_config_.remote_rx_network_type = tt::packet_queue::DispatchRemoteNetworkType::NOC0;

    static_config_.test_results_buf_addr_arg = 0;
    static_config_.test_results_buf_size_bytes = 0;
    static_config_.timeout_cycles = 0;

    for (int idx = 0; idx < downstream_kernels_.size(); idx++) {
        static_config_.remote_tx_queue_id[idx] = 0;
        static_config_.remote_tx_network_type[idx] = (uint32_t)tt::packet_queue::DispatchRemoteNetworkType::NOC0;
        static_config_.output_depacketize_cb_log_page_size[idx] = DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE;
        static_config_.output_depacketize_local_sem_id[idx] =
            tt::tt_metal::CreateSemaphore(*program_, logical_core_, 0, GetCoreType());
        static_config_.output_depacketize_remove_header[idx] = 1;
    }
}

void DemuxKernel::GenerateDependentConfigs() {
    // Upstream, expect EthTunneler or DEMUX
    TT_ASSERT(upstream_kernels_.size() == 1);
    if (auto us = dynamic_cast<EthTunnelerKernel*>(upstream_kernels_[0])) {
        dependent_config_.remote_rx_x = us->GetVirtualCore().x;
        dependent_config_.remote_rx_y = us->GetVirtualCore().y;
        dependent_config_.remote_rx_queue_id = us->GetStaticConfig().vc_count.value() * 2 - 1;
    } else if (auto us = dynamic_cast<DemuxKernel*>(upstream_kernels_[0])) {
        dependent_config_.remote_rx_x = us->GetVirtualCore().x;
        dependent_config_.remote_rx_y = us->GetVirtualCore().y;
        dependent_config_.remote_rx_queue_id = us->GetDownstreamPort(this) + 1;  // TODO: can this be cleaned up?
        // TODO: why is just this one different? Just match previous implementation for now
        if (us->GetDownstreamPort(this) == 1) {
            static_config_.endpoint_id_start_index =
                static_config_.endpoint_id_start_index.value() + downstream_kernels_.size();
        }
    } else {
        TT_FATAL(false, "Unexpected kernel type upstream of DEMUX");
    }

    // Downstream, expect DISPATCH_H or DEMUX
    TT_ASSERT(downstream_kernels_.size() <= tt::packet_queue::MAX_SWITCH_FAN_OUT && downstream_kernels_.size() > 0);
    dependent_config_.output_depacketize = 0;  // Populated per downstream kernel
    for (int idx = 0; idx < downstream_kernels_.size(); idx++) {
        FDKernel* k = downstream_kernels_[idx];
        dependent_config_.remote_tx_x[idx] = k->GetVirtualCore().x;
        dependent_config_.remote_tx_y[idx] = k->GetVirtualCore().y;
        // Expect downstream to be either a DISPATCH or another DEMUX
        if (auto dispatch_kernel = dynamic_cast<DispatchKernel*>(k)) {
            dependent_config_.remote_tx_queue_start_addr_words[idx] =
                dispatch_kernel->GetStaticConfig().dispatch_cb_base.value() >> 4;
            dependent_config_.remote_tx_queue_size_words[idx] = dispatch_kernel->GetDispatchBufferSize() >> 4;
            dependent_config_.output_depacketize =
                dependent_config_.output_depacketize.value() | (1 << idx);  // Only depacketize for dispatch downstream
            dependent_config_.output_depacketize_downstream_sem_id[idx] =
                dispatch_kernel->GetStaticConfig().my_dispatch_cb_sem_id;
            uint32_t dest_map_array[4] = {0, 1, 2, 3};  // TODO: how to set these generically? Currently just matching
                                                        // the hard-coded previous implementation
            uint64_t dest_endpoint_output_map = tt::packet_queue::packet_switch_dest_pack(dest_map_array, 4);
            dependent_config_.dest_endpoint_output_map_hi = (uint32_t)(dest_endpoint_output_map >> 32);
            dependent_config_.dest_endpoint_output_map_lo = (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF);
        } else if (auto demux_kernel = dynamic_cast<DemuxKernel*>(k)) {
            dependent_config_.remote_tx_queue_start_addr_words[idx] =
                demux_kernel->GetStaticConfig().rx_queue_start_addr_words.value();
            dependent_config_.remote_tx_queue_size_words[idx] = 0x1000;  // TODO: hard-coded on previous implementation
            uint64_t dest_endpoint_output_map;
            if (device_->num_hw_cqs() == 1) {
                uint32_t dest_map_array[4] = {0, 0, 1, 1};  // TODO: how to set these generically? Currently just
                                                            // matching the hard-coded previous implementation
                dest_endpoint_output_map = tt::packet_queue::packet_switch_dest_pack(dest_map_array, 4);
            } else {
                uint32_t dest_map_array[8] = {0, 0, 0, 0, 1, 1, 1, 1};
                dest_endpoint_output_map = tt::packet_queue::packet_switch_dest_pack(dest_map_array, 8);
            }
            dependent_config_.dest_endpoint_output_map_hi = (uint32_t)(dest_endpoint_output_map >> 32);
            dependent_config_.dest_endpoint_output_map_lo = (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF);
        } else {
            TT_FATAL(false, "Unexpected kernel type downstream of DEMUX");
        }
    }
}

void DemuxKernel::CreateKernel() {
    std::vector<uint32_t> compile_args = {
        static_config_.endpoint_id_start_index.value(),
        static_config_.rx_queue_start_addr_words.value(),
        static_config_.rx_queue_size_words.value(),
        static_config_.demux_fan_out.value(),
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
        dependent_config_.remote_rx_x.value(),
        dependent_config_.remote_rx_y.value(),
        dependent_config_.remote_rx_queue_id.value(),
        static_config_.remote_rx_network_type.value(),
        dependent_config_.dest_endpoint_output_map_hi.value(),
        dependent_config_.dest_endpoint_output_map_lo.value(),
        static_config_.test_results_buf_addr_arg.value(),
        static_config_.test_results_buf_size_bytes.value(),
        static_config_.timeout_cycles.value(),
        dependent_config_.output_depacketize.value(),
        0,
        0,
        0,
        0  // Populate output_depacketize_config after
    };
    for (int idx = 0; idx < tt::packet_queue::MAX_SWITCH_FAN_OUT; idx++) {
        if (dependent_config_.remote_tx_x[idx]) {
            compile_args[4 + idx] |= (dependent_config_.remote_tx_x[idx].value() & 0xFF);
            compile_args[4 + idx] |= (dependent_config_.remote_tx_y[idx].value() & 0xFF) << 8;
            compile_args[4 + idx] |= (static_config_.remote_tx_queue_id[idx].value() & 0xFF) << 16;
            compile_args[4 + idx] |= (static_config_.remote_tx_network_type[idx].value() & 0xFF) << 24;
        }
        if (dependent_config_.remote_tx_queue_start_addr_words[idx]) {
            compile_args[8 + idx * 2] = dependent_config_.remote_tx_queue_start_addr_words[idx].value();
            compile_args[9 + idx * 2] = dependent_config_.remote_tx_queue_size_words[idx].value();
        }
        if (static_config_.output_depacketize_cb_log_page_size[idx]) {
            // To match previous implementation, zero these out if output_depacketize is not set. TODO: don't have to do
            // this
            if (dependent_config_.output_depacketize.value() & (1 << idx)) {
                compile_args[26 + idx] |= (static_config_.output_depacketize_cb_log_page_size[idx].value() & 0xFF);
                compile_args[26 + idx] |= (dependent_config_.output_depacketize_downstream_sem_id[idx].value() & 0xFF)
                                          << 8;
                compile_args[26 + idx] |= (static_config_.output_depacketize_local_sem_id[idx].value() & 0xFF) << 16;
                compile_args[26 + idx] |= (static_config_.output_depacketize_remove_header[idx].value() & 0xFF) << 24;
            }
        }
    }
    TT_ASSERT(compile_args.size() == 30);
    const auto& grid_size = device_->grid_size();
    const auto& hal = MetalContext::instance().hal();
    std::map<std::string, std::string> defines = {
        // All of these unused, remove later
        {"MY_NOC_X", std::to_string(hal.noc_coordinate(noc_selection_.non_dispatch_noc, grid_size.x, 0))},
        {"MY_NOC_Y", std::to_string(hal.noc_coordinate(noc_selection_.non_dispatch_noc, grid_size.y, 0))},
        {"UPSTREAM_NOC_INDEX", std::to_string(noc_selection_.upstream_noc)},
        {"UPSTREAM_NOC_X", std::to_string(hal.noc_coordinate(noc_selection_.upstream_noc, grid_size.x, 0))},
        {"UPSTREAM_NOC_Y", std::to_string(hal.noc_coordinate(noc_selection_.upstream_noc, grid_size.y, 0))},
        {"DOWNSTREAM_NOC_X", std::to_string(hal.noc_coordinate(noc_selection_.downstream_noc, grid_size.x, 0))},
        {"DOWNSTREAM_NOC_Y", std::to_string(hal.noc_coordinate(noc_selection_.downstream_noc, grid_size.y, 0))},
        {"DOWNSTREAM_SUBORDINATE_NOC_X",
         std::to_string(hal.noc_coordinate(noc_selection_.downstream_noc, grid_size.x, 0))},
        {"DOWNSTREAM_SUBORDINATE_NOC_Y",
         std::to_string(hal.noc_coordinate(noc_selection_.downstream_noc, grid_size.y, 0))},
        {"SKIP_NOC_LOGGING", "1"}};
    configure_kernel_variant(dispatch_kernel_file_names[DEMUX], compile_args, defines, false, false, false);
}
