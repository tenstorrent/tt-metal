// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "eth_tunneler.hpp"
#include "eth_router.hpp"
#include "demux.hpp"
#include "mux.hpp"

#include <host_api.hpp>
#include <tt_metal.hpp>

using namespace tt::tt_metal;

void EthTunnelerKernel::GenerateStaticConfigs() {
    chip_id_t downstream_device_id = FDKernel::GetDownstreamDeviceId(device_id_);
    // For MMIO devices, the above function just gets one of the possible downstream devices, we've populated this
    // specific case with servicing_device_id
    if (device_->is_mmio_capable()) {
        downstream_device_id = servicing_device_id_;
    }
    if (this->IsRemote()) {
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(downstream_device_id);
        logical_core_ =
            dispatch_core_manager::instance().tunneler_core(device_->id(), downstream_device_id, channel, cq_id_);
    } else {
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_->id());
        logical_core_ = dispatch_core_manager::instance().us_tunneler_core_local(device_->id(), channel, cq_id_);
    }
    static_config_.endpoint_id_start_index = 0xDACADACA;
    static_config_.in_queue_start_addr_words = 0x19000 >> 4;
    static_config_.in_queue_size_words = 0x4000 >> 4;
    static_config_.kernel_status_buf_addr_arg = 0x39000;
    static_config_.kernel_status_buf_size_bytes = 0x7000;
    static_config_.timeout_cycles = 0;
}

void EthTunnelerKernel::GenerateDependentConfigs() {
    if (this->IsRemote()) {
        // For remote tunneler, we don't actually have the device constructed for the paired tunneler, so can't pull
        // info from it. Core coord can be computed without the device, and relevant fields match this tunneler.
        chip_id_t downstream_device_id = FDKernel::GetDownstreamDeviceId(device_id_);
        uint16_t downstream_channel = tt::Cluster::instance().get_assigned_channel_for_device(downstream_device_id);
        tt_cxy_pair paired_logical_core =
            dispatch_core_manager::instance().us_tunneler_core_local(downstream_device_id, downstream_channel, cq_id_);
        tt_cxy_pair paired_physical_coord =
            tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(paired_logical_core, CoreType::ETH);

        // Upstream, we expect a US_TUNNELER_LOCAL and one or more PACKET_ROUTER
        EthTunnelerKernel* tunneler_kernel = nullptr;
        std::vector<EthRouterKernel*> router_kernels;
        for (auto k : upstream_kernels_) {
            if (auto rk = dynamic_cast<EthRouterKernel*>(k)) {
                router_kernels.push_back(rk);
            } else if (auto tk = dynamic_cast<EthTunnelerKernel*>(k)) {
                tunneler_kernel = tk;
            } else {
                TT_FATAL(false, "Unexpected kernel type upstream of TUNNELER");
            }
        }
        TT_ASSERT(tunneler_kernel && !tunneler_kernel->IsRemote());

        // Remote sender is the upstream packet router, one queue per router output lane.
        int remote_idx = 0;
        for (auto router_kernel : router_kernels) {
            uint32_t router_vc_count = router_kernel->GetStaticConfig().vc_count.value();
            uint32_t router_fwd_vc_count = router_kernel->GetStaticConfig().fwd_vc_count.value();
            for (int idx = 0; idx < router_fwd_vc_count; idx++) {
                dependent_config_.remote_sender_x[remote_idx] = router_kernel->GetVirtualCore().x;
                dependent_config_.remote_sender_y[remote_idx] = router_kernel->GetVirtualCore().y;
                // Router output lane ids start after it's input lane ids, assume after lanes that go to on-device
                // kernels
                dependent_config_.remote_sender_queue_id[remote_idx] =
                    router_vc_count + idx + router_vc_count - router_fwd_vc_count;
                dependent_config_.remote_sender_network_type[remote_idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
                remote_idx++;
            }
        }
        // Last upstream connection is the return path from other tunneler
        dependent_config_.remote_sender_x[this->static_config_.vc_count.value() - 1] = paired_physical_coord.x;
        dependent_config_.remote_sender_y[this->static_config_.vc_count.value() - 1] = paired_physical_coord.y;
        dependent_config_.remote_sender_queue_id[this->static_config_.vc_count.value() - 1] =
            this->static_config_.vc_count.value() * 2 - 1;
        dependent_config_.remote_sender_network_type[this->static_config_.vc_count.value() - 1] =
            (uint32_t)DispatchRemoteNetworkType::ETH;
        dependent_config_.inner_stop_mux_d_bypass = 0;

        // Downstream, we expect the same US_TUNNELER_LOCAL and a DEMUX (tunnel start)/MUX_D (non-tunnel start)
        TT_ASSERT(downstream_kernels_.size() == 2);
        auto ds_tunneler_kernel = dynamic_cast<EthTunnelerKernel*>(downstream_kernels_[0]);
        auto other_ds_kernel = downstream_kernels_[1];
        if (!ds_tunneler_kernel) {
            ds_tunneler_kernel = dynamic_cast<EthTunnelerKernel*>(downstream_kernels_[1]);
            auto other_ds_kernel = downstream_kernels_[0];
        }
        TT_ASSERT(ds_tunneler_kernel == tunneler_kernel);
        for (uint32_t idx = 0; idx < static_config_.vc_count.value(); idx++) {
            if (idx == static_config_.vc_count.value() - 1) {
                // Last VC is the return VC, driving a DEMUX or MUX_D
                dependent_config_.remote_receiver_x[idx] = other_ds_kernel->GetVirtualCore().x;
                dependent_config_.remote_receiver_y[idx] = other_ds_kernel->GetVirtualCore().y;
                dependent_config_.remote_receiver_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
                if (auto demux_kernel = dynamic_cast<DemuxKernel*>(other_ds_kernel)) {
                    dependent_config_.remote_receiver_queue_start[idx] =
                        demux_kernel->GetStaticConfig().rx_queue_start_addr_words;
                    dependent_config_.remote_receiver_queue_size[idx] =
                        demux_kernel->GetStaticConfig().rx_queue_size_words;
                    dependent_config_.remote_receiver_queue_id[idx] = 0;  // DEMUX input queue id always 0
                } else if (auto mux_kernel = dynamic_cast<MuxKernel*>(other_ds_kernel)) {
                    dependent_config_.remote_receiver_queue_start[idx] =
                        mux_kernel->GetStaticConfig().rx_queue_start_addr_words.value() +
                        mux_kernel->GetStaticConfig().rx_queue_size_words.value() *
                            (mux_kernel->GetStaticConfig().mux_fan_in.value() - 1);
                    dependent_config_.remote_receiver_queue_size[idx] =
                        mux_kernel->GetStaticConfig().rx_queue_size_words;
                    // MUX input queue id for tunneler is the last one (counting up from 0)
                    dependent_config_.remote_receiver_queue_id[idx] =
                        mux_kernel->GetStaticConfig().mux_fan_in.value() - 1;
                } else {
                    TT_FATAL(false, "Unexpected kernel type downstream of ETH_TUNNELER");
                }
            } else {
                dependent_config_.remote_receiver_x[idx] = paired_physical_coord.x;
                dependent_config_.remote_receiver_y[idx] = paired_physical_coord.y;
                // Tunneler upstream queue ids start counting up from 0
                dependent_config_.remote_receiver_queue_id[idx] = idx;
                dependent_config_.remote_receiver_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::ETH;
                dependent_config_.remote_receiver_queue_start[idx] =
                    static_config_.in_queue_start_addr_words.value() +
                    idx * this->static_config_.in_queue_size_words.value();
                dependent_config_.remote_receiver_queue_size[idx] = this->static_config_.in_queue_size_words;
            }
        }
    } else {
        // Upstream, we expect a US_TUNNELER_REMOTE and a MUX_D. Same deal where upstream tunneler may not be populated
        // yet since its device may not be created yet.
        chip_id_t upstream_device_id = FDKernel::GetUpstreamDeviceId(device_id_);
        uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_id_);
        tt_cxy_pair paired_logical_core =
            dispatch_core_manager::instance().tunneler_core(upstream_device_id, device_id_, channel, cq_id_);
        tt_cxy_pair paired_physical_coord =
            tt::Cluster::instance().get_virtual_coordinate_from_logical_coordinates(paired_logical_core, CoreType::ETH);

        TT_ASSERT(upstream_kernels_.size() == 2);
        auto tunneler_kernel = dynamic_cast<EthTunnelerKernel*>(upstream_kernels_[0]);
        auto mux_kernel = dynamic_cast<MuxKernel*>(upstream_kernels_[1]);
        if (!tunneler_kernel) {
            tunneler_kernel = dynamic_cast<EthTunnelerKernel*>(upstream_kernels_[1]);
            mux_kernel = dynamic_cast<MuxKernel*>(upstream_kernels_[0]);
        }
        TT_ASSERT(tunneler_kernel && mux_kernel);
        TT_ASSERT(tunneler_kernel->IsRemote());
        for (uint32_t idx = 0; idx < static_config_.vc_count.value(); idx++) {
            if (idx == static_config_.vc_count.value() - 1) {
                // Last VC is the return VC, driven by the mux
                dependent_config_.remote_sender_x[idx] = mux_kernel->GetVirtualCore().x;
                dependent_config_.remote_sender_y[idx] = mux_kernel->GetVirtualCore().y;
                // MUX output queue id is counted after all of it's inputs
                dependent_config_.remote_sender_queue_id[idx] = mux_kernel->GetStaticConfig().mux_fan_in.value();
                dependent_config_.remote_sender_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
            } else {
                dependent_config_.remote_sender_x[idx] = paired_physical_coord.x;
                dependent_config_.remote_sender_y[idx] = paired_physical_coord.y;
                // Tunneler downstream queue ids start counting after the upstream ones
                dependent_config_.remote_sender_queue_id[idx] = this->static_config_.vc_count.value() + idx;
                dependent_config_.remote_sender_network_type[idx] = (uint32_t)DispatchRemoteNetworkType::ETH;
            }
        }

        // Downstream, we expect the same US_TUNNELER_REMOTE and one or more VC_PACKER_ROUTER
        EthTunnelerKernel* ds_tunneler_kernel = nullptr;
        std::vector<EthRouterKernel*> router_kernels;
        for (auto k : downstream_kernels_) {
            if (auto rk = dynamic_cast<EthRouterKernel*>(k)) {
                router_kernels.push_back(rk);
            } else if (auto tk = dynamic_cast<EthTunnelerKernel*>(k)) {
                ds_tunneler_kernel = tk;
            } else {
                TT_FATAL(false, "Unexpected kernel type downstream of TUNNELER");
            }
        }
        TT_ASSERT(ds_tunneler_kernel && ds_tunneler_kernel == tunneler_kernel);

        // Remote receiver is the downstream router, one queue per router input lane
        int remote_idx = 0;
        for (auto router_kernel : router_kernels) {
            for (int idx = 0; idx < router_kernel->GetStaticConfig().vc_count.value(); idx++) {
                dependent_config_.remote_receiver_x[remote_idx] = router_kernel->GetVirtualCore().x;
                dependent_config_.remote_receiver_y[remote_idx] = router_kernel->GetVirtualCore().y;
                dependent_config_.remote_receiver_queue_id[remote_idx] =
                    idx;  // Queue ids start counting from 0 at input
                dependent_config_.remote_receiver_network_type[remote_idx] = (uint32_t)DispatchRemoteNetworkType::NOC0;
                dependent_config_.remote_receiver_queue_start[remote_idx] =
                    router_kernel->GetStaticConfig().rx_queue_start_addr_words.value() +
                    idx * router_kernel->GetStaticConfig().rx_queue_size_words.value();
                dependent_config_.remote_receiver_queue_size[remote_idx] =
                    router_kernel->GetStaticConfig().rx_queue_size_words.value();
                remote_idx++;
            }
        }
        // Last receiver connection is the return VC, connected to the paired tunneler
        uint32_t return_vc_id = static_config_.vc_count.value() - 1;
        dependent_config_.remote_receiver_x[return_vc_id] = paired_physical_coord.x;
        dependent_config_.remote_receiver_y[return_vc_id] = paired_physical_coord.y;
        dependent_config_.remote_receiver_queue_id[return_vc_id] = return_vc_id;
        dependent_config_.remote_receiver_network_type[return_vc_id] = (uint32_t)DispatchRemoteNetworkType::ETH;
        dependent_config_.remote_receiver_queue_start[return_vc_id] =
            static_config_.in_queue_start_addr_words.value() +
            (return_vc_id) * this->static_config_.in_queue_size_words.value();
        dependent_config_.remote_receiver_queue_size[return_vc_id] = this->static_config_.in_queue_size_words;
        dependent_config_.inner_stop_mux_d_bypass = 0;
        // For certain chips in a tunnel (between first stop and end of tunnel, not including), we do a bypass
        if (static_config_.vc_count.value() > (device_->num_hw_cqs() + 1) &&
            static_config_.vc_count.value() < (4 * device_->num_hw_cqs() + 1)) {
            dependent_config_.inner_stop_mux_d_bypass =
                (return_vc_id << 24) |
                (((tunneler_kernel->GetStaticConfig().vc_count.value() - device_->num_hw_cqs()) * 2 - 1) << 16) |
                (paired_physical_coord.y << 8) | (paired_physical_coord.x);
        }
    }
}

void EthTunnelerKernel::CreateKernel() {
    std::vector<uint32_t> compile_args = {
        static_config_.endpoint_id_start_index.value(),
        static_config_.vc_count.value(),  // # Tunnel lanes = VC count
        static_config_.in_queue_start_addr_words.value(),
        static_config_.in_queue_size_words.value(),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  // Populate remote_receiver_config after
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  // Populate remote_receiver_queue_* after
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  // Populate remote_sender_* after
        static_config_.kernel_status_buf_addr_arg.value(),
        static_config_.kernel_status_buf_size_bytes.value(),
        static_config_.timeout_cycles.value(),
        dependent_config_.inner_stop_mux_d_bypass.value()};
    for (int idx = 0; idx < MAX_TUNNEL_LANES; idx++) {
        if (dependent_config_.remote_receiver_x[idx]) {
            compile_args[4 + idx] |= (dependent_config_.remote_receiver_x[idx].value() & 0xFF);
            compile_args[4 + idx] |= (dependent_config_.remote_receiver_y[idx].value() & 0xFF) << 8;
            compile_args[4 + idx] |= (dependent_config_.remote_receiver_queue_id[idx].value() & 0xFF) << 16;
            compile_args[4 + idx] |= (dependent_config_.remote_receiver_network_type[idx].value() & 0xFF) << 24;
        }
        if (dependent_config_.remote_receiver_queue_start[idx]) {
            compile_args[14 + idx * 2] = dependent_config_.remote_receiver_queue_start[idx].value();
            compile_args[15 + idx * 2] = dependent_config_.remote_receiver_queue_size[idx].value();
        } else {
            compile_args[15 + idx * 2] = 2;  // Dummy size for unused VCs
        }
        if (dependent_config_.remote_sender_x[idx]) {
            compile_args[34 + idx] |= (dependent_config_.remote_sender_x[idx].value() & 0xFF);
            compile_args[34 + idx] |= (dependent_config_.remote_sender_y[idx].value() & 0xFF) << 8;
            compile_args[34 + idx] |= (dependent_config_.remote_sender_queue_id[idx].value() & 0xFF) << 16;
            compile_args[34 + idx] |= (dependent_config_.remote_sender_network_type[idx].value() & 0xFF) << 24;
        }
    }
    TT_ASSERT(compile_args.size() == 48);
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
    configure_kernel_variant(
        dispatch_kernel_file_names[is_remote_ ? US_TUNNELER_REMOTE : US_TUNNELER_LOCAL],
        compile_args,
        defines,
        true,
        false,
        false);
}

uint32_t EthTunnelerKernel::GetRouterQueueIdOffset(FDKernel* k, bool upstream) {
    uint32_t queue_id = (upstream) ? 0 : static_config_.vc_count.value();
    std::vector<FDKernel*>& kernels = (upstream) ? upstream_kernels_ : downstream_kernels_;
    for (auto kernel : kernels) {
        if (auto router_kernel = dynamic_cast<EthRouterKernel*>(kernel)) {
            if (k == kernel) {
                return queue_id;
            }
            queue_id += (upstream) ? router_kernel->GetStaticConfig().fwd_vc_count.value()
                                   : router_kernel->GetStaticConfig().vc_count.value();
        }
    }
    TT_ASSERT(false, "Couldn't find router kernel");
    return queue_id;
}
uint32_t EthTunnelerKernel::GetRouterId(FDKernel* k, bool upstream) {
    std::vector<FDKernel*>& search = (upstream) ? upstream_kernels_ : downstream_kernels_;
    uint32_t router_id = 0;
    for (auto kernel : search) {
        if (auto router_kernel = dynamic_cast<EthRouterKernel*>(kernel)) {
            if (k == kernel) {
                return router_id;
            }
            router_id++;
        }
    }
    TT_ASSERT(false, "Couldn't find router kernel");
    return router_id;
}
