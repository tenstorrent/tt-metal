// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_mux.hpp"
#include "context/metal_context.hpp"
#include "dispatch/kernel_config/fd_kernel.hpp"
#include "fabric/fabric_mux_config.hpp"
#include "hal_types.hpp"
#include "tt_align.hpp"
#include "tt_metal.hpp"
#include <algorithm>
#include <tt-metalium/fabric.hpp>

namespace tt::tt_metal {

void FabricMux::GenerateStaticConfigs() {
    constexpr uint32_t k_NumSlotsPerChannel = 8;

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto l1_base = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);

    static_config_.buffer_base_address = tt::align(l1_base, hal.get_alignment(tt::tt_metal::HalMemType::L1));

    const uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_->id());
    logical_core_ = MetalContext::instance().get_dispatch_core_manager().mux_core(device_->id(), channel, this->cq_id_);

    // Count number of value kernels that need the channels
    const auto kernels_requiring_full_size_channel =
        std::count_if(upstream_kernels_.begin(), upstream_kernels_.end(), [](const FDKernel* kernel) {
            return kernel->GetNodeId() != -1;
        });
    const auto kernels_requiring_header_only_channel =
        std::count_if(downstream_kernels_.begin(), downstream_kernels_.end(), [](const FDKernel* kernel) {
            return kernel->GetNodeId() != -1;
        });

    log_info(
        tt::LogDevice,
        "Create FabricMux on Device {} with {} Header Only Channels and {} Full Size Channels",
        device_->id(),
        kernels_requiring_header_only_channel,
        kernels_requiring_full_size_channel);

    // Setup the buffer sizes depending on how many upstream/downstream kernels are connected
    static_config_.num_full_size_channels = kernels_requiring_full_size_channel;
    static_config_.num_header_only_channels = kernels_requiring_header_only_channel;
    static_config_.buffer_size_bytes = sizeof(tt::tt_fabric::PacketHeader) + 4096;

    mux_kernel_config_ = std::make_shared<tt::tt_fabric::FabricMuxConfig>(
        static_config_.num_full_size_channels.value(),
        static_config_.num_header_only_channels.value(),
        k_NumSlotsPerChannel,
        k_NumSlotsPerChannel,
        static_config_.buffer_size_bytes.value(),
        static_config_.buffer_base_address.value());
    mux_ct_args = mux_kernel_config_->get_fabric_mux_compile_time_args();

    mux_rt_args.clear();
    tt_fabric::append_fabric_connection_rt_args(
        device_id_, servicing_device_id_, 0, *program_, {logical_core_}, mux_rt_args);
}

void FabricMux::GenerateDependentConfigs() {}

void FabricMux::CreateKernel() {
    std::map<string, string> defines = {
        {"DISPATCH_KERNEL", "1"},
    };

    auto mux_kernel = tt::tt_metal::CreateKernel(
        *program_,
        dispatch_kernel_file_names[FABRIC_MUX],
        logical_core_,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = mux_ct_args,
            .defines = defines});

    tt::tt_metal::SetRuntimeArgs(*program_, mux_kernel, logical_core_, mux_rt_args);
}

void FabricMux::ConfigureCore() {
    // TODO: Only need to clear the read/write pointers to 0
    std::vector<uint32_t> mux_zero_vec((mux_kernel_config_->get_num_bytes_to_clear() / sizeof(uint32_t)), 0);
    tt::tt_metal::detail::WriteToDeviceL1(
        device_, logical_core_, mux_kernel_config_->get_start_address_to_clear(), mux_zero_vec);
}

int FabricMux::GetWorkerChannelIndex(int worker_id, tt::tt_fabric::FabricMuxChannelType channel_type) const {
    const auto id_checker = [=](const FDKernel* kernel) { return kernel->GetNodeId() == worker_id; };
    int position = -1;
    if (channel_type == tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL) {
        position = std::distance(
            upstream_kernels_.begin(), std::find_if(upstream_kernels_.begin(), upstream_kernels_.end(), id_checker));
    } else if (channel_type == tt::tt_fabric::FabricMuxChannelType::HEADER_ONLY_CHANNEL) {
        position = std::distance(
            downstream_kernels_.begin(),
            std::find_if(downstream_kernels_.begin(), downstream_kernels_.end(), id_checker));
    }

    TT_FATAL(position != -1, "Worker ID {} not found in upstream/downstream kernels", worker_id);
    return position;
}

void assemble_fabric_mux_client_config_args(
    int node_id,
    tt::tt_fabric::FabricMuxChannelType ch_type,
    const FabricMux* fabric_mux,
    fabric_mux_client_config& config) {
    const auto ch_index = fabric_mux->GetWorkerChannelIndex(node_id, ch_type);
    const CoreCoord& fabric_mux_core = fabric_mux->GetVirtualCore();
    config.virtual_x = fabric_mux_core.x;
    config.virtual_y = fabric_mux_core.y;
    config.num_buffers_per_channel = ch_type == tt::tt_fabric::FabricMuxChannelType::HEADER_ONLY_CHANNEL
                                         ? fabric_mux->GetMuxKernelConfig()->num_buffers_header_only_channel
                                         : fabric_mux->GetMuxKernelConfig()->num_buffers_full_size_channel;
    config.channel_buffer_size_bytes = fabric_mux->GetMuxKernelConfig()->get_buffer_size_bytes(ch_type);
    config.channel_base_address = fabric_mux->GetMuxKernelConfig()->get_channel_base_address(ch_type, ch_index);
    config.connection_info_address = fabric_mux->GetMuxKernelConfig()->get_connection_info_address(ch_type, ch_index);
    config.connection_handshake_address =
        fabric_mux->GetMuxKernelConfig()->get_connection_handshake_address(ch_type, ch_index);
    config.flow_control_address = fabric_mux->GetMuxKernelConfig()->get_flow_control_address(ch_type, ch_index);
    config.buffer_index_address = fabric_mux->GetMuxKernelConfig()->get_buffer_index_address(ch_type, ch_index);
    config.status_address = fabric_mux->GetMuxKernelConfig()->get_status_address();
    config.termination_signal_address = fabric_mux->GetMuxKernelConfig()->get_termination_signal_address();
}

}  // namespace tt::tt_metal
