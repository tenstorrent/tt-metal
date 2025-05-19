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
    constexpr uint32_t k_NumSlotsPerChannel = 16;

    uint32_t l1_base = 0;
    uint32_t l1_size = 0;

    if (GetCoreType() == CoreType::WORKER) {
        l1_base = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);
        l1_size = tt::tt_metal::MetalContext::instance().hal().get_dev_size(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE);
    } else {
        l1_base = MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
        l1_size = MetalContext::instance().hal().get_dev_size(
            tt::tt_metal::HalProgrammableCoreType::IDLE_ETH, tt::tt_metal::HalL1MemAddrType::BASE);
    }
    static_config_.buffer_base_address =
        tt::align(l1_base, tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1));

    const uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_->id());
    logical_core_ = MetalContext::instance().get_dispatch_core_manager().fabric_mux_core(device_->id(), channel, this->cq_id_);

    // Count number of value kernels that need the channels
    const auto kernels_requiring_full_size_channel =
        std::count_if(upstream_kernels_.begin(), upstream_kernels_.end(), [](const FDKernel* kernel) {
            return kernel->GetNodeId() != -1;
        });
    const auto kernels_requiring_header_only_channel =
        std::count_if(downstream_kernels_.begin(), downstream_kernels_.end(), [](const FDKernel* kernel) {
            return kernel->GetNodeId() != -1;
        });

    // Setup the buffer sizes depending on how many upstream/downstream kernels are connected
    static_config_.num_full_size_channels = kernels_requiring_full_size_channel;
    static_config_.num_header_only_channels = kernels_requiring_header_only_channel;
    static_config_.buffer_size_bytes = sizeof(tt::tt_fabric::PacketHeader) + 4096;

    uint32_t programmable_core_type_index =
        get_programmable_core_type_index(GetCoreType(), /*is_active_eth_core=*/false);
    mux_kernel_config_ = std::make_shared<tt::tt_fabric::FabricMuxConfig>(
        static_config_.num_full_size_channels.value(),
        static_config_.num_header_only_channels.value(),
        k_NumSlotsPerChannel,
        k_NumSlotsPerChannel,
        static_config_.buffer_size_bytes.value(),
        static_config_.buffer_base_address.value());
    mux_ct_args = mux_kernel_config_->get_fabric_mux_compile_time_args();

    log_info(
        tt::LogDevice,
        "Create FabricMux on Device {} with {} Header Only Channels and {} Full Size Channels. {} x {} B Buffers Each "
        "Full Channel. {} status {:#x} termination {:#x}",
        device_->id(),
        kernels_requiring_header_only_channel,
        kernels_requiring_full_size_channel,
        k_NumSlotsPerChannel,
        static_config_.buffer_size_bytes.value(),
        logical_core_.str(),
        mux_kernel_config_->get_status_address(),
        mux_kernel_config_->get_termination_signal_address());

    uint32_t mux_buffer_end = mux_kernel_config_->get_start_address_to_clear() + mux_kernel_config_->get_num_bytes_to_clear();
    TT_ASSERT(
        mux_buffer_end < l1_size,
        "Fabric MUX Buffer End {} Exceeds Max L1 {}", mux_buffer_end, l1_size);

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
    const auto& kernels = channel_type == tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL ? upstream_kernels_
                                                                                                 : downstream_kernels_;
    for (int i = 0; i < kernels.size(); ++i) {
        if (kernels[i]->GetNodeId() == worker_id) {
            return i;
        }
    }

    TT_THROW("Worker ID {} not found in upstream/downstream kernels", worker_id);
    return -1;
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
