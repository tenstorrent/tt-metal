// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ccl_common.hpp"

#include <cstdint>

#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>

namespace tt::tt_fabric {

// Template helper function for generating EDM kernels
static tt::tt_metal::KernelHandle generate_edm_kernel_impl(
    tt::tt_metal::Program& program,
    const tt::tt_metal::IDevice* device,
    const tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder,
    const std::string& kernel_path,
    const CoreCoord& eth_core,
    tt::tt_metal::DataMovementProcessor risc_id,
    tt::tt_metal::NOC noc_id,
    std::optional<tt::tt_metal::KernelBuildOptLevel> opt_level = std::nullopt) {
    edm_builder.dump_to_log();

    std::vector<uint32_t> const edm_kernel_rt_args = edm_builder.get_runtime_args();
    // Ethernet Kernels
    const std::vector<uint32_t> eth_sender_ct_args = edm_builder.get_compile_time_args((uint32_t)risc_id);
    log_trace(tt::LogOp, "EDM core (x={},y={}):", eth_core.x, eth_core.y);
    log_trace(tt::LogOp, "CT ARGS:");
    for (auto const& s : eth_sender_ct_args) {
        log_trace(tt::LogOp, "\t{}", s);
    }

    auto kernel_config =
        tt::tt_metal::EthernetConfig{.noc = noc_id, .processor = risc_id, .compile_args = eth_sender_ct_args};
    if (opt_level.has_value()) {
        kernel_config.opt_level = opt_level.value();
    }
    auto eth_sender_kernel = tt::tt_metal::CreateKernel(
        program,
        kernel_path,
        eth_core,
        kernel_config);

    tt::tt_metal::SetRuntimeArgs(program, eth_sender_kernel, eth_core, edm_kernel_rt_args);

    return eth_sender_kernel;
}

tt::tt_metal::KernelHandle generate_edm_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::IDevice* device,
    const tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder,
    const CoreCoord& eth_core,
    const tt::tt_metal::DataMovementProcessor risc_id,
    tt::tt_metal::NOC noc_id) {
    return generate_edm_kernel_impl(
        program,
        device,
        edm_builder,
        "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_datamover.cpp",
        eth_core,
        risc_id,
        noc_id,
        tt::tt_metal::KernelBuildOptLevel::O3);
}

}  // namespace tt::tt_fabric
