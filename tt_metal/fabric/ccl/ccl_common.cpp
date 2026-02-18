// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ccl_common.hpp"

#include <cstdint>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt_metal/fabric/erisc_datamover_builder.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include "tt_metal/impl/kernels/kernel.hpp"

namespace tt::tt_fabric {

tt::tt_metal::KernelHandle generate_erisc_datamover_kernel(const FabricEriscDatamoverKernelConfig& edm_kernel_config) {
    log_trace(tt::LogFabric, "EDM core (x={},y={}):", edm_kernel_config.eth_core.x, edm_kernel_config.eth_core.y);
    log_trace(tt::LogFabric, "CT ARGS:");
    for ([[maybe_unused]] const auto& s : edm_kernel_config.compile_time_args) {
        log_trace(tt::LogFabric, "\t{}", s);
    }

    auto kernel_config = tt::tt_metal::EthernetConfig{
        .noc = edm_kernel_config.noc_id,
        .processor = edm_kernel_config.risc_id,
        .compile_args = edm_kernel_config.compile_time_args,
        .named_compile_args = edm_kernel_config.named_compile_time_args,
    };
    if (edm_kernel_config.opt_level.has_value()) {
        kernel_config.opt_level = edm_kernel_config.opt_level.value();
    }
    auto eth_sender_kernel = tt::tt_metal::CreateKernel(
        edm_kernel_config.program, edm_kernel_config.kernel_path, edm_kernel_config.eth_core, kernel_config);

    tt::tt_metal::SetRuntimeArgs(
        edm_kernel_config.program, eth_sender_kernel, edm_kernel_config.eth_core, edm_kernel_config.runtime_args);

    std::stringstream ss;
    ss << "EDM ARGS:\n";
    for (const auto& s : edm_kernel_config.runtime_args) {
        ss << "\t" << s << "\n";
    }
    log_trace(tt::LogFabric, "{}", ss.str());

    return eth_sender_kernel;
}

tt::tt_metal::KernelHandle generate_edm_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder,
    const CoreCoord& eth_core,
    const tt::tt_metal::DataMovementProcessor risc_id,
    tt::tt_metal::NOC noc_id) {
    edm_builder.dump_to_log();
    const auto [eth_sender_ct_args, named_ct_args] = edm_builder.get_compile_time_args((uint32_t)risc_id);
    return generate_erisc_datamover_kernel(tt::tt_fabric::FabricEriscDatamoverKernelConfig{
        .program = program,
        .kernel_path = "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp",
        .eth_core = eth_core,
        .risc_id = risc_id,
        .noc_id = noc_id,
        .compile_time_args = eth_sender_ct_args,
        .named_compile_time_args = named_ct_args,
        .runtime_args = edm_builder.get_runtime_args(),
        .opt_level = tt::tt_metal::KernelBuildOptLevel::O3,
    });
}

}  // namespace tt::tt_fabric
