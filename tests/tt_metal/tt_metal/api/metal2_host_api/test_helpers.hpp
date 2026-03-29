// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared test utilities for Metal 2.0 Host API tests.
// These helpers create minimal valid spec objects for testing.

#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

namespace tt::tt_metal::experimental::metal2_host_api::test_helpers {

// ============================================================================
// Constants
// ============================================================================

// Minimal valid kernel source code for testing
inline constexpr const char* MINIMAL_DM_KERNEL_SOURCE = "void kernel_main() {}";
inline constexpr const char* MINIMAL_COMPUTE_KERNEL_SOURCE = "void kernel_main() {}";

// ============================================================================
// Spec Creation Helpers
// ============================================================================

// Helper to create a minimal valid KernelSpec for data movement
inline KernelSpec MakeMinimalDMKernel(
    const std::string& name, const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes, uint8_t num_threads = 1) {
    return KernelSpec{
        .unique_id = name,
        .source = MINIMAL_DM_KERNEL_SOURCE,
        .source_type = KernelSpec::SourceType::SOURCE_CODE,
        .target_nodes = nodes,
        .num_threads = num_threads,
        .config_spec =
            DataMovementConfiguration{
                .gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{},
            },
    };
}

// Helper to create a minimal valid KernelSpec for compute
inline KernelSpec MakeMinimalComputeKernel(
    const std::string& name, const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes, uint8_t num_threads = 1) {
    return KernelSpec{
        .unique_id = name,
        .source = MINIMAL_COMPUTE_KERNEL_SOURCE,
        .source_type = KernelSpec::SourceType::SOURCE_CODE,
        .target_nodes = nodes,
        .num_threads = num_threads,
        .config_spec = ComputeConfiguration{},
    };
}

// Helper to create a minimal valid DataflowBufferSpec
inline DataflowBufferSpec MakeMinimalDFB(
    const std::string& name,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes,
    uint32_t entry_size = 1024,
    uint32_t num_entries = 2) {
    return DataflowBufferSpec{
        .unique_id = name,
        .target_nodes = nodes,
        .entry_size = entry_size,
        .num_entries = num_entries,
    };
}

// Helper to create a minimal valid WorkerSpec
inline WorkerSpec MakeMinimalWorker(
    const std::string& name,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes,
    const std::vector<KernelSpecName>& kernels,
    const std::vector<DFBSpecName>& dfbs = {}) {
    return WorkerSpec{
        .unique_id = name,
        .kernels = kernels,
        .dataflow_buffers = dfbs,
        .target_nodes = nodes,
    };
}

// Helper to bind a DFB to a kernel as producer or consumer
inline void BindDFBToKernel(
    KernelSpec& kernel,
    const std::string& dfb_name,
    const std::string& accessor_name,
    KernelSpec::DFBEndpointType endpoint_type,
    DFBAccessPattern access_pattern = DFBAccessPattern::STRIDED) {
    kernel.dfb_bindings.push_back(KernelSpec::DFBBinding{
        .dfb_spec_name = dfb_name,
        .local_accessor_name = accessor_name,
        .endpoint_type = endpoint_type,
        .access_pattern = access_pattern,
    });
}

// Helper to create a minimal valid ProgramSpec with one DM and one compute kernel
inline ProgramSpec MakeMinimalValidProgramSpec() {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Create a DM kernel (producer) and compute kernel (consumer)
    auto dm_kernel = MakeMinimalDMKernel("dm_kernel", node);
    auto compute_kernel = MakeMinimalComputeKernel("compute_kernel", node);

    // Create a DFB with data format (required for compute endpoint)
    auto dfb = MakeMinimalDFB("dfb_0", node);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    // Bind the DFB
    BindDFBToKernel(dm_kernel, "dfb_0", "input_dfb", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(compute_kernel, "dfb_0", "input_dfb", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {dm_kernel, compute_kernel};
    spec.dataflow_buffers = {dfb};

    // Create a WorkerSpec
    spec.workers =
        std::vector<WorkerSpec>{MakeMinimalWorker("worker_0", node, {"dm_kernel", "compute_kernel"}, {"dfb_0"})};

    return spec;
}

}  // namespace tt::tt_metal::experimental::metal2_host_api::test_helpers
