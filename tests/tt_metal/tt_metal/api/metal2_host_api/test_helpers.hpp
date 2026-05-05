// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared test utilities for Metal 2.0 Host API tests.
// These helpers create minimal valid spec objects for testing.

#include <cstdlib>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

// This file contains shortcut helper functions to create minimal valid ProgramSpec
// objects for unit tests. This cuts boilerplate in a unit testing context.
//
// This is NOT intended as a recommended pattern for production code!
// See the Metal 2.0 Host API documentation and programming examples for
// recommended patterns for constructing ProgramSpec objects in production code.

namespace tt::tt_metal::experimental::metal2_host_api::test_helpers {

// ============================================================================
// Test environment helpers
// ============================================================================

// Saves and overrides TT_METAL_SLOW_DISPATCH_MODE on construction;
// restores to its prior state (set or unset) on destruction.
// (Unit test need SLOW_DISPATCH_MODE=1 to make a MeshDevice successfully.)
class ScopedSlowDispatchOverride {
public:
    ScopedSlowDispatchOverride() {
        if (const char* prev = std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
            prev_value_.emplace(prev);
        }
        setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", /*overwrite=*/1);
    }
    ~ScopedSlowDispatchOverride() {
        if (prev_value_) {
            setenv("TT_METAL_SLOW_DISPATCH_MODE", prev_value_->c_str(), /*overwrite=*/1);
        } else {
            unsetenv("TT_METAL_SLOW_DISPATCH_MODE");
        }
    }

    ScopedSlowDispatchOverride(const ScopedSlowDispatchOverride&) = delete;
    ScopedSlowDispatchOverride& operator=(const ScopedSlowDispatchOverride&) = delete;

private:
    std::optional<std::string> prev_value_;
};

// ============================================================================
// Constants
// ============================================================================

// Minimal valid kernel source code for testing
inline constexpr const char* MINIMAL_KERNEL_SOURCE = "void kernel_main() {}";

// ============================================================================
// Spec Creation Helpers
// ============================================================================
//
// Note: KernelSpec and DataflowBufferSpec do not directly encode target_nodes.
// Placement is stated on WorkUnitSpec; pass node sets to MakeMinimalWorkUnit instead.

// Helper to create a minimal valid KernelSpec for data movement (Gen2/Quasar)
inline KernelSpec MakeMinimalDMKernel(const std::string& name, uint8_t num_threads = 1) {
    return KernelSpec{
        .unique_id = name,
        .source = KernelSpec::SourceCode{MINIMAL_KERNEL_SOURCE},
        .num_threads = num_threads,
        .config_spec =
            DataMovementConfiguration{
                .gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{},
            },
    };
}

// Helper to create a minimal valid KernelSpec for data movement (Gen1/WH/BH)
inline KernelSpec MakeMinimalGen1DMKernel(
    const std::string& name,
    tt::tt_metal::DataMovementProcessor processor = tt::tt_metal::DataMovementProcessor::RISCV_0) {
    return KernelSpec{
        .unique_id = name,
        .source = KernelSpec::SourceCode{MINIMAL_KERNEL_SOURCE},
        .num_threads = 1,
        .config_spec =
            DataMovementConfiguration{
                .gen1_data_movement_config =
                    DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = processor,
                    },
            },
    };
}

// Helper to create a minimal valid KernelSpec for compute
inline KernelSpec MakeMinimalComputeKernel(const std::string& name, uint8_t num_threads = 1) {
    return KernelSpec{
        .unique_id = name,
        .source = KernelSpec::SourceCode{MINIMAL_KERNEL_SOURCE},
        .num_threads = num_threads,
        .config_spec = ComputeConfiguration{},
    };
}

// Helper to create a minimal valid DataflowBufferSpec
inline DataflowBufferSpec MakeMinimalDFB(
    const std::string& name, uint32_t entry_size = 1024, uint32_t num_entries = 2) {
    return DataflowBufferSpec{
        .unique_id = name,
        .entry_size = entry_size,
        .num_entries = num_entries,
    };
}

// Helper to create a minimal valid WorkUnitSpec
inline WorkUnitSpec MakeMinimalWorkUnit(
    const std::string& name,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes,
    const std::vector<KernelSpecName>& kernels) {
    return WorkUnitSpec{
        .unique_id = name,
        .kernels = kernels,
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

// Helper to create a minimal valid ProgramSpec for Gen1 (WH/BH): DM producer (RISCV_0) + compute consumer
inline ProgramSpec MakeMinimalGen1ValidProgramSpec() {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel", tt::tt_metal::DataMovementProcessor::RISCV_0);
    auto compute_kernel = MakeMinimalComputeKernel("compute_kernel");

    auto dfb = MakeMinimalDFB("dfb_0");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    BindDFBToKernel(dm_kernel, "dfb_0", "input_dfb", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(compute_kernel, "dfb_0", "input_dfb", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {dm_kernel, compute_kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = {MakeMinimalWorkUnit("work_unit_0", node, {"dm_kernel", "compute_kernel"})};

    return spec;
}

// Helper to create a minimal valid ProgramSpec with one DM and one compute kernel
inline ProgramSpec MakeMinimalValidProgramSpec() {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.program_id = "test_program";

    // Create a DM kernel (producer) and compute kernel (consumer)
    auto dm_kernel = MakeMinimalDMKernel("dm_kernel");
    auto compute_kernel = MakeMinimalComputeKernel("compute_kernel");

    // Create a DFB with data format (required for compute endpoint)
    auto dfb = MakeMinimalDFB("dfb_0");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    // Bind the DFB
    BindDFBToKernel(dm_kernel, "dfb_0", "input_dfb", KernelSpec::DFBEndpointType::PRODUCER);
    BindDFBToKernel(compute_kernel, "dfb_0", "input_dfb", KernelSpec::DFBEndpointType::CONSUMER);

    spec.kernels = {dm_kernel, compute_kernel};
    spec.dataflow_buffers = {dfb};

    // Create a WorkUnitSpec
    spec.work_units = {MakeMinimalWorkUnit("work_unit_0", node, {"dm_kernel", "compute_kernel"})};

    return spec;
}

}  // namespace tt::tt_metal::experimental::metal2_host_api::test_helpers
