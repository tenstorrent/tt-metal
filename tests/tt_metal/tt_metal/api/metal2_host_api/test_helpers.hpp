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
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/work_split.hpp>
#include "experimental/metal2_host_api/data_movement_hardware_config.hpp"
#include "kernel_types.hpp"

// This file contains shortcut helper functions to create minimal valid ProgramSpec
// objects for unit tests. This cuts boilerplate in a unit testing context.
//
// This is NOT intended as a recommended pattern for production code!
// See the Metal 2.0 Host API documentation and programming examples for
// recommended patterns for constructing ProgramSpec objects in production code.

namespace tt::tt_metal::experimental::test_helpers {

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
inline KernelSpec MakeMinimalGen2DMKernel(std::string name, uint32_t num_threads = 1) {
    return KernelSpec{
        .unique_id = KernelSpecName{std::move(name)},
        .source = KernelSpec::SourceCode{MINIMAL_KERNEL_SOURCE},
        .num_threads = num_threads,
        .hw_config = DataMovementGen2Config{},
    };
}

// Helper to create a minimal valid KernelSpec for data movement (Gen1/WH/BH)
//
// The NOC is derived from the processor (RISCV_0 -> NOC_0, RISCV_1 -> NOC_1) so that the common
// pairing of an RISCV_0 kernel with an RISCV_1 kernel on the same node gets distinct NOCs. On Gen1,
// two dedicated-NOC DM kernels sharing a NOC hang the device (validation rejects it), so a helper
// that always defaulted to NOC_0 would produce a pair that fails validation.
inline KernelSpec MakeMinimalGen1DMKernel(
    std::string name, tt::tt_metal::DataMovementProcessor processor = tt::tt_metal::DataMovementProcessor::RISCV_0) {
    const tt::tt_metal::NOC noc = (processor == tt::tt_metal::DataMovementProcessor::RISCV_0)
                                      ? tt::tt_metal::NOC::NOC_0
                                      : tt::tt_metal::NOC::NOC_1;
    return KernelSpec{
        .unique_id = KernelSpecName{std::move(name)},
        .source = KernelSpec::SourceCode{MINIMAL_KERNEL_SOURCE},
        .num_threads = 1,
        .hw_config = DataMovementGen1Config{.processor = processor, .noc = noc}};
}

// Helper to create a minimal valid KernelSpec for data movement whose Gen1 config is built
// from the READER role via CreateReaderGen1DataMovementConfig (Gen1/WH/BH).
inline KernelSpec MakeMinimalReaderDMKernel(std::string name) {
    return KernelSpec{
        .unique_id = KernelSpecName{std::move(name)},
        .source = KernelSpec::SourceCode{MINIMAL_KERNEL_SOURCE},
        .num_threads = 1,
        .hw_config = CreateReaderGen1DataMovementConfig(),
    };
}

// Helper to create a minimal valid KernelSpec for data movement whose Gen1 config is built
// from the WRITER role via CreateWriterGen1DataMovementConfig (Gen1/WH/BH).
inline KernelSpec MakeMinimalWriterDMKernel(std::string name) {
    return KernelSpec{
        .unique_id = KernelSpecName{std::move(name)},
        .source = KernelSpec::SourceCode{MINIMAL_KERNEL_SOURCE},
        .num_threads = 1,
        .hw_config = CreateWriterGen1DataMovementConfig(),
    };
}

// Helper to create a minimal valid KernelSpec for compute (Gen2/Quasar)
inline KernelSpec MakeMinimalGen2ComputeKernel(std::string name, uint32_t num_threads = 1) {
    return KernelSpec{
        .unique_id = KernelSpecName{std::move(name)},
        .source = KernelSpec::SourceCode{MINIMAL_KERNEL_SOURCE},
        .num_threads = num_threads,
        .hw_config = ComputeGen2Config{},
    };
}

// Helper to create a minimal valid KernelSpec for compute (Gen1/WH/BH)
inline KernelSpec MakeMinimalGen1ComputeKernel(std::string name, uint32_t num_threads = 1) {
    return KernelSpec{
        .unique_id = KernelSpecName{std::move(name)},
        .source = KernelSpec::SourceCode{MINIMAL_KERNEL_SOURCE},
        .num_threads = num_threads,
        .hw_config = ComputeGen1Config{},
    };
}

// Helper to create a minimal valid DataflowBufferSpec
inline DataflowBufferSpec MakeMinimalDFB(std::string name, uint32_t entry_size = 1024, uint32_t num_entries = 2) {
    return DataflowBufferSpec{
        .unique_id = DFBSpecName{std::move(name)},
        .entry_size = entry_size,
        .num_entries = num_entries,
    };
}

// Helper to create a minimal valid WorkUnitSpec
inline WorkUnitSpec MakeMinimalWorkUnit(std::string name, const Nodes& nodes, const std::vector<std::string>& kernels) {
    std::vector<KernelSpecName> kernel_names;
    kernel_names.reserve(kernels.size());
    for (const auto& kernel : kernels) {
        kernel_names.emplace_back(kernel);
    }
    return WorkUnitSpec{
        .name = std::move(name),
        .kernels = std::move(kernel_names),
        .target_nodes = nodes,
    };
}

// Helper to create a minimal valid TensorParameter.
// Default layout: BFLOAT16, ROW_MAJOR, interleaved, shape {1, 32}. Hardware-agnostic;
// works on any mock device (alignment + virtualized cores resolved by MakeProgramFromSpec).
// buffer_type defaults to DRAM; pass BufferType::L1 for an SRAM-resident parameter.
inline TensorParameter MakeMinimalTensorParameter(
    std::string name, tt::tt_metal::BufferType buffer_type = tt::tt_metal::BufferType::DRAM) {
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config = tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, buffer_type};
    auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
    auto spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 32}, tensor_layout);
    return TensorParameter{
        .unique_id = TensorParamName{std::move(name)},
        .spec = std::move(spec),
    };
}

// Helper to add a TensorBinding to a kernel.
inline void BindTensorParameterToKernel(
    KernelSpec& kernel, std::string tensor_parameter_name, std::string accessor_name) {
    kernel.tensor_bindings.push_back(TensorBinding{
        .tensor_parameter_name = TensorParamName{std::move(tensor_parameter_name)},
        .accessor_name = std::move(accessor_name),
    });
}

// Helper to create a height-sharded TensorParameter for tests that exercise the
// sharded TAA CTA payload (rank/num_banks/tensor_shape/shard_shape/bank_coords).
//
// Defaults give a simple legal layout: BFLOAT16 tile-layout tensor of `logical_shape`,
// sharded across the first `num_cores` cores of the worker grid with shard shape
// `shard_shape`. Caller is responsible for choosing a shape that fits the grid.
inline TensorParameter MakeShardedTensorParameter(
    std::string name,
    const tt::tt_metal::Shape& logical_shape,
    const std::array<uint32_t, 2>& shard_shape,
    uint32_t num_cores) {
    auto shard_grid = tt::tt_metal::num_cores_to_corerangeset(num_cores, CoreCoord{num_cores, 1}, /*row_wise=*/true);
    tt::tt_metal::ShardSpec shard_spec{
        shard_grid, {shard_shape[0], shard_shape[1]}, tt::tt_metal::ShardOrientation::ROW_MAJOR};
    tt::tt_metal::MemoryConfig memory_config{
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED, tt::tt_metal::BufferType::L1, shard_spec};
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE);
    auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
    return TensorParameter{
        .unique_id = TensorParamName{std::move(name)},
        .spec = tt::tt_metal::TensorSpec(logical_shape, std::move(tensor_layout)),
    };
}

// Helper to create a minimal valid ProgramSpec for Gen1 (WH/BH): DM producer (RISCV_0) + compute consumer
inline ProgramSpec MakeMinimalGen1ValidProgramSpec() {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel", tt::tt_metal::DataMovementProcessor::RISCV_0);
    auto compute_kernel = MakeMinimalGen1ComputeKernel("compute_kernel");

    auto dfb = MakeMinimalDFB("dfb_0");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    dm_kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "input_dfb"));
    compute_kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_0"}, "input_dfb"));

    spec.kernels = {dm_kernel, compute_kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = {MakeMinimalWorkUnit("work_unit_0", node, {"dm_kernel", "compute_kernel"})};

    return spec;
}

// Helper to create a minimal valid ProgramSpec with one DM and one compute kernel
inline ProgramSpec MakeMinimalValidProgramSpec() {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Create a DM kernel (producer) and compute kernel (consumer)
    auto dm_kernel = MakeMinimalGen2DMKernel("dm_kernel");
    auto compute_kernel = MakeMinimalGen2ComputeKernel("compute_kernel");

    // Create a DFB with data format (required for compute endpoint)
    auto dfb = MakeMinimalDFB("dfb_0");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    // Bind the DFB
    dm_kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "input_dfb"));
    compute_kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_0"}, "input_dfb"));

    spec.kernels = {dm_kernel, compute_kernel};
    spec.dataflow_buffers = {dfb};

    // Create a WorkUnitSpec
    spec.work_units = {MakeMinimalWorkUnit("work_unit_0", node, {"dm_kernel", "compute_kernel"})};

    return spec;
}

}  // namespace tt::tt_metal::experimental::test_helpers
