// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared ProgramSpec / ProgramRunArgs builders for the Metal 2.0 ProgramRunArgs unit tests.

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"

#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental::test_helpers {

// Create a ProgramSpec with specified RTA schema for the DM kernel
// (The compute kernel has no RTAs)
inline ProgramSpec MakeSpecWithRTAs(const NodeCoord& /*node*/, size_t num_per_node_rtas, size_t num_common_rtas) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Set the RTA schema on the dm_kernel (first kernel)
    spec.kernels[0].advanced_options =
        KernelAdvancedOptions{.num_runtime_varargs = num_per_node_rtas, .num_common_runtime_varargs = num_common_rtas};

    // compute_kernel has no RTAs (defaults: 0 / 0)

    return spec;
}

inline ProgramSpec MakeSpecWithAliasedDfbs(uint32_t es_a, uint32_t ne_a, uint32_t es_b, uint32_t ne_b) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    DataflowBufferSpec dfb_a = MakeMinimalDFB("dfb_a", es_a, ne_a);
    dfb_a.data_format_metadata = tt::DataFormat::Float16_b;
    dfb_a.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_b"}}};
    DataflowBufferSpec dfb_b = MakeMinimalDFB("dfb_b", es_b, ne_b);
    dfb_b.data_format_metadata = tt::DataFormat::Float16_b;
    dfb_b.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_a"}}};
    spec.dataflow_buffers = {dfb_a, dfb_b};

    // Replace the single dfb_0 bindings: each kernel binds both aliased DFBs.
    spec.kernels[0].dfb_bindings = {
        ProducerOf(DFBSpecName{"dfb_a"}, "input_dfb_a"),
        ProducerOf(DFBSpecName{"dfb_b"}, "input_dfb_b"),
    };
    spec.kernels[1].dfb_bindings = {
        ConsumerOf(DFBSpecName{"dfb_a"}, "input_dfb_a"),
        ConsumerOf(DFBSpecName{"dfb_b"}, "input_dfb_b"),
    };
    return spec;
}

// Helper to create ProgramRunArgs for a single kernel
inline ProgramRunArgs::KernelRunArgs MakeKernelRunArgs(
    KernelSpecName kernel,
    const NodeCoord& node,
    const std::vector<uint32_t>& per_node_args,
    const std::vector<uint32_t>& common_args) {
    return ProgramRunArgs::KernelRunArgs{
        .kernel = std::move(kernel),
        .advanced_options =
            AdvancedKernelRunArgs{
                .runtime_varargs = {{node, per_node_args}},
                .common_runtime_varargs = common_args,
            },
    };
}

// Helper to create complete ProgramRunArgs for the minimal spec (both kernels)
inline ProgramRunArgs MakeRunArgsForMinimalSpec(
    const NodeCoord& node,
    const std::vector<uint32_t>& dm_per_node_args,
    const std::vector<uint32_t>& dm_common_args,
    const std::vector<uint32_t>& compute_per_node_args = {},
    const std::vector<uint32_t>& compute_common_args = {}) {
    ProgramRunArgs params;
    params.kernel_run_args.push_back(
        MakeKernelRunArgs(KernelSpecName{"dm_kernel"}, node, dm_per_node_args, dm_common_args));
    params.kernel_run_args.push_back(
        MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, compute_per_node_args, compute_common_args));
    return params;
}

// Make a ProgramSpec where the DM kernel has a named-RTA / named-CRTA schema.
inline ProgramSpec MakeSpecWithNamedArgs(
    const NodeCoord& node, const std::vector<std::string>& named_rtas, const std::vector<std::string>& named_crtas) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = named_rtas;
    spec.kernels[0].runtime_arg_schema.common_runtime_arg_names = named_crtas;
    (void)node;  // node inherited from MakeMinimalValidProgramSpec (0,0)
    return spec;
}

// Helper: build a ProgramSpec with a borrowed-memory DFB backed by a TensorParameter.
// DFB default size: 32 bytes (entry_size 16 * num_entries 2); fits inside
// MakeMinimalTensorParameter's 1x32 BFLOAT16 default (64 bytes).
inline ProgramSpec MakeBorrowedDFBProgramSpecForRunArgs(
    const std::string& tensor_param_name = "borrowed_tensor",
    uint32_t dfb_entry_size = 16,
    uint32_t dfb_num_entries = 2) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "borrowed_dfb_test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb", dfb_entry_size, dfb_num_entries);
    dfb.borrowed_from = TensorParamName{tensor_param_name};

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    auto tensor_param = MakeMinimalTensorParameter(tensor_param_name, tt::tt_metal::BufferType::L1);
    BindTensorParameterToKernel(producer, tensor_param_name, "borrowed_t");

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.tensor_parameters = {tensor_param};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};
    return spec;
}

// Helper: read the patched tensor binding address out of a kernel's CRTA buffer.
// Mirrors the offset arithmetic UpdateTensorArgs uses internally, so a regression in either
// site shows up as a test failure.
inline uint32_t ReadBindingAddressFromCRTA(
    const Program& program, const std::string& kernel_name, const std::string& tensor_parameter_name) {
    auto kernel = program.impl().get_kernel_by_spec_name(kernel_name);
    for (const auto& handle : kernel->tensor_binding_handles()) {
        if (handle.tensor_parameter_name == tensor_parameter_name) {
            const uint32_t word_index = handle.addr_crta_offset / sizeof(uint32_t);
            return kernel->common_runtime_args_data().data()[word_index];
        }
    }
    ADD_FAILURE() << "No binding handle for '" << tensor_parameter_name << "' on kernel '" << kernel_name << "'";
    return 0;
}

}  // namespace tt::tt_metal::experimental::test_helpers
