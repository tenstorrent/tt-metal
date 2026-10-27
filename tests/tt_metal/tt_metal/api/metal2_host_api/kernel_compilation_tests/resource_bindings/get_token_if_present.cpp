// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile smoke tests for get_token_if_present: construct a resource from a present token, and
// keep the nullptr path compiling (mock Wormhole).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;

TEST_F(ProgramSpecTestGen1, CPU_GetTokenIfPresentReturnsNullptrWhenNoBindingsComputeJITSmoke) {
    // The DM counterpart is CPU_GetTokenIfPresentConstructsWhenNoBindingsJITSmoke.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "binding_lookup_empty_compute";

    auto compute = MakeMinimalGen1ComputeKernel("compute");
    compute.source = KernelSpec::SourceCode{R"(
#include "api/tensor/local_tensor_accessor.h"
// Codegen omits these when the kernel has no DFB / scratchpad bindings.
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"

void kernel_main() {
    static_assert(dfb::get_token_if_present<"missing">() == nullptr);
    if (const auto* token = dfb::get_token_if_present<"missing">()) {
        DataflowBuffer buf(*token);
        (void)buf;
    }

    static_assert(tensor::get_token_if_present<"missing">() == nullptr);
    if (const auto* token = tensor::get_token_if_present<"missing">()) {
        LocalTensorAccessor<uint32_t> local_accessor(*token);
        (void)local_accessor;
    }

    static_assert(scratch::get_token_if_present<"missing">() == nullptr);
    if (const auto* token = scratch::get_token_if_present<"missing">()) {
        Scratchpad<int32_t> pad(*token);
        (void)pad;
    }
}
)"};

    spec.kernels = {compute};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"compute"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_GetTokenIfPresentConstructsWhenNoBindingsJITSmoke) {
    // The compute counterpart is CPU_GetTokenIfPresentReturnsNullptrWhenNoBindingsComputeJITSmoke.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "binding_lookup_empty_constructs";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
#include "api/tensor/local_tensor_accessor.h"
// Codegen omits these when the kernel has no DFB / scratchpad bindings.
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"

void kernel_main() {
    static_assert(dfb::get_token_if_present<"missing">() == nullptr);
    if (const auto* token = dfb::get_token_if_present<"missing">()) {
        DataflowBuffer buf(*token);
        (void)buf;
    }

    static_assert(tensor::get_token_if_present<"missing">() == nullptr);
    if (const auto* token = tensor::get_token_if_present<"missing">()) {
        TensorAccessor accessor(*token);
        LocalTensorAccessor<uint32_t> local_accessor(*token);
        (void)accessor;
        (void)local_accessor;
    }

    static_assert(scratch::get_token_if_present<"missing">() == nullptr);
    if (const auto* token = scratch::get_token_if_present<"missing">()) {
        Scratchpad<int32_t> pad(*token);
        (void)pad;
    }
}
)"};

    spec.kernels = {dm_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_GetTokenIfPresentConstructsDataflowBufferJITSmoke) {
    // dfb_present is bound; dfb_absent is not. Both lookups use the same `if (const auto* token = ...)`
    // shape so the absent path still type-checks DataflowBuffer(*token) in the false branch.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.kernels[0].dfb_bindings[0].accessor_name = "dfb_present";
    spec.kernels[1].dfb_bindings[0].accessor_name = "dfb_present";

    spec.kernels[0].source = KernelSpec::SourceCode{R"(
void kernel_main() {
    static_assert(dfb::get_token_if_present<"dfb_present">() == &dfb::dfb_present);
    if (const auto* token = dfb::get_token_if_present<"dfb_present">()) {
        DataflowBuffer buf(*token);
        (void)buf;
    }

    static_assert(dfb::get_token_if_present<"dfb_absent">() == nullptr);
    if (const auto* token = dfb::get_token_if_present<"dfb_absent">()) {
        DataflowBuffer buf(*token);
        (void)buf;
    }
}
)"};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_GetTokenIfPresentConstructsDataflowBufferComputeJITSmoke) {
    // Compute-kernel counterpart of CPU_GetTokenIfPresentConstructsDataflowBufferJITSmoke.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.kernels[0].dfb_bindings[0].accessor_name = "dfb_present";
    spec.kernels[1].dfb_bindings[0].accessor_name = "dfb_present";

    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 1024}};
    spec.kernels[1].scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"});

    spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor", tt::tt_metal::BufferType::L1)};
    BindTensorParameterToKernel(spec.kernels[1], "input_tensor", "tensor_present");

    spec.kernels[1].source = KernelSpec::SourceCode{R"(
#include "api/tensor/local_tensor_accessor.h"

void kernel_main() {
    static_assert(dfb::get_token_if_present<"dfb_present">() == &dfb::dfb_present);
    if (const auto* token = dfb::get_token_if_present<"dfb_present">()) {
        DataflowBuffer buf(*token);
        (void)buf;
    }

    static_assert(dfb::get_token_if_present<"dfb_absent">() == nullptr);
    if (const auto* token = dfb::get_token_if_present<"dfb_absent">()) {
        DataflowBuffer buf(*token);
        (void)buf;
    }

    static_assert(scratch::get_token_if_present<"scratch">() == &scratch::scratch);
    if (const auto* token = scratch::get_token_if_present<"scratch">()) {
        Scratchpad<int32_t> pad(*token);
        (void)pad;
    }

    static_assert(tensor::get_token_if_present<"tensor_present">() == &tensor::tensor_present);
    if (const auto* token = tensor::get_token_if_present<"tensor_present">()) {
        LocalTensorAccessor<uint32_t> local_accessor(*token);
        (void)local_accessor;
    }
}
)"};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_GetTokenIfPresentPrUsageExampleJITSmoke) {
    // Kernel shape from the PR description: always-present `normal`, optional `bias` via
    // get_token_if_present + std::optional. Compiles the same source on DM and compute, with
    // and without `bias` bound.
    constexpr const char* kSource = R"(
#include <optional>
void kernel_main() {
    DataflowBuffer dfb_normal(dfb::normal);

    const auto* bias_token = dfb::get_token_if_present<"bias">();

    std::optional<DataflowBufferAnyPattern> dfb_bias;
    if (bias_token != nullptr) {
        dfb_bias.emplace(*bias_token);
    }

    dfb_normal.push_back(1);
    if (dfb_bias) {
        dfb_bias->push_back(1);
    }
}
)";

    auto compile_variant = [&](bool with_bias) {
        NodeCoord node{0, 0};

        auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
        dm_kernel.source = KernelSpec::SourceCode{kSource};
        auto compute_kernel = MakeMinimalGen1ComputeKernel("compute_kernel");
        compute_kernel.source = KernelSpec::SourceCode{kSource};

        auto normal = MakeMinimalDFB("normal");
        normal.data_format_metadata = tt::DataFormat::Float16_b;
        dm_kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"normal"}, "normal"));
        compute_kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"normal"}, "normal"));

        ProgramSpec spec;
        spec.name = with_bias ? "pr_usage_bias_present" : "pr_usage_bias_absent";
        spec.dataflow_buffers = {normal};
        if (with_bias) {
            auto bias = MakeMinimalDFB("bias");
            bias.data_format_metadata = tt::DataFormat::Float16_b;
            spec.dataflow_buffers.push_back(bias);
            dm_kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"bias"}, "bias"));
            compute_kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"bias"}, "bias"));
        }
        spec.kernels = {dm_kernel, compute_kernel};
        spec.work_units = {MakeMinimalWorkUnit("work_unit", node, {"dm_kernel", "compute_kernel"})};

        Program program = MakeProgramFromSpec(*mesh_device_, spec);
        EXPECT_NO_THROW(program.impl().compile(mesh_device_.get())) << "with_bias=" << with_bias;
    };

    compile_variant(/*with_bias=*/false);
    compile_variant(/*with_bias=*/true);
}

TEST_F(ProgramSpecTestGen1, CPU_GetTokenIfPresentConstructsScratchpadJITSmoke) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "binding_lookup_constructs_scratch";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    static_assert(scratch::get_token_if_present<"scratch_present">() == &scratch::scratch_present);
    if (const auto* token = scratch::get_token_if_present<"scratch_present">()) {
        Scratchpad<int32_t> pad(*token);
        (void)pad;
    }

    static_assert(scratch::get_token_if_present<"scratch_absent">() == nullptr);
    if (const auto* token = scratch::get_token_if_present<"scratch_absent">()) {
        Scratchpad<int32_t> pad(*token);
        (void)pad;
    }
}
)"};
    dm_kernel.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch_present"});

    spec.kernels = {dm_kernel};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 1024}};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_GetTokenIfPresentConstructsTensorAccessorJITSmoke) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "binding_lookup_constructs_tensor";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
#include "api/tensor/local_tensor_accessor.h"

void kernel_main() {
    static_assert(tensor::get_token_if_present<"tensor_present">() == &tensor::tensor_present);
    if (const auto* token = tensor::get_token_if_present<"tensor_present">()) {
        TensorAccessor accessor(*token);
        LocalTensorAccessor<uint32_t> local_accessor(*token);
        (void)accessor;
        (void)local_accessor;
    }

    static_assert(tensor::get_token_if_present<"tensor_absent">() == nullptr);
    if (const auto* token = tensor::get_token_if_present<"tensor_absent">()) {
        TensorAccessor accessor(*token);
        LocalTensorAccessor<uint32_t> local_accessor(*token);
        (void)accessor;
        (void)local_accessor;
    }
}
)"};

    spec.kernels = {dm_kernel};
    spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor", tt::tt_metal::BufferType::L1)};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "tensor_present");
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_GetTokenIfPresentDisambiguatesMultipleBindingsJITSmoke) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    auto dfb_b = MakeMinimalDFB("dfb_1");
    dfb_b.data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers.push_back(dfb_b);
    spec.kernels[0].dfb_bindings[0].accessor_name = "dfb_a";
    spec.kernels[0].dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_1"}, "dfb_b"));
    spec.kernels[1].dfb_bindings[0].accessor_name = "dfb_a";
    spec.kernels[1].dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_1"}, "dfb_b"));

    spec.tensor_parameters = {
        MakeMinimalTensorParameter("t0", tt::tt_metal::BufferType::L1),
        MakeMinimalTensorParameter("t1", tt::tt_metal::BufferType::L1),
    };
    BindTensorParameterToKernel(spec.kernels[0], "t0", "tensor_a");
    BindTensorParameterToKernel(spec.kernels[0], "t1", "tensor_b");

    spec.scratchpads = {
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024},
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_1"}, .size_per_node = 1024},
    };
    spec.kernels[0].scratchpad_bindings = {
        KernelSpec::ScratchpadBinding{
            .scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "scratch_a"},
        KernelSpec::ScratchpadBinding{
            .scratchpad_spec_name = ScratchpadSpecName{"scratch_1"}, .accessor_name = "scratch_b"},
    };

    spec.kernels[0].source = KernelSpec::SourceCode{R"(
#include "api/tensor/local_tensor_accessor.h"

void kernel_main() {
    static_assert(dfb::get_token_if_present<"dfb_a">() == &dfb::dfb_a);
    if (const auto* token = dfb::get_token_if_present<"dfb_a">()) {
        DataflowBuffer buf(*token);
        (void)buf;
    }

    static_assert(dfb::get_token_if_present<"dfb_b">() == &dfb::dfb_b);
    if (const auto* token = dfb::get_token_if_present<"dfb_b">()) {
        DataflowBuffer buf(*token);
        (void)buf;
    }

    static_assert(tensor::get_token_if_present<"tensor_a">() == &tensor::tensor_a);
    if (const auto* token = tensor::get_token_if_present<"tensor_a">()) {
        TensorAccessor accessor(*token);
        LocalTensorAccessor<uint32_t> local_accessor(*token);
        (void)accessor;
        (void)local_accessor;
    }

    static_assert(tensor::get_token_if_present<"tensor_b">() == &tensor::tensor_b);
    if (const auto* token = tensor::get_token_if_present<"tensor_b">()) {
        TensorAccessor accessor(*token);
        LocalTensorAccessor<uint32_t> local_accessor(*token);
        (void)accessor;
        (void)local_accessor;
    }

    static_assert(scratch::get_token_if_present<"scratch_a">() == &scratch::scratch_a);
    if (const auto* token = scratch::get_token_if_present<"scratch_a">()) {
        Scratchpad<int32_t> pad(*token);
        (void)pad;
    }

    static_assert(scratch::get_token_if_present<"scratch_b">() == &scratch::scratch_b);
    if (const auto* token = scratch::get_token_if_present<"scratch_b">()) {
        Scratchpad<int32_t> pad(*token);
        (void)pad;
    }
}
)"};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
