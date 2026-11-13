// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::MakePerNodeTensorParameter;
using test_helpers::ProgramSpecTestGen1;

TEST_F(ProgramSpecTestGen1, CPU_PerNodeTensorBindingLocalTensorAccessorJITSmoke) {
    // A DM and a compute kernel, each with 1 named RTA and a per-node tensor bound as "per_node". The source
    // checks the generated token type and builds LocalTensorAccessors from the token.
    const char* source = R"(
#include <type_traits>
#include "api/tensor/local_tensor_accessor.h"
void kernel_main() {
    // The binding's address is in the RTA word right after the named RTA: byte offset 4.
    static_assert(std::is_same_v<tensor::per_node_t, ::tensor_accessor::PerNodeTensorBindingToken<0u, 4u>>);
    static_assert(get_dataformat(tensor::per_node) == DataFormat::Float16_b);
    LocalTensorAccessor<uint32_t> accessor(tensor::per_node);
    (void)accessor.local_mem();
    auto accessors = make_local_tensor_accessors<uint32_t>(std::make_tuple(tensor::per_node));
    (void)accessors;
}
)";
    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    auto compute_kernel = MakeMinimalGen1ComputeKernel("compute_kernel");
    for (KernelSpec* kernel : {&dm_kernel, &compute_kernel}) {
        kernel->source = KernelSpec::SourceCode{source};
        kernel->runtime_arg_schema = {.runtime_arg_names = {"foo"}};
        BindTensorParameterToKernel(*kernel, "t", "per_node");
    }
    const ProgramSpec spec{
        .name = "per_node_ta_smoke",
        .kernels = {dm_kernel, compute_kernel},
        .tensor_parameters = {MakePerNodeTensorParameter("t", /*num_cores=*/1)},
        .work_units = {MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel", "compute_kernel"})},
    };
    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
