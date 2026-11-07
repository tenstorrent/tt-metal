// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakePerNodeTensorParameter;
using test_helpers::ProgramSpecTestGen1;

TEST_F(ProgramSpecTestGen1, CPU_PerNodeTensorBinding_TakesRtaSlotAfterNamedRtas) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.kernels[0].runtime_arg_schema = {.runtime_arg_names = {"foo", "bar"}, .common_runtime_arg_names = {"baz"}};
    spec.tensor_parameters = {
        MakePerNodeTensorParameter("per_node_a", /*num_cores=*/2),
        MakeMinimalTensorParameter("plain_tensor"),
        MakePerNodeTensorParameter("per_node_b", /*num_cores=*/2)};
    BindTensorParameterToKernel(spec.kernels[0], "per_node_a", "per_node_a");
    BindTensorParameterToKernel(spec.kernels[0], "plain_tensor", "plain_ta");
    BindTensorParameterToKernel(spec.kernels[0], "per_node_b", "per_node_b");
    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    const auto& handles = kernel->tensor_binding_handles();
    ASSERT_EQ(handles.size(), 3u);
    const auto& per_node_a = handles[0];
    const auto& plain = handles[1];
    const auto& per_node_b = handles[2];
    // Each per-node binding takes one RTA word after the 2 named RTAs, in binding order, so the RTA varargs
    // start at word 4.
    EXPECT_TRUE(per_node_a.is_per_node);
    EXPECT_EQ(per_node_a.addr_rta_offset, 2u * sizeof(uint32_t));
    EXPECT_TRUE(per_node_b.is_per_node);
    EXPECT_EQ(per_node_b.addr_rta_offset, 3u * sizeof(uint32_t));
    const KernelRtaLayout rta_layout = kernel->get_rta_layout();
    EXPECT_EQ(rta_layout.num_named_words, 2u);
    EXPECT_EQ(rta_layout.binding_section_words, 2u);
    EXPECT_EQ(rta_layout.vararg_section_offset, 4u);
    // The plain binding keeps its CRTA address word right after the named CRTA, and it is the only word in the
    // CRTA binding section: the per-node bindings take no CRTA words.
    EXPECT_FALSE(plain.is_per_node);
    EXPECT_EQ(plain.addr_crta_offset, 1u * sizeof(uint32_t));
    const KernelCrtaLayout crta_layout = kernel->get_crta_layout();
    EXPECT_EQ(crta_layout.binding_section_words, 1u);
    EXPECT_EQ(crta_layout.vararg_section_offset, 2u);
}

TEST_F(ProgramSpecTestGen1, CPU_PerNodeTensorBinding_RejectsDynamicTensorShape) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.tensor_parameters = {MakePerNodeTensorParameter("per_node", /*num_cores=*/2)};
    spec.tensor_parameters[0].relaxations = {.dynamic_tensor_shape = true};
    BindTensorParameterToKernel(spec.kernels[0], "per_node", "per_node");
    EXPECT_ANY_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
