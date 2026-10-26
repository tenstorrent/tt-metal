// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// The API idioms that must keep compiling with AccessorName / ArgName.
//
// Kernel-visible names were plain std::string before AccessorName existed, and use sites rely
// on the implicit conversions into it. This file is mostly a compile test: if an idiom here
// stops compiling, existing user code breaks the same way. The EXPECTs check that the idioms
// also produce the names they did before.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <string>
#include <string_view>

#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace tt::tt_metal::experimental {
namespace {

TEST(AccessorNameIdiomsTest, CPU_ArgNameIdiomsCompile) {
    KernelSpec k{.compile_time_args = {{"Ht", 8}}, .runtime_arg_schema = {.runtime_arg_names = {"a", "b"}}};
    k.runtime_arg_schema.runtime_arg_names.push_back("c");
    k.runtime_arg_schema.runtime_arg_names.insert(k.runtime_arg_schema.runtime_arg_names.end(), {"d", "e"});
    k.runtime_arg_schema.common_runtime_arg_names.emplace_back("f");

    const std::string name = "g";
    k.runtime_arg_schema.runtime_arg_names.push_back(name);
    k.compile_time_args[name] = 4u;

    KernelRunArgs r{.kernel = KernelSpecName{"k"}};
    r.runtime_arg_values[name][NodeCoord{0, 0}] = 1;

    EXPECT_EQ(k.runtime_arg_schema.runtime_arg_names.size(), 6u);
    EXPECT_TRUE(k.runtime_arg_schema.runtime_arg_names[0] == "a");
    EXPECT_TRUE(k.runtime_arg_schema.runtime_arg_names[5] == name);
    EXPECT_TRUE(k.compile_time_args.contains("g"));
    EXPECT_TRUE(r.runtime_arg_values.contains("g"));
}

TEST(AccessorNameIdiomsTest, CPU_BindingIdiomsCompile) {
    DFBBinding d{.dfb_spec_name = DFBSpecName{"x"}, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER};
    DFBBinding unnamed{.dfb_spec_name = DFBSpecName{"x"}, .endpoint_type = DFBEndpointType::CONSUMER};
    auto p = ProducerOf(DFBSpecName{"x"}, "out");

    const std::string str = "out";
    const std::string_view view = "out";
    auto from_string = ProducerOf(DFBSpecName{"x"}, str);
    auto from_view = ProducerOf(DFBSpecName{"x"}, view);

    TensorBinding positional{TensorParamName{"t"}, "v_beta"};
    TensorBinding assigned;
    assigned.accessor_name = str + "_" + std::to_string(2);

    KernelAdvancedOptions::TensorBindingSequence sequence{.sequence_name = "inputs", .members = {"in0", "in1"}};

    EXPECT_TRUE(d.accessor_name == "in");
    EXPECT_TRUE(unnamed.accessor_name.view().empty());
    EXPECT_TRUE(p.accessor_name == from_string.accessor_name);
    EXPECT_TRUE(from_view.accessor_name == str);
    EXPECT_TRUE(positional.accessor_name == "v_beta");
    EXPECT_TRUE(assigned.accessor_name == "out_2");
    EXPECT_TRUE(sequence.members[1] == "in1");
}

}  // namespace
}  // namespace tt::tt_metal::experimental
