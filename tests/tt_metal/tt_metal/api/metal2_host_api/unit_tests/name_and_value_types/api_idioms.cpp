// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// The API idioms that must keep compiling with AccessorName / ArgName and ArgValue.
//
// Names and argument values were plain std::string and uint32_t before these types existed,
// and use sites rely on the implicit conversions into them. This file is mostly a compile
// test: if an idiom here stops compiling, existing user code breaks the same way. The
// EXPECTs check that the idioms also produce the values they did before.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <tt-metalium/experimental/command_list.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "impl/metal2_host_api/arg_value_access.hpp"

namespace tt::tt_metal::experimental {
namespace {

std::uint32_t word(const ArgValue& value) { return ArgValueAccess::u32(value); }

TEST(ApiIdiomsTest, CPU_KernelSpecIdiomsCompile) {
    KernelSpec k{.compile_time_args = {{"Ht", 8}}, .runtime_arg_schema = {.runtime_arg_names = {"a", "b"}}};
    k.runtime_arg_schema.runtime_arg_names.push_back("c");
    k.runtime_arg_schema.runtime_arg_names.insert(k.runtime_arg_schema.runtime_arg_names.end(), {"d", "e"});
    k.runtime_arg_schema.common_runtime_arg_names.emplace_back("f");

    const std::string name = "g";
    k.runtime_arg_schema.runtime_arg_names.push_back(name);
    k.compile_time_args["Wt"] = 4u;
    k.compile_time_args.emplace("flag", true);

    EXPECT_EQ(k.runtime_arg_schema.runtime_arg_names.size(), 6u);
    EXPECT_TRUE(k.runtime_arg_schema.runtime_arg_names[0] == "a");
    EXPECT_TRUE(*k.compile_time_args.get("Ht") == 8u);
    EXPECT_EQ(word(*k.compile_time_args.get("flag")), 1u);
}

TEST(ApiIdiomsTest, CPU_BindingIdiomsCompile) {
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

TEST(ApiIdiomsTest, CPU_RunArgsIdiomsCompile) {
    KernelRunArgs r{.kernel = KernelSpecName{"k"}};
    r.runtime_arg_values["a"][NodeCoord{0, 0}] = 1;
    std::size_t n = 12;
    uint32_t u = 3;
    bool flag = true;
    r.runtime_arg_values["a"][NodeCoord{0, 1}] = n;
    r.runtime_arg_values["a"][NodeCoord{0, 2}] = -1;  // wraps, as for a uint32_t
    r.runtime_arg_values["a"][NodeCoord{0, 3}] = flag;
    AddRuntimeArgsForNode(r.runtime_arg_values, NodeCoord{0, 0}, {{"b", 2}, {"c", u}});

    EXPECT_EQ(word(r.runtime_arg_values["a"][NodeCoord{0, 1}]), 12u);
    EXPECT_EQ(word(r.runtime_arg_values["a"][NodeCoord{0, 2}]), 0xFFFFFFFFu);
    EXPECT_EQ(word(r.runtime_arg_values["a"][NodeCoord{0, 3}]), 1u);
    EXPECT_EQ(word(r.runtime_arg_values["c"][NodeCoord{0, 0}]), 3u);

    r.runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"a", 1}});
    r.common_runtime_arg_values = {{"c", 3}};
    EXPECT_EQ(r.runtime_arg_values.size(), 1u);
    EXPECT_TRUE(*r.common_runtime_arg_values.get("c") == 3);

    // A legacy name -> value table, iterated into the run args.
    Table<std::string, uint32_t> legacy = {{"lead", 4}};
    for (const auto& [name, value] : legacy) {
        r.runtime_arg_values[name][NodeCoord{1, 1}] = value;
    }
    EXPECT_EQ(word(r.runtime_arg_values["lead"][NodeCoord{1, 1}]), 4u);
}

TEST(ApiIdiomsTest, CPU_CommandListPatchIdiomsCompile) {
    std::size_t n = 9;
    CmdListArgPatch patch{
        .runtime_args = {{CmdListRuntimeArgName{"p"}, 5}},
        .common_runtime_args = {{CmdListCommonRuntimeArgName{"q"}, n}},
    };
    patch.runtime_args[CmdListRuntimeArgName{"r"}] = -1;

    EXPECT_EQ(word(*patch.common_runtime_args.get(CmdListCommonRuntimeArgName{"q"})), 9u);
    EXPECT_EQ(word(*patch.runtime_args.get(CmdListRuntimeArgName{"r"})), 0xFFFFFFFFu);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
