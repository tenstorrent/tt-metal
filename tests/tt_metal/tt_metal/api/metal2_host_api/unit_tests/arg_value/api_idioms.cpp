// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// The API idioms that must keep compiling with ArgValue.
//
// Argument values were plain uint32_t before ArgValue existed, and use sites rely on the
// implicit conversions into it. This file is mostly a compile test: if an idiom here stops
// compiling, existing user code breaks the same way. The EXPECTs check that the idioms also
// produce the values they did before.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include <tt-metalium/experimental/command_list.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace tt::tt_metal::experimental {
namespace {

TEST(ApiIdiomsTest, CPU_CompileTimeArgIdiomsCompile) {
    KernelSpec k{.compile_time_args = {{"Ht", 8}}, .runtime_arg_schema = {.runtime_arg_names = {"a", "b"}}};
    k.compile_time_args["Wt"] = 4u;
    k.compile_time_args.emplace("flag", true);
    k.compile_time_args.insert({"neg", -1});

    EXPECT_TRUE(*k.compile_time_args.get("Ht") == 8u);
    EXPECT_EQ(k.compile_time_args.get("flag")->to_uint32(), 1u);
    EXPECT_EQ(k.compile_time_args.get("neg")->to_uint32(), 0xFFFFFFFFu);
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

    EXPECT_EQ((r.runtime_arg_values["a"][NodeCoord{0, 1}].to_uint32()), 12u);
    EXPECT_EQ((r.runtime_arg_values["a"][NodeCoord{0, 2}].to_uint32()), 0xFFFFFFFFu);
    EXPECT_EQ((r.runtime_arg_values["a"][NodeCoord{0, 3}].to_uint32()), 1u);
    EXPECT_EQ((r.runtime_arg_values["c"][NodeCoord{0, 0}].to_uint32()), 3u);

    r.runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"a", 1}});
    r.common_runtime_arg_values = {{"c", 3}};
    EXPECT_EQ(r.runtime_arg_values.size(), 1u);
    EXPECT_TRUE(*r.common_runtime_arg_values.get("c") == 3);

    // A legacy name -> value table, iterated into the run args.
    Table<std::string, uint32_t> legacy = {{"lead", 4}};
    for (const auto& [name, value] : legacy) {
        r.runtime_arg_values[name][NodeCoord{1, 1}] = value;
    }
    EXPECT_EQ((r.runtime_arg_values["lead"][NodeCoord{1, 1}].to_uint32()), 4u);
}

TEST(ApiIdiomsTest, CPU_CommandListPatchIdiomsCompile) {
    std::size_t n = 9;
    CmdListArgPatch patch{
        .runtime_args = {{CmdListRuntimeArgName{"p"}, 5}},
        .common_runtime_args = {{CmdListCommonRuntimeArgName{"q"}, n}},
    };
    patch.runtime_args[CmdListRuntimeArgName{"r"}] = -1;

    EXPECT_EQ(patch.common_runtime_args.get(CmdListCommonRuntimeArgName{"q"})->to_uint32(), 9u);
    EXPECT_EQ(patch.runtime_args.get(CmdListRuntimeArgName{"r"})->to_uint32(), 0xFFFFFFFFu);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
