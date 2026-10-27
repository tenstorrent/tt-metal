// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MergeProgramRunArgs: pure data helper, no device needed.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace tt::tt_metal::experimental {
namespace {

const ProgramRunArgs::KernelRunArgs* FindKernel(const ProgramRunArgs& p, const std::string& name) {
    for (const auto& kra : p.kernel_run_args) {
        if (kra.kernel == KernelSpecName{name}) {
            return &kra;
        }
    }
    return nullptr;
}

TEST(MergeProgramRunArgs, CPU_UnionsDisjointCRTAsForSameKernel) {
    // The motivating pattern: two pieces carrying disjoint named CRTAs for the SAME kernel merge
    // into one kernel entry holding both.
    ProgramRunArgs first_piece;
    first_piece.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .common_runtime_arg_values = {{"keep", 10}},
    });
    ProgramRunArgs second_piece;
    second_piece.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .common_runtime_arg_values = {{"change", 20}},
    });

    std::vector<ProgramRunArgs> rest{second_piece};
    ProgramRunArgs merged = MergeProgramRunArgs(std::move(first_piece), rest);

    const auto* dm = FindKernel(merged, "dm_kernel");
    ASSERT_NE(dm, nullptr);
    auto keep = dm->common_runtime_arg_values.get("keep");
    auto change = dm->common_runtime_arg_values.get("change");
    ASSERT_TRUE(keep.has_value());
    ASSERT_TRUE(change.has_value());
    EXPECT_EQ(*keep, 10u);
    EXPECT_EQ(*change, 20u);
}

TEST(MergeProgramRunArgs, CPU_ConflictingArgFails) {
    ProgramRunArgs a;
    a.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .common_runtime_arg_values = {{"x", 1}},
    });
    ProgramRunArgs b;
    b.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .common_runtime_arg_values = {{"x", 2}},  // same arg in both inputs → conflict
    });
    std::vector<ProgramRunArgs> rest{b};
    EXPECT_THAT(
        [&] { MergeProgramRunArgs(std::move(a), rest); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("specified in more than one ProgramRunArgs")));
}

TEST(MergeProgramRunArgs, CPU_AppendsDistinctKernel) {
    ProgramRunArgs a;
    a.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .common_runtime_arg_values = {{"x", 1}},
    });
    ProgramRunArgs b;
    b.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"compute_kernel"},
        .common_runtime_arg_values = {{"y", 2}},
    });
    std::vector<ProgramRunArgs> rest{b};
    ProgramRunArgs merged = MergeProgramRunArgs(std::move(a), rest);
    EXPECT_NE(FindKernel(merged, "dm_kernel"), nullptr);
    EXPECT_NE(FindKernel(merged, "compute_kernel"), nullptr);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
