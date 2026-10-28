// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/host_api.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental::test_helpers {

inline void CheckScratchpadPlacementOverride(distributed::MeshDevice& device, bool fast_dispatch) {
    if (device.arch() != tt::ARCH::WORMHOLE_B0 && device.arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Requires Wormhole or Blackhole";
    }
    if (device.compute_with_storage_grid_size().x < 4) {
        GTEST_SKIP() << "Requires four worker columns";
    }
    constexpr uint32_t report_addr = 100 * 1024;
    auto a = MakeMinimalGen1DMKernel("a");
    a.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    volatile tt_l1_ptr uint32_t* report =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::report_addr));
    report[0] = 0;
    if (get_arg(args::enabled)) {
        auto pad = Scratchpad<uint32_t>(scratch::pad);
        for (uint32_t i = 0; i < pad.size(); ++i) {
            pad[i] = get_arg(args::pattern) + i;
        }
        report[0] = static_cast<uint32_t>(local_mem::detail::AddressAccess::address(pad.local_mem()));
    }
    report[1] = get_arg(args::pattern);
}
)"};
    a.runtime_arg_schema.runtime_arg_names = {"report_addr", "enabled", "pattern"};
    a.scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    auto b = a;
    b.unique_id = KernelSpecName{"b"};
    ProgramSpec spec{
        .kernels = {a, b},
        .scratchpads = {ScratchpadSpec{
            .unique_id = ScratchpadSpecName{"pad"},
            .size_per_node = 64,
            .advanced_options =
                {.override_placement =
                     CoreRangeSet(std::vector<CoreRange>{CoreRange({0, 0}), CoreRange({2, 0}, {3, 0})})}}},
        .work_units = {
            MakeMinimalWorkUnit("a", NodeRange({0, 0}, {1, 0}), {"a"}),
            MakeMinimalWorkUnit("b", NodeCoord{2, 0}, {"b"})}};
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinateRange devices(device.shape());
    workload.add_program(devices, MakeProgramFromSpec(device, spec));
    auto& program = workload.get_programs().at(devices);
    const uint32_t base =
        program.impl().get_kernel_by_spec_name("a")->scratchpad_binding_handles()[0].allocated_address;
    ASSERT_NE(base, 0u);

    for (uint32_t repeat = 0; repeat < 2; ++repeat) {
        const uint32_t pattern = 0xC0DE0000u + repeat * 256;
        ProgramRunArgs params;
        ProgramRunArgs::KernelRunArgs args_a{.kernel = KernelSpecName{"a"}};
        for (uint32_t x = 0; x < 2; ++x) {
            AddRuntimeArgsForNode(
                args_a.runtime_arg_values,
                NodeCoord{x, 0},
                {{"report_addr", report_addr}, {"enabled", x == 0}, {"pattern", pattern}});
        }
        params.kernel_run_args = {
            args_a,
            ProgramRunArgs::KernelRunArgs{
                .kernel = KernelSpecName{"b"},
                .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                    {2, 0}, {{"report_addr", report_addr}, {"enabled", 1}, {"pattern", pattern}})}};
        SetProgramRunArgs(program, params);
        std::vector<uint32_t> sentinel(16, 0xDEADBEEFu);
        // The extra reserved node has no kernel and must remain untouched.
        slow_dispatch::WriteToL1(device, {3, 0}, base, sentinel);
        std::vector<uint32_t> zero_report(2, 0);
        for (uint32_t x = 0; x < 3; ++x) {
            slow_dispatch::WriteToL1(device, {x, 0}, report_addr, zero_report);
        }
        if (fast_dispatch) {
            distributed::EnqueueMeshWorkload(device.mesh_command_queue(), workload, true);
        } else {
            // LaunchProgram transfers ownership into its returned workload.
            auto completed = LaunchProgram(device, std::move(program));
            program = std::move(completed.get_programs().at(devices));
        }
        for (uint32_t x = 0; x < 3; ++x) {
            std::vector<uint32_t> report;
            slow_dispatch::ReadFromL1(device, {x, 0}, report_addr, 8, report);
            ASSERT_EQ(report.size(), 2u);
            EXPECT_EQ(report[1], pattern);
            EXPECT_EQ(report[0], x == 1 ? 0u : base);
            if (x != 1) {
                std::vector<uint32_t> contents;
                slow_dispatch::ReadFromL1(device, {x, 0}, base, 64, contents);
                ASSERT_EQ(contents.size(), 16u);
                for (uint32_t i = 0; i < 16; ++i) {
                    EXPECT_EQ(contents[i], pattern + i);
                }
            }
        }
        std::vector<uint32_t> untouched;
        slow_dispatch::ReadFromL1(device, {3, 0}, base, 64, untouched);
        EXPECT_EQ(untouched, sentinel);
    }
}

}  // namespace tt::tt_metal::experimental::test_helpers
