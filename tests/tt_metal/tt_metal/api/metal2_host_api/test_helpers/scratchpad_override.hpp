// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>
#include <numeric>
#include <vector>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/host_api.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental::test_helpers {

inline bool SupportsScratchpadOverrideTest(const distributed::MeshDevice& device) {
    return (device.arch() == tt::ARCH::WORMHOLE_B0 || device.arch() == tt::ARCH::BLACKHOLE) &&
           device.compute_with_storage_grid_size().x >= 4;
}

template <typename DispatchFixture>
class ScratchpadPlacementFixture : public DispatchFixture {
protected:
    distributed::MeshDevice& fixture_device() { return *this->devices_.at(0); }

    void SetUp() override {
        DispatchFixture::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        if (!SupportsScratchpadOverrideTest(fixture_device())) {
            GTEST_SKIP() << "Requires Gen1 hardware and four columns";
        }
    }
};

inline constexpr uint32_t scratchpad_report_address = 100 * 1024;

inline ProgramSpec MakeGuardedScratchpadSpecForTest() {
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
    return ProgramSpec{
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
}

inline ProgramRunArgs MakeGuardedScratchpadArgsForTest(uint32_t pattern) {
    ProgramRunArgs::KernelRunArgs a{.kernel = KernelSpecName{"a"}};
    for (uint32_t x = 0; x < 2; ++x) {
        AddRuntimeArgsForNode(
            a.runtime_arg_values,
            {x, 0},
            {{"report_addr", scratchpad_report_address}, {"enabled", x == 0}, {"pattern", pattern}});
    }
    ProgramRunArgs args;
    args.kernel_run_args = {
        a,
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"b"},
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                {2, 0}, {{"report_addr", scratchpad_report_address}, {"enabled", 1}, {"pattern", pattern}})}};
    return args;
}

inline uint32_t ScratchpadAddressForTest(const Program& program) {
    return program.impl().get_kernel_by_spec_name("a")->scratchpad_binding_handles()[0].allocated_address;
}

inline void InitializeScratchpadReportsForTest(distributed::MeshDevice& device, const Program& program) {
    std::vector<uint32_t> zero_report(2, 0);
    for (uint32_t x = 0; x < 3; ++x) {
        slow_dispatch::WriteToL1(device, {x, 0}, scratchpad_report_address, zero_report);
    }
    // The extra reserved node has no kernel and must remain untouched.
    std::vector<uint32_t> sentinel(16, 0xDEADBEEFu);
    slow_dispatch::WriteToL1(device, {3, 0}, ScratchpadAddressForTest(program), sentinel);
}

inline std::vector<std::vector<uint32_t>> ReadScratchpadResultsForTest(
    distributed::MeshDevice& device, const Program& program) {
    std::vector<std::vector<uint32_t>> results(6);
    for (uint32_t x = 0; x < 3; ++x) {
        slow_dispatch::ReadFromL1(device, {x, 0}, scratchpad_report_address, 8, results[x]);
    }
    const auto address = ScratchpadAddressForTest(program);
    slow_dispatch::ReadFromL1(device, {0, 0}, address, 64, results[3]);
    slow_dispatch::ReadFromL1(device, {2, 0}, address, 64, results[4]);
    slow_dispatch::ReadFromL1(device, {3, 0}, address, 64, results[5]);
    return results;
}

inline std::vector<std::vector<uint32_t>> ExpectedScratchpadResultsForTest(uint32_t address, uint32_t pattern) {
    std::vector<uint32_t> contents(16);
    std::iota(contents.begin(), contents.end(), pattern);
    return {
        {address, pattern},
        {0, pattern},
        {address, pattern},
        contents,
        contents,
        std::vector<uint32_t>(16, 0xDEADBEEFu)};
}

inline void EnqueueScratchpadOverrideForTest(
    distributed::MeshDevice& device, distributed::MeshWorkload& workload, bool fast_dispatch) {
    if (fast_dispatch) {
        distributed::EnqueueMeshWorkload(device.mesh_command_queue(), workload, true);
    } else {
        auto& [range, program] = *workload.get_programs().begin();
        // LaunchProgram transfers ownership into its returned workload.
        auto completed = LaunchProgram(device, std::move(program));
        program = std::move(completed.get_programs().at(range));
    }
}

}  // namespace tt::tt_metal::experimental::test_helpers
