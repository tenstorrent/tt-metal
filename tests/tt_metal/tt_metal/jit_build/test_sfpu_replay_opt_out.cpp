// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Device-free (mock Blackhole) coverage for ComputeConfig::disable_sfpu_replay_optimization — the
// per-kernel opt-out that compiles a kernel's TRISC binaries with -mno-tt-tensix-optimize-replay
// (tenstorrent/tt-metal#58433).
//
//  - Knob off: no TRISC compile recipe carries the flag (default builds are unchanged).
//  - Knob on: all three TRISC recipes (unpack, math, pack) carry it, nothing else in them changes, and
//    the kernel's JIT cache key changes. The kernel still compiles.
//  - ProgramDescriptor: ComputeConfigDescriptor::disable_sfpu_replay_optimization reaches the kernel.
//
// Everything here runs without silicon: the fixture configures a mock Blackhole device, and
// compilation is pure host-side JIT.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "mock_blackhole_fixture.hpp"

namespace tt::tt_metal {

namespace {

constexpr const char* kBlankComputeKernel = "tests/tt_metal/tt_metal/test_kernels/compute/blank.cpp";
constexpr const char* kNoReplayFlag = "-mno-tt-tensix-optimize-replay";

// Removes every occurrence of the opt-out flag, so the rest of a recipe can be compared.
std::string without_flag(std::string cflags) {
    const std::string flag(kNoReplayFlag);
    for (auto pos = cflags.find(flag); pos != std::string::npos; pos = cflags.find(flag)) {
        cflags.erase(pos, flag.size());
    }
    std::string squeezed;
    for (char c : cflags) {
        if (!(c == ' ' && !squeezed.empty() && squeezed.back() == ' ')) {
            squeezed += c;
        }
    }
    return squeezed;
}

}  // namespace

// Fixture lives in the named namespace (see test_trisc2_rvv.cpp: gcc Unity builds reject an
// anonymous-namespace gtest base with -Werror=subobject-linkage).
class SfpuReplayOptOutMockBlackholeFixture : public MockBlackholeMeshDispatchFixture {
protected:
    std::shared_ptr<Kernel> compile_blank(bool disable_sfpu_replay_optimization) {
        distributed::MeshDevice* device = devices_.at(0).get();
        Program program = CreateProgram();
        KernelHandle handle = CreateKernel(
            program,
            kBlankComputeKernel,
            CoreCoord{0, 0},
            ComputeConfig{.disable_sfpu_replay_optimization = disable_sfpu_replay_optimization});
        program.impl().compile(device);
        return program.impl().get_kernel(handle);
    }

    // The kernel's exported compile recipe cflags for one compute processor (0=unpack, 1=math, 2=pack).
    std::string recipe_cflags(const std::shared_ptr<Kernel>& kernel, int processor_id) {
        return kernel_build_state(*kernel, processor_id).export_target_recipe(kernel.get()).cflags;
    }
};

TEST_F(SfpuReplayOptOutMockBlackholeFixture, KnobOffRecipesOmitTheFlag) {
    auto kernel = compile_blank(/*disable_sfpu_replay_optimization=*/false);
    for (int processor_id = 0; processor_id < 3; processor_id++) {
        EXPECT_EQ(recipe_cflags(kernel, processor_id).find(kNoReplayFlag), std::string::npos)
            << "trisc" << processor_id;
    }
}

TEST_F(SfpuReplayOptOutMockBlackholeFixture, KnobOnFlagReachesEveryTriscRecipe) {
    auto kernel_off = compile_blank(/*disable_sfpu_replay_optimization=*/false);
    auto kernel_on = compile_blank(/*disable_sfpu_replay_optimization=*/true);

    // Opting out must re-key the JIT cache.
    EXPECT_NE(kernel_on->get_full_kernel_name(), kernel_off->get_full_kernel_name());

    for (int processor_id = 0; processor_id < 3; processor_id++) {
        const std::string on = recipe_cflags(kernel_on, processor_id);
        EXPECT_NE(on.find(kNoReplayFlag), std::string::npos) << "trisc" << processor_id;
        // Nothing else in the recipe changes.
        EXPECT_EQ(without_flag(on), without_flag(recipe_cflags(kernel_off, processor_id))) << "trisc" << processor_id;
    }
}

TEST_F(SfpuReplayOptOutMockBlackholeFixture, ProgramDescriptorCarriesTheKnob) {
    ProgramDescriptor descriptor;
    descriptor.kernels.push_back(KernelDescriptor{
        .kernel_source = kBlankComputeKernel,
        .core_ranges = CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0})),
        .config = ComputeConfigDescriptor{.disable_sfpu_replay_optimization = true},
    });
    Program program(descriptor);
    program.impl().compile(devices_.at(0).get());
    auto kernel = program.impl().get_kernel(0);
    EXPECT_NE(recipe_cflags(kernel, 1).find(kNoReplayFlag), std::string::npos);
}

}  // namespace tt::tt_metal
