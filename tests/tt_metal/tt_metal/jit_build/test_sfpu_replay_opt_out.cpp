// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compile-only coverage for internal::DISABLE_SFPU_REPLAY_OPTIMIZATION_DEFINE — the kernel define that
// compiles a compute kernel's TRISC binaries with -mno-tt-tensix-optimize-replay
// (tenstorrent/tt-metal#58433).
//
//  - Define absent: no TRISC compile recipe carries the flag (default builds are unchanged).
//  - Define present: every TRISC recipe (unpack, math, pack) carries it, nothing else in their cflags
//    changes, and the kernel's JIT cache key changes. The kernel still compiles.
//  - ProgramDescriptor: the define in KernelDescriptor::defines reaches the kernel.
//
// Compile-only; see jit_build_device_fixture.hpp for which device each runner uses. On Quasar only the
// define-absent case runs, built through Metal 2.0: the opt-out is not implemented there, and Quasar
// rejects the legacy and ProgramDescriptor kernel APIs.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <internal/kernel_build_options.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build_device_fixture.hpp"

namespace tt::tt_metal {

namespace {

constexpr const char* kBlankComputeKernel = "tests/tt_metal/tt_metal/test_kernels/compute/blank.cpp";
constexpr const char* kNoReplayFlag = "-mno-tt-tensix-optimize-replay";
constexpr const char* kNoReplayDefine = internal::DISABLE_SFPU_REPLAY_OPTIMIZATION_DEFINE;
constexpr const char* kQuasarNotImplemented = "SFPU replay opt-out is not implemented for Quasar";

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
class SfpuReplayOptOutFixture : public JitBuildDeviceFixture {
protected:
    // Legacy-API build (Wormhole/Blackhole only; Quasar rejects legacy compute kernels).
    std::shared_ptr<Kernel> compile_blank(bool disable_sfpu_replay_optimization) {
        distributed::MeshDevice* device = devices_.at(0).get();
        Program program = CreateProgram();
        KernelHandle handle = CreateKernel(
            program,
            kBlankComputeKernel,
            CoreCoord{0, 0},
            ComputeConfig{
                .defines = disable_sfpu_replay_optimization ? std::map<std::string, std::string>{{kNoReplayDefine, "1"}}
                                                            : std::map<std::string, std::string>{}});
        program.impl().compile(device);
        return program.impl().get_kernel(handle);
    }

    // Quasar: the blank compute kernel, without the define, built through Metal 2.0 and compiled.
    std::shared_ptr<Kernel> compile_blank_metal2() {
        return compile_single_kernel_spec(KernelSpec{
            .unique_id = KernelSpecName{"blank"},
            .source = std::filesystem::path(kBlankComputeKernel),
            .hw_config = ComputeHardwareConfig{},
        });
    }
};

TEST_F(SfpuReplayOptOutFixture, DefineAbsentRecipesOmitTheFlag) {
    auto kernel = is_quasar() ? compile_blank_metal2() : compile_blank(/*disable_sfpu_replay_optimization=*/false);
    for (uint32_t processor_id : binary_processor_types(*kernel)) {
        EXPECT_EQ(recipe_cflags(kernel, processor_id).find(kNoReplayFlag), std::string::npos)
            << "trisc" << processor_id;
    }
}

TEST_F(SfpuReplayOptOutFixture, DefineFlagReachesEveryTriscRecipe) {
    JIT_BUILD_SKIP_ON_QUASAR(kQuasarNotImplemented);
    auto kernel_off = compile_blank(/*disable_sfpu_replay_optimization=*/false);
    auto kernel_on = compile_blank(/*disable_sfpu_replay_optimization=*/true);

    // Opting out must re-key the JIT cache.
    EXPECT_NE(kernel_on->get_full_kernel_name(), kernel_off->get_full_kernel_name());

    for (uint32_t processor_id : binary_processor_types(*kernel_on)) {
        const std::string on = recipe_cflags(kernel_on, processor_id);
        EXPECT_NE(on.find(kNoReplayFlag), std::string::npos) << "trisc" << processor_id;
        // Nothing else in the recipe changes.
        EXPECT_EQ(without_flag(on), without_flag(recipe_cflags(kernel_off, processor_id))) << "trisc" << processor_id;
    }
}

TEST_F(SfpuReplayOptOutFixture, ProgramDescriptorCarriesTheDefine) {
    JIT_BUILD_SKIP_ON_QUASAR(
        std::string(kQuasarNotImplemented) +
        "; also, ProgramDescriptor lowers to the legacy CreateKernel, which Quasar rejects");
    ProgramDescriptor descriptor;
    descriptor.kernels.push_back(KernelDescriptor{
        .kernel_source = kBlankComputeKernel,
        .core_ranges = CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0})),
        .defines = {{kNoReplayDefine, "1"}},
        .config = ComputeConfigDescriptor{},
    });
    Program program(*devices_.at(0), descriptor);
    program.impl().compile(devices_.at(0).get());
    auto kernel = program.impl().get_kernel(0);
    EXPECT_NE(recipe_cflags(kernel, /*math*/ 1).find(kNoReplayFlag), std::string::npos);
}

}  // namespace tt::tt_metal
