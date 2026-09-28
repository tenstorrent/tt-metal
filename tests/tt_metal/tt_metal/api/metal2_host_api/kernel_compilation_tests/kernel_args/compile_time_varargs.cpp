// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile tests for compile-time varargs, with and without kernel asserts (mock Wormhole).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <numeric>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <hostdevcommon/tensor_accessor/arg_config.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;

// ============================================================================
// Compile-time varargs — positional CTA prefix + JIT smoke
// ============================================================================
//
// Host: KernelAdvancedOptions::compile_time_varargs (folded into compile_time_args_ as a prefix).
// Kernel: get_compile_time_vararg* — thin wrappers over kernel_compile_time_args[0..N).
// TensorBinding CTA payloads follow that prefix in KERNEL_COMPILE_TIME_ARGS.

TEST_F(ProgramSpecTestGen1, CPU_CompileTimeVarargsReadableFromKernel) {
    NodeCoord node{0, 0};
    const std::vector<uint32_t> cta_varargs = {0xCAFEBABEu, 0xDEADBEEFu, 0x11112222u};

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    static_assert(get_num_compile_time_varargs() == 3u);
    static_assert(get_compile_time_vararg<0>() == 0xCAFEBABEu);
    static_assert(get_compile_time_vararg<1>() == 0xDEADBEEFu);
    static_assert(get_compile_time_vararg<2>() == 0x11112222u);
    static_assert(get_compile_time_vararg(0) == 0xCAFEBABEu);
    static_assert(get_compile_time_vararg(1) == 0xDEADBEEFu);
    static_assert(get_compile_time_vararg(2) == 0x11112222u);
}
)"};
    dm_kernel.advanced_options.compile_time_varargs = cta_varargs;

    ProgramSpec spec;
    spec.name = "cta_varargs_readable";
    spec.kernels = {dm_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_EmptyCompileTimeVarargsReadableFromKernel) {
    NodeCoord node{0, 0};

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    static_assert(get_num_compile_time_varargs() == 0u);
}
)"};
    // Default / empty compile_time_varargs — accessors must still be emitted and compile.

    ProgramSpec spec;
    spec.name = "cta_varargs_empty";
    spec.kernels = {dm_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

// Template accessor bounds-checks at compile time: size==1 but get_compile_time_vararg<1>()
// must fail JIT (static_assert in the generated helper).
TEST_F(ProgramSpecTestGen1, CPU_OutOfRangeCompileTimeVarargTemplateFailsToCompile) {
    NodeCoord node{0, 0};

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    // Should not compile
    (void)get_compile_time_vararg<1>();
}
)"};
    dm_kernel.advanced_options.compile_time_varargs = {0xCAFEBABEu};

    ProgramSpec spec;
    spec.name = "cta_varargs_oob_template";
    spec.kernels = {dm_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    // MakeProgramFromSpec compiles; OOB template index must fail that build.
    EXPECT_ANY_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

// Stress: bake 1024 iota words and constexpr-walk them on the kernel side.
TEST_F(ProgramSpecTestGen1, CPU_CompileTimeVarargsIota1024ReadableFromKernel) {
    NodeCoord node{0, 0};
    constexpr uint32_t kNumVarargs = 1024u;
    std::vector<uint32_t> cta_varargs(kNumVarargs);
    std::iota(cta_varargs.begin(), cta_varargs.end(), 0u);

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
constexpr bool compile_time_varargs_are_iota() {
    if (get_num_compile_time_varargs() != 1024u) {
        return false;
    }
    for (uint32_t i = 0; i < get_num_compile_time_varargs(); ++i) {
        if (get_compile_time_vararg(i) != i) {
            return false;
        }
    }
    return true;
}

static_assert(compile_time_varargs_are_iota());

void kernel_main() {}
)"};
    dm_kernel.advanced_options.compile_time_varargs = cta_varargs;

    ProgramSpec spec;
    spec.name = "cta_varargs_iota_1024";
    spec.kernels = {dm_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

// Same stress as above, on a compute (TRISC) kernel — CTA varargs are available on DM and compute.
TEST_F(ProgramSpecTestGen1, CPU_CompileTimeVarargsIota1024ReadableFromComputeKernel) {
    NodeCoord node{0, 0};
    constexpr uint32_t kNumVarargs = 1024u;
    std::vector<uint32_t> cta_varargs(kNumVarargs);
    std::iota(cta_varargs.begin(), cta_varargs.end(), 0u);

    auto compute_kernel = MakeMinimalGen1ComputeKernel("compute_kernel");
    compute_kernel.source = KernelSpec::SourceCode{R"(
constexpr bool compile_time_varargs_are_iota() {
    if (get_num_compile_time_varargs() != 1024u) {
        return false;
    }
    for (uint32_t i = 0; i < get_num_compile_time_varargs(); ++i) {
        if (get_compile_time_vararg(i) != i) {
            return false;
        }
    }
    return true;
}

static_assert(compile_time_varargs_are_iota());

void kernel_main() {}
)"};
    compute_kernel.advanced_options.compile_time_varargs = cta_varargs;

    ProgramSpec spec;
    spec.name = "cta_varargs_iota_1024_compute";
    spec.kernels = {compute_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"compute_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

// CTA varargs are a positional prefix ahead of TensorBinding CTA payloads.
TEST_F(ProgramSpecTestGen1, CPU_CompileTimeVarargsPrefixBeforeTensorBindingCTAs) {
    NodeCoord node{0, 0};
    constexpr uint32_t kVararg0 = 0xCAFEBABEu;
    constexpr uint32_t kVararg1 = 0xDEADBEEFu;
    const std::vector<uint32_t> cta_varargs = {kVararg0, kVararg1};
    const uint32_t n = static_cast<uint32_t>(cta_varargs.size());

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    static_assert(get_num_compile_time_varargs() == 2u);
    static_assert(get_compile_time_vararg<0>() == 0xCAFEBABEu);
    static_assert(get_compile_time_vararg<1>() == 0xDEADBEEFu);
    // Binding CTA_OFFSET is shifted past the CTA-vararg prefix.
    static_assert(tensor::input_ta_t::args_t::is_dram);
    TensorAccessor accessor(tensor::input_ta);
    auto noc_addr = accessor.get_noc_addr(0);
    (void)noc_addr;
}
)"};
    dm_kernel.advanced_options.compile_time_varargs = cta_varargs;

    ProgramSpec spec;
    spec.name = "cta_varargs_with_tensor_binding";
    spec.kernels = {dm_kernel};
    spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor", tt::tt_metal::BufferType::DRAM)};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    const auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    const auto& handles = kernel->tensor_binding_handles();
    ASSERT_EQ(handles.size(), 1u);
    EXPECT_EQ(handles[0].cta_offset, n) << "TensorBinding CTA payload must start after the CTA-vararg prefix";
    EXPECT_EQ(kernel->get_compile_time_vararg_count(), n);

    const auto compile_args = kernel->compile_time_args();
    ASSERT_GE(compile_args.size(), n + 1u) << "positional CTAs should hold varargs then binding args_config";
    EXPECT_THAT(
        std::vector<uint32_t>(compile_args.begin(), compile_args.begin() + n),
        ::testing::ElementsAreArray(cta_varargs));
    const auto args_config =
        tensor_accessor::ArgsConfig(static_cast<tensor_accessor::ArgsConfig::Underlying>(compile_args[n]));
    EXPECT_TRUE(args_config.test(tensor_accessor::ArgConfig::IsDram))
        << "positional compile_args[N] must be the tensor args_config after the CTA-vararg prefix";

    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

// ============================================================================
// Compile-time varargs with kernel asserts enabled
// ============================================================================
//
// get_compile_time_vararg(idx) is constexpr AND bounds-checked, and those two properties collide
// with how ASSERT is defined. The lightweight-assert flavor expands to inline asm ("ebreak");
// inline assembly is unconditionally not allowed in a constexpr function in C++17.
// These tests ensure that this caveat is properly handled. The assertion-on environment is
// simulated in a lightweight (and hacky) way by defining the assertion-enable macro manually.

TEST_F(ProgramSpecTestGen1, CPU_InRangeCompileTimeVarargCompilesWithAssertsEnabled) {
    NodeCoord node{0, 0};

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.compiler_options.defines = {{"LIGHTWEIGHT_KERNEL_ASSERTS", "1"}, {"FORCE_WATCHER_OFF", "1"}};
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    // Constant-evaluated: proves the accessor is still usable in a constant expression while the
    // out-of-range path carries a (non-constexpr) assert.
    static_assert(get_compile_time_vararg(0) == 0xCAFEBABEu);
    static_assert(get_compile_time_vararg(2) == 0x11112222u);
    // Runtime-evaluated: the index is not a constant expression, so this is the path that actually
    // emits the bounds check and the ebreak.
    volatile uint32_t idx = 1;
    volatile uint32_t sink = get_compile_time_vararg(idx);
    (void)sink;
}
)"};
    dm_kernel.advanced_options.compile_time_varargs = {0xCAFEBABEu, 0xDEADBEEFu, 0x11112222u};

    ProgramSpec spec;
    spec.name = "cta_varargs_in_range_asserts_on";
    spec.kernels = {dm_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    IDevice* device = mesh_device_->get_devices()[0];
    EXPECT_NO_THROW(detail::CompileProgram(device, program));
}

// Out-of-range index in a constant expression, asserts on: must still fail the build. Constant
// evaluation reaches the non-constexpr out-of-range report, which is not a constant expression --
// so this is a clean build failure rather than a read past kernel_compile_time_args.
TEST_F(ProgramSpecTestGen1, CPU_OutOfRangeCompileTimeVarargFailsToCompileWithAssertsEnabled) {
    NodeCoord node{0, 0};

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.compiler_options.defines = {{"LIGHTWEIGHT_KERNEL_ASSERTS", "1"}, {"FORCE_WATCHER_OFF", "1"}};
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    // Only 1 vararg is baked, so index 1 is out of range. Should not compile.
    static_assert(get_compile_time_vararg(1) == 0u);
}
)"};
    dm_kernel.advanced_options.compile_time_varargs = {0xCAFEBABEu};

    ProgramSpec spec;
    spec.name = "cta_varargs_oob_asserts_on";
    spec.kernels = {dm_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    // MakeProgramFromSpec compiles; the OOB constant evaluation must fail that build.
    EXPECT_ANY_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
