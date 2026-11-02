// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile tests for array args (mock Wormhole): the generated accessors compile and read alike on the
// DM and compute paths, CTA array elements are constant expressions, and TT_KERNEL rejects arrays.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;

// Reads every kind of array arg, next to a scalar and a vararg of each runtime kind, so the arrays'
// offsets are exercised between them.
constexpr const char* kArrayArgsKernelSource = R"(
// CTA arrays: elements, size and the view itself are constant expressions.
static_assert(get_arg(args::page_sizes).size() == 3u);
static_assert(get_arg(args::page_sizes)[0] == 64u);
static_assert(get_arg(args::page_sizes)[2] == 256u);
template <uint32_t N>
struct AsTemplateArg {
    static constexpr uint32_t value = N;
};
static_assert(AsTemplateArg<get_arg(args::page_sizes)[1]>::value == 128u);
static_assert(decltype(get_arg(args::sem_addrs))::size() == 4u);
static_assert(decltype(get_arg(args::dims))::size() == 2u);

void kernel_main() {
    volatile uint32_t sink = get_arg(args::count) + get_arg(args::shape_scale);

    auto addrs = get_arg(args::sem_addrs);
    for (uint32_t i = 0; i < addrs.size(); ++i) {
        sink = sink + addrs[i];
    }
    for (uint32_t d : get_arg(args::dims)) {
        sink = sink + d;
    }
    for (uint32_t p : get_arg(args::page_sizes)) {
        sink = sink + p;
    }
    auto pages = get_arg(args::page_sizes);  // copies the view, not the array
    sink = sink + pages[sink % pages.size()];
    sink = sink + get_vararg(0) + get_common_vararg(0);
}
)";

void DeclareArrayArgs(KernelSpec& kernel) {
    kernel.compile_time_array_args = {{"page_sizes", {64, 128, 256}}};
    kernel.runtime_arg_schema = {
        .runtime_arg_names = {"count"},
        .common_runtime_arg_names = {"shape_scale"},
        .runtime_array_args = {{.name = "sem_addrs", .size = 4}},
        .common_runtime_array_args = {{.name = "dims", .size = 2}},
    };
    kernel.advanced_options.num_runtime_varargs = 1;
    kernel.advanced_options.num_common_runtime_varargs = 1;
}

ProgramSpec MakeSingleKernelSpec(const KernelSpec& kernel) {
    ProgramSpec spec;
    spec.name = "array_args_compile";
    spec.kernels = {kernel};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {kernel.unique_id.get()})};
    return spec;
}

TEST_F(ProgramSpecTestGen1, CPU_ArrayArgsCompileInDataMovementKernel) {
    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{kArrayArgsKernelSource};
    DeclareArrayArgs(dm_kernel);

    Program program = MakeProgramFromSpec(*mesh_device_, MakeSingleKernelSpec(dm_kernel));
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_ArrayArgsCompileInComputeKernel) {
    auto compute_kernel = MakeMinimalGen1ComputeKernel("compute_kernel");
    compute_kernel.source = KernelSpec::SourceCode{kArrayArgsKernelSource};
    DeclareArrayArgs(compute_kernel);

    Program program = MakeProgramFromSpec(*mesh_device_, MakeSingleKernelSpec(compute_kernel));
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_OutOfRangeConstantCtaArrayIndexFailsToCompile) {
    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    // Should not compile: the bounds check is not a constant expression.
    constexpr uint32_t past_end = get_arg(args::page_sizes)[3];
    (void)past_end;
}
)"};
    dm_kernel.compile_time_array_args = {{"page_sizes", {64, 128, 256}}};

    // MakeProgramFromSpec compiles; the out-of-range index must fail that build.
    EXPECT_ANY_THROW(MakeProgramFromSpec(*mesh_device_, MakeSingleKernelSpec(dm_kernel)));
}

TEST_F(ProgramSpecTestGen1, CPU_TtKernelWithArrayArgsFails) {
    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
#include "experimental/kernel_args.h"
TT_KERNEL void entry(uint32_t count) { (void)count; }
)"};
    dm_kernel.runtime_arg_schema = {
        .runtime_arg_names = {"count"},
        .runtime_array_args = {{.name = "sem_addrs", .size = 4}},
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, MakeSingleKernelSpec(dm_kernel)); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("belongs to a kernel that declares array arguments")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
