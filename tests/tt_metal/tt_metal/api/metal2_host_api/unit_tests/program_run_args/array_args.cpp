// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Array args through SetProgramRunArgs, UpdateProgramRunArgs and MergeProgramRunArgs: which values are
// accepted, and where they land in the kernel's arg buffers.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramRunArgsTestGen1;

const NodeCoord kNodeA{0, 0};
const NodeCoord kNodeB{1, 0};

// One DM kernel on kNodeA and kNodeB, declaring:
//   RTAs:  scalar "a", then arrays "x" (3 elements) and "y" (2 elements)
//   CRTAs: scalar "c", then array "d" (2 elements), then 2 common varargs
// so the per-node RTA buffer is [a, x0, x1, x2, y0, y1] and the CRTA buffer is [c, d0, d1, cv0, cv1].
ProgramSpec MakeArrayRunArgsSpec() {
    auto kernel = MakeMinimalGen1DMKernel("dm_kernel");
    kernel.runtime_arg_schema = {
        .runtime_arg_names = {"a"},
        .common_runtime_arg_names = {"c"},
        .runtime_array_args = {{.name = "x", .size = 3}, {.name = "y", .size = 2}},
        .common_runtime_array_args = {{.name = "d", .size = 2}},
    };
    kernel.advanced_options.num_common_runtime_varargs = 2;

    ProgramSpec spec;
    spec.name = "array_run_args";
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit", NodeRangeSet{NodeRange{kNodeA, kNodeB}}, {"dm_kernel"})};
    return spec;
}

// Complete run args for MakeArrayRunArgsSpec. Every value is `base` plus a distinct offset.
ProgramRunArgs::KernelRunArgs MakeArrayKernelRunArgs(uint32_t base) {
    ProgramRunArgs::KernelRunArgs args{.kernel = KernelSpecName{"dm_kernel"}};
    args.runtime_arg_values["a"][kNodeA] = base + 1;
    args.runtime_arg_values["a"][kNodeB] = base + 2;
    args.runtime_array_arg_values["x"][kNodeA] = {base + 10, base + 11, base + 12};
    args.runtime_array_arg_values["x"][kNodeB] = {base + 20, base + 21, base + 22};
    args.runtime_array_arg_values["y"][kNodeA] = {base + 30, base + 31};
    args.runtime_array_arg_values["y"][kNodeB] = {base + 40, base + 41};
    args.common_runtime_arg_values["c"] = base + 50;
    args.common_runtime_array_arg_values["d"] = {base + 60, base + 61};
    args.advanced_options.common_runtime_varargs = {base + 70, base + 71};
    return args;
}

ProgramRunArgs WrapKernelRunArgs(ProgramRunArgs::KernelRunArgs args) {
    ProgramRunArgs params;
    params.kernel_run_args.push_back(std::move(args));
    return params;
}

std::vector<uint32_t> RtaWords(const Program& program, const NodeCoord& node) {
    const auto& rta = program.impl().get_kernel_by_spec_name("dm_kernel")->runtime_args_data(node);
    return {rta.data(), rta.data() + rta.size()};
}

std::vector<uint32_t> CrtaWords(const Program& program) {
    const auto& crta = program.impl().get_kernel_by_spec_name("dm_kernel")->common_runtime_args_data();
    return {crta.data(), crta.data() + crta.size()};
}

void ExpectArgsWithBase(const Program& program, uint32_t base) {
    using ::testing::ElementsAre;
    EXPECT_THAT(
        RtaWords(program, kNodeA), ElementsAre(base + 1, base + 10, base + 11, base + 12, base + 30, base + 31));
    EXPECT_THAT(
        RtaWords(program, kNodeB), ElementsAre(base + 2, base + 20, base + 21, base + 22, base + 40, base + 41));
    EXPECT_THAT(CrtaWords(program), ElementsAre(base + 50, base + 60, base + 61, base + 70, base + 71));
}

// ============================================================================
// SetProgramRunArgs: layout
// ============================================================================

TEST_F(ProgramRunArgsTestGen1, CPU_ArraysFollowScalarsInNamedSection) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeArrayRunArgsSpec());

    SetProgramRunArgs(program, WrapKernelRunArgs(MakeArrayKernelRunArgs(0)));  // first call: allocates
    ExpectArgsWithBase(program, 0);

    SetProgramRunArgs(program, WrapKernelRunArgs(MakeArrayKernelRunArgs(1000)));  // fast path: in place
    ExpectArgsWithBase(program, 1000);
}

TEST_F(ProgramRunArgsTestGen1, CPU_ArrayOnlyKernelSerializesArrays) {
    // No scalar RTAs: the arrays alone make up the named section.
    auto kernel = MakeMinimalGen1DMKernel("dm_kernel");
    kernel.runtime_arg_schema.runtime_array_args = {{.name = "x", .size = 2}};
    ProgramSpec spec;
    spec.name = "array_only";
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", kNodeA, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    ProgramRunArgs::KernelRunArgs args{.kernel = KernelSpecName{"dm_kernel"}};
    args.runtime_array_arg_values["x"][kNodeA] = {5, 6};
    SetProgramRunArgs(program, WrapKernelRunArgs(args));
    EXPECT_THAT(RtaWords(program, kNodeA), ::testing::ElementsAre(5u, 6u));
}

TEST_F(ProgramRunArgsTestGen1, CPU_ArrayValuesSerializeAsUint32Conversion) {
    auto kernel = MakeMinimalGen1DMKernel("dm_kernel");
    kernel.runtime_arg_schema.common_runtime_array_args = {{.name = "d", .size = 3}};
    ProgramSpec spec;
    spec.name = "array_conversion";
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", kNodeA, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    const std::size_t wide = 12;
    ProgramRunArgs::KernelRunArgs args{.kernel = KernelSpecName{"dm_kernel"}};
    args.common_runtime_array_arg_values["d"] = {-1, wide, true};
    SetProgramRunArgs(program, WrapKernelRunArgs(args));
    EXPECT_THAT(CrtaWords(program), ::testing::ElementsAre(0xFFFFFFFFu, 12u, 1u));
}

// ============================================================================
// SetProgramRunArgs: validation
// ============================================================================

TEST_F(ProgramRunArgsTestGen1, CPU_WrongLengthRtaArrayFails) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeArrayRunArgsSpec());
    auto args = MakeArrayKernelRunArgs(0);
    args.runtime_array_arg_values["x"][kNodeB] = {1, 2};

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, WrapKernelRunArgs(args)); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("RTA array 'x' expects 3 elements, but 2 were provided")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_WrongLengthCrtaArrayFails) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeArrayRunArgsSpec());
    auto args = MakeArrayKernelRunArgs(0);
    args.common_runtime_array_arg_values["d"] = {1, 2, 3};

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, WrapKernelRunArgs(args)); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("CRTA array 'd' expects 2 elements, but 3 were provided")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RtaArrayMissingOnNodeFails) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeArrayRunArgsSpec());
    auto args = MakeArrayKernelRunArgs(0);
    args.runtime_array_arg_values["y"].erase(kNodeB);

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, WrapKernelRunArgs(args)); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("node 1-0 expects 2 RTA arrays, but 1 were provided")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_CrtaArrayMissingFails) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeArrayRunArgsSpec());
    auto args = MakeArrayKernelRunArgs(0);
    args.common_runtime_array_arg_values.erase("d");

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, WrapKernelRunArgs(args)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("is missing CRTA array 'd'")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_UndeclaredArraysFail) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeArrayRunArgsSpec());

    auto rta_args = MakeArrayKernelRunArgs(0);
    rta_args.runtime_array_arg_values["z"][kNodeA] = {1};
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, WrapKernelRunArgs(rta_args)); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("sets RTA array 'z' which is not declared in the schema")));

    auto crta_args = MakeArrayKernelRunArgs(0);
    crta_args.common_runtime_array_arg_values["z"] = {1};
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, WrapKernelRunArgs(crta_args)); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("sets CRTA array 'z' which is not declared in the schema")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RtaArrayOnForeignNodeFails) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeArrayRunArgsSpec());
    auto args = MakeArrayKernelRunArgs(0);
    args.runtime_array_arg_values["x"][NodeCoord{5, 5}] = {1, 2, 3};

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, WrapKernelRunArgs(args)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "is setting runtime_array_arg_values for node 5-5, but the kernel does not run on that node")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_ArrayOnlyKernelMayNotOmitKernelRunArgs) {
    // A kernel whose only runtime args are arrays still has args to supply.
    auto kernel = MakeMinimalGen1DMKernel("dm_kernel");
    kernel.runtime_arg_schema.common_runtime_array_args = {{.name = "d", .size = 2}};
    ProgramSpec spec;
    spec.name = "array_only_omitted";
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", kNodeA, {"dm_kernel"})};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, ProgramRunArgs{}); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("non-empty RTA/CRTA schema but has no runtime parameters")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_WrongLengthArrayFailsEvenWithoutValidation) {
    // The length check is kept under skip_validation: a wrong-length value would otherwise overrun into
    // the neighbouring args. Checked on the first-call path, the in-place fast path, and the CRTA buffer.
    Program program = MakeProgramFromSpec(*mesh_device_, MakeArrayRunArgsSpec());
    auto long_rta = MakeArrayKernelRunArgs(0);
    long_rta.runtime_array_arg_values["x"][kNodeA] = {1, 2, 3, 4};
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, WrapKernelRunArgs(long_rta), /*skip_validation=*/true); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'dm_kernel' array 'x' expects 3 elements, but 4 were provided")));

    SetProgramRunArgs(program, WrapKernelRunArgs(MakeArrayKernelRunArgs(0)));
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, WrapKernelRunArgs(long_rta), /*skip_validation=*/true); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'dm_kernel' array 'x' expects 3 elements, but 4 were provided")));

    auto long_crta = MakeArrayKernelRunArgs(0);
    long_crta.common_runtime_array_arg_values["d"] = {1, 2, 3};
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, WrapKernelRunArgs(long_crta), /*skip_validation=*/true); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'dm_kernel' array 'd' expects 2 elements, but 3 were provided")));
}

// ============================================================================
// UpdateProgramRunArgs
// ============================================================================

TEST_F(ProgramRunArgsTestGen1, CPU_UpdateReplacesWholeArraysAndKeepsTheRest) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeArrayRunArgsSpec());
    SetProgramRunArgs(program, WrapKernelRunArgs(MakeArrayKernelRunArgs(0)));

    // Replace "y" on kNodeB only, and "d".
    ProgramRunArgs::KernelRunArgs update{.kernel = KernelSpecName{"dm_kernel"}};
    update.runtime_array_arg_values["y"][kNodeB] = {900, 901};
    update.common_runtime_array_arg_values["d"] = {960, 961};
    UpdateProgramRunArgs(program, WrapKernelRunArgs(update));

    using ::testing::ElementsAre;
    EXPECT_THAT(RtaWords(program, kNodeA), ElementsAre(1u, 10u, 11u, 12u, 30u, 31u));
    EXPECT_THAT(RtaWords(program, kNodeB), ElementsAre(2u, 20u, 21u, 22u, 900u, 901u));
    EXPECT_THAT(CrtaWords(program), ElementsAre(50u, 960u, 961u, 70u, 71u));
}

TEST_F(ProgramRunArgsTestGen1, CPU_UpdateWithWrongLengthArrayFails) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeArrayRunArgsSpec());
    SetProgramRunArgs(program, WrapKernelRunArgs(MakeArrayKernelRunArgs(0)));

    ProgramRunArgs::KernelRunArgs rta_update{.kernel = KernelSpecName{"dm_kernel"}};
    rta_update.runtime_array_arg_values["x"][kNodeA] = {1};
    EXPECT_THAT(
        [&] { UpdateProgramRunArgs(program, WrapKernelRunArgs(rta_update)); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("RTA array 'x' expects 3 elements, but 1 were provided")));
    EXPECT_THAT(
        [&] { UpdateProgramRunArgs(program, WrapKernelRunArgs(rta_update), /*skip_validation=*/true); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'dm_kernel' array 'x' expects 3 elements, but 1 were provided")));

    ProgramRunArgs::KernelRunArgs crta_update{.kernel = KernelSpecName{"dm_kernel"}};
    crta_update.common_runtime_array_arg_values["d"] = {1};
    EXPECT_THAT(
        [&] { UpdateProgramRunArgs(program, WrapKernelRunArgs(crta_update)); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("CRTA array 'd' expects 2 elements, but 1 were provided")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_UpdateWithUndeclaredArrayFails) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeArrayRunArgsSpec());
    SetProgramRunArgs(program, WrapKernelRunArgs(MakeArrayKernelRunArgs(0)));

    ProgramRunArgs::KernelRunArgs update{.kernel = KernelSpecName{"dm_kernel"}};
    update.runtime_array_arg_values["z"][kNodeA] = {1};
    EXPECT_THAT(
        [&] { UpdateProgramRunArgs(program, WrapKernelRunArgs(update)); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("sets RTA array 'z' which is not declared in the schema")));
}

// ============================================================================
// MergeProgramRunArgs
// ============================================================================

TEST_F(ProgramRunArgsTestGen1, CPU_MergeUnionsDisjointArrays) {
    // One input carries the scalars and varargs, the other the arrays.
    auto full = MakeArrayKernelRunArgs(0);
    ProgramRunArgs::KernelRunArgs scalars{
        .kernel = full.kernel,
        .runtime_arg_values = full.runtime_arg_values,
        .common_runtime_arg_values = full.common_runtime_arg_values,
        .advanced_options = full.advanced_options,
    };
    ProgramRunArgs::KernelRunArgs arrays{
        .kernel = full.kernel,
        .runtime_array_arg_values = full.runtime_array_arg_values,
        .common_runtime_array_arg_values = full.common_runtime_array_arg_values,
    };
    const ProgramRunArgs rest[] = {WrapKernelRunArgs(arrays)};
    ProgramRunArgs merged = MergeProgramRunArgs(WrapKernelRunArgs(scalars), rest);

    Program program = MakeProgramFromSpec(*mesh_device_, MakeArrayRunArgsSpec());
    SetProgramRunArgs(program, merged);
    ExpectArgsWithBase(program, 0);
}

TEST_F(ProgramRunArgsTestGen1, CPU_MergeWithRepeatedArrayFails) {
    ProgramRunArgs::KernelRunArgs first{.kernel = KernelSpecName{"dm_kernel"}};
    first.runtime_array_arg_values["x"][kNodeA] = {1, 2, 3};
    first.common_runtime_array_arg_values["d"] = {1, 2};

    ProgramRunArgs::KernelRunArgs repeats_rta{.kernel = KernelSpecName{"dm_kernel"}};
    repeats_rta.runtime_array_arg_values["x"][kNodeA] = {4, 5, 6};
    const ProgramRunArgs rest_rta[] = {WrapKernelRunArgs(repeats_rta)};
    EXPECT_THAT(
        [&] { MergeProgramRunArgs(WrapKernelRunArgs(first), rest_rta); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("runtime array arg 'x' is specified in more than one ProgramRunArgs")));

    ProgramRunArgs::KernelRunArgs repeats_crta{.kernel = KernelSpecName{"dm_kernel"}};
    repeats_crta.common_runtime_array_arg_values["d"] = {3, 4};
    const ProgramRunArgs rest_crta[] = {WrapKernelRunArgs(repeats_crta)};
    EXPECT_THAT(
        [&] { MergeProgramRunArgs(WrapKernelRunArgs(first), rest_crta); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("common runtime array arg 'd' is specified in more than one ProgramRunArgs")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
