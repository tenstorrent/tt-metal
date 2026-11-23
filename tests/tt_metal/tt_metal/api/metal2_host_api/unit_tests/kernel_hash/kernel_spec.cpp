// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KernelSpec fields that must change the kernel's JIT cache key (compute_hash).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/kernel_compile_utils.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/build.hpp"
#include "jit_build/jit_build_options.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen2ComputeKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

// ----------------------------------------------------------------------------
// Kernel hash sensitivity to scratchpad bindings
// ----------------------------------------------------------------------------
//
// The kernel's JIT cache key is its compute_hash(); a scratchpad binding flows into the device-side
// codegen (the scratch:: namespace + the CRTA-injected base address) and into the kernel's
// ScratchpadBindingHandles, so a kernel that binds a scratchpad must NOT hash equal to one that
// doesn't — otherwise it would silently reuse a stale cached binary.

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadBindingAffectsKernelHash) {
    // Same kernel source, differing only in whether the kernel binds a scratchpad. The bound variant
    // carries an extra ScratchpadBindingHandle, so the hashes must differ.
    auto make_bound_spec = [] {
        ProgramSpec spec;
        spec.name = "scratchpad_hash_bound";
        auto dm_kernel = MakeMinimalGen2DMKernel("dm_kernel");
        dm_kernel.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
            .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"});
        spec.kernels = {dm_kernel};
        spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 1024}};
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};
        return spec;
    };
    auto make_unbound_spec = [] {
        ProgramSpec spec;
        spec.name = "scratchpad_hash_unbound";
        auto dm_kernel = MakeMinimalGen2DMKernel("dm_kernel");
        spec.kernels = {dm_kernel};
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};
        return spec;
    };

    Program prog_bound(*mesh_device_, make_bound_spec());
    Program prog_unbound(*mesh_device_, make_unbound_spec());

    auto hash_bound = prog_bound.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    auto hash_unbound = prog_unbound.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    EXPECT_NE(hash_bound, hash_unbound)
        << "A kernel that binds a scratchpad must not share a JIT cache slot with one that doesn't.";
}

ProgramSpec MakeRuntimeArgNameSpec(std::vector<std::string> rta_names, std::vector<std::string> crta_names) {
    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.runtime_arg_schema.runtime_arg_names = std::move(rta_names);
    dm_kernel.runtime_arg_schema.common_runtime_arg_names = std::move(crta_names);
    ProgramSpec spec;
    spec.name = "runtime_arg_name_hash";
    spec.kernels = {dm_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};
    return spec;
}

TEST_F(ProgramSpecTestGen1, CPU_RuntimeArgNamePartitionAffectsKernelHash) {
    Program prog_left(*mesh_device_, MakeRuntimeArgNameSpec({"a", "bc"}, {}));
    Program prog_right(*mesh_device_, MakeRuntimeArgNameSpec({"ab", "c"}, {}));
    EXPECT_NE(
        prog_left.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash(),
        prog_right.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash());
}

TEST_F(ProgramSpecTestGen1, CPU_CommonRuntimeArgNamePartitionAffectsKernelHash) {
    Program prog_left(*mesh_device_, MakeRuntimeArgNameSpec({}, {"a", "bc"}));
    Program prog_right(*mesh_device_, MakeRuntimeArgNameSpec({}, {"ab", "c"}));
    EXPECT_NE(
        prog_left.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash(),
        prog_right.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash());
}

// ----------------------------------------------------------------------------
// Kernel hash sensitivity to array args
// ----------------------------------------------------------------------------
//
// An RTA/CRTA array's name and size are baked into the generated header (its accessor type and byte
// offset), and so are a CTA array's values. Its runtime values are not.

ProgramSpec MakeArrayArgSpec(const std::function<void(KernelSpec&)>& declare_args) {
    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    declare_args(dm_kernel);
    ProgramSpec spec;
    spec.name = "array_arg_hash";
    spec.kernels = {dm_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};
    return spec;
}

uint64_t DMKernelHash(const Program& program) {
    return program.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
}

TEST_F(ProgramSpecTestGen1, CPU_RuntimeArraySizeAffectsKernelHash) {
    Program prog_2(*mesh_device_, MakeArrayArgSpec([](KernelSpec& k) {
        k.runtime_arg_schema.runtime_array_args = {{.name = "x", .size = 2}};
    }));
    Program prog_3(*mesh_device_, MakeArrayArgSpec([](KernelSpec& k) {
        k.runtime_arg_schema.runtime_array_args = {{.name = "x", .size = 3}};
    }));
    EXPECT_NE(DMKernelHash(prog_2), DMKernelHash(prog_3));
}

TEST_F(ProgramSpecTestGen1, CPU_CommonRuntimeArraySizeAffectsKernelHash) {
    Program prog_2(*mesh_device_, MakeArrayArgSpec([](KernelSpec& k) {
        k.runtime_arg_schema.common_runtime_array_args = {{.name = "dims", .size = 2}};
    }));
    Program prog_3(*mesh_device_, MakeArrayArgSpec([](KernelSpec& k) {
        k.runtime_arg_schema.common_runtime_array_args = {{.name = "dims", .size = 3}};
    }));
    EXPECT_NE(DMKernelHash(prog_2), DMKernelHash(prog_3));
}

TEST_F(ProgramSpecTestGen1, CPU_RuntimeScalarVersusSizeOneArrayAffectsKernelHash) {
    // Same name, same word in the dispatch buffer, but a different accessor type in the generated header.
    Program prog_scalar(
        *mesh_device_, MakeArrayArgSpec([](KernelSpec& k) { k.runtime_arg_schema.runtime_arg_names = {"x"}; }));
    Program prog_array(*mesh_device_, MakeArrayArgSpec([](KernelSpec& k) {
        k.runtime_arg_schema.runtime_array_args = {{.name = "x", .size = 1}};
    }));
    EXPECT_NE(DMKernelHash(prog_scalar), DMKernelHash(prog_array));
}

TEST_F(ProgramSpecTestGen1, CPU_CompileTimeArrayValueAffectsKernelHash) {
    Program prog_a(
        *mesh_device_, MakeArrayArgSpec([](KernelSpec& k) { k.compile_time_array_args = {{"sizes", {1, 2}}}; }));
    Program prog_b(
        *mesh_device_, MakeArrayArgSpec([](KernelSpec& k) { k.compile_time_array_args = {{"sizes", {1, 3}}}; }));
    EXPECT_NE(DMKernelHash(prog_a), DMKernelHash(prog_b));
}

TEST_F(ProgramSpecTestGen1, CPU_CompileTimeArrayPartitionAffectsKernelHash) {
    // The same words split differently between two arrays.
    Program prog_left(*mesh_device_, MakeArrayArgSpec([](KernelSpec& k) {
        k.compile_time_array_args = {{"a", {1}}, {"b", {2, 3}}};
    }));
    Program prog_right(*mesh_device_, MakeArrayArgSpec([](KernelSpec& k) {
        k.compile_time_array_args = {{"a", {1, 2}}, {"b", {3}}};
    }));
    EXPECT_NE(DMKernelHash(prog_left), DMKernelHash(prog_right));
}

TEST_F(ProgramSpecTestGen1, CPU_RuntimeArrayValuesDoNotAffectKernelHash) {
    const NodeCoord node{0, 0};
    Program program(*mesh_device_, MakeArrayArgSpec([](KernelSpec& k) {
        k.runtime_arg_schema.runtime_array_args = {{.name = "x", .size = 2}};
        k.runtime_arg_schema.common_runtime_array_args = {{.name = "dims", .size = 2}};
    }));
    const uint64_t hash_before = DMKernelHash(program);

    ProgramRunArgs params;
    ProgramRunArgs::KernelRunArgs args{.kernel = KernelSpecName{"dm_kernel"}};
    args.runtime_array_arg_values["x"][node] = {7, 8};
    args.common_runtime_array_arg_values["dims"] = {32, 64};
    params.kernel_run_args.push_back(std::move(args));
    program.set_program_run_args(params);

    EXPECT_EQ(DMKernelHash(program), hash_before);
}

// ----------------------------------------------------------------------------
// Kernel hash sensitivity to the opt level
// ----------------------------------------------------------------------------
//
// compiler_options.opt_level sets the -O level of the kernel's compile and link, so the same source at another
// level is a different binary. The kernel builds into <kernel name>/<KernelCompileHash>/, which hashes
// compute_hash(). Without the opt level in it, changing only the level reuses the binary cached at the old one.

constexpr const char* kOptLevelKernelName = "kernel";

Program MakeOptLevelProgram(distributed::MeshDevice& mesh_device, KernelSpec kernel, KernelBuildOptLevel opt_level) {
    kernel.compiler_options.opt_level = opt_level;
    ProgramSpec spec;
    spec.name = "opt_level_hash";
    spec.kernels = {std::move(kernel)};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {kOptLevelKernelName})};
    return Program(mesh_device, spec);
}

void ExpectOptLevelAffectsKernelHash(distributed::MeshDevice& mesh_device, const KernelSpec& kernel) {
    Program prog_os = MakeOptLevelProgram(mesh_device, kernel, KernelBuildOptLevel::Os);
    Program prog_o2 = MakeOptLevelProgram(mesh_device, kernel, KernelBuildOptLevel::O2);
    Program prog_o2_again = MakeOptLevelProgram(mesh_device, kernel, KernelBuildOptLevel::O2);
    auto kernel_os = prog_os.impl().get_kernel_by_spec_name(kOptLevelKernelName);
    auto kernel_o2 = prog_o2.impl().get_kernel_by_spec_name(kOptLevelKernelName);
    auto kernel_o2_again = prog_o2_again.impl().get_kernel_by_spec_name(kOptLevelKernelName);

    EXPECT_NE(kernel_os->compute_hash(), kernel_o2->compute_hash())
        << "Kernels that differ only in opt level must not share a JIT cache slot.";
    EXPECT_EQ(kernel_o2->compute_hash(), kernel_o2_again->compute_hash());

    // The build directory name. Same build key and build options, so only compute_hash() can tell them apart.
    JitBuildEnv build_env;
    JitBuildOptions build_options(build_env);
    constexpr uint64_t build_key = 0;
    EXPECT_NE(
        tt::tt_metal::detail::KernelCompileHash(kernel_os, build_options, build_key),
        tt::tt_metal::detail::KernelCompileHash(kernel_o2, build_options, build_key));
}

TEST_F(ProgramSpecTestGen1, CPU_OptLevelAffectsDataMovementKernelHash) {
    ExpectOptLevelAffectsKernelHash(*mesh_device_, MakeMinimalGen1DMKernel(kOptLevelKernelName));
}

TEST_F(ProgramSpecTestGen1, CPU_OptLevelAffectsComputeKernelHash) {
    ExpectOptLevelAffectsKernelHash(*mesh_device_, MakeMinimalGen1ComputeKernel(kOptLevelKernelName));
}

TEST_F(ProgramSpecTestQuasar, CPU_OptLevelAffectsDataMovementKernelHash) {
    ExpectOptLevelAffectsKernelHash(*mesh_device_, MakeMinimalGen2DMKernel(kOptLevelKernelName));
}

TEST_F(ProgramSpecTestQuasar, CPU_OptLevelAffectsComputeKernelHash) {
    ExpectOptLevelAffectsKernelHash(*mesh_device_, MakeMinimalGen2ComputeKernel(kOptLevelKernelName));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
