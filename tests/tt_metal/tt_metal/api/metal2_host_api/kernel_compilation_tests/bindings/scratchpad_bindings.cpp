// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile smoke tests for scratchpad bindings (mock Wormhole).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;

// ----------------------------------------------------------------------------
// Scratchpad JIT-compile smoke tests (device-side accessor composes & compiles)
// ----------------------------------------------------------------------------
//
// Like the TensorAccessor smoke above, these JIT-compile a kernel that constructs a Scratchpad from
// its binding token (scratch::<name>) and reads the CRTA-injected base address — exercising the
// generated scratch:: namespace + ScratchpadBindingToken object and the device-side Scratchpad ctor. Compile-only on
// the mock Wormhole device (Gen1: the Quasar TRISC firmware isn't built in this checkout, so a
// Quasar JIT-compile would fail at link).

TEST_F(ProgramSpecTestGen1, CPU_ScratchpadAccessorBindingJITSmokeDMKernel) {
    // DM kernel constructs a Scratchpad from its binding token and reads the CRTA-injected base address.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "scratch_smoke_dm";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    Scratchpad<int32_t> pad(scratch::scratch);
    volatile uint32_t base = pad.get_base_address();
    (void)base;
}

static_assert(scratch::get_token_if_present<"scratch">() == &scratch::scratch);
static_assert(scratch::get_token_if_present<"not_a_scratch">() == nullptr);
)"};
    dm_kernel.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"});

    spec.kernels = {dm_kernel};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 1024}};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

// Compute-kernel counterpart. A scratchpad binding works on a compute kernel: scratchpad.h only
// forward-declares get_common_arg_val (resolved by the kernel's own API header — api/compute/common.h
// on TRISC, api/dataflow/dataflow_api.h on DM) and is otherwise NOC-free, so the device-side
// Scratchpad<T> ctor composes on the compute (TRISC) build path as well as DM.
TEST_F(ProgramSpecTestGen1, CPU_ScratchpadAccessorBindingJITSmokeComputeKernel) {
    // MakeMinimalGen1ValidProgramSpec wires a DM producer (kernels[0]) into a compute consumer
    // (kernels[1]) through a DFB; bind the scratchpad to the compute kernel and have it construct a
    // Scratchpad from its binding token.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.kernels[1].source = KernelSpec::SourceCode{R"(
void kernel_main() {
    Scratchpad<int32_t> pad(scratch::scratch);
    volatile uint32_t base = pad.get_base_address();
    (void)base;
}

static_assert(scratch::get_token_if_present<"scratch">() == &scratch::scratch);
static_assert(scratch::get_token_if_present<"not_a_scratch">() == nullptr);
)"};

    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 1024}};
    spec.kernels[1].scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"});

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

// Compile-only: a range-based for loop over a Scratchpad must compile. Exercises begin()/end() and the
// CoreLocalMem<T> iterator ops the loop desugars to (operator++, operator!=, operator*). Compile-only on
// the mock Gen1 device — no run: reading the uninitialized region would be UB at runtime, but this test
// only JIT-compiles the kernel, so the loop just needs to be well-formed.
TEST_F(ProgramSpecTestGen1, CPU_ScratchpadRangeBasedForCompiles) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "scratch_range_for";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    Scratchpad<int32_t> pad(scratch::scratch);
    int32_t acc = 0;
    for (auto& elem : pad) {
        acc += elem;
    }
    volatile int32_t sink = acc;  // keep the loop live so the range-for is actually instantiated
    (void)sink;
}

static_assert(scratch::get_token_if_present<"scratch">() == &scratch::scratch);
static_assert(scratch::get_token_if_present<"not_a_scratch">() == nullptr);
)"};
    dm_kernel.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"});

    spec.kernels = {dm_kernel};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 1024}};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
