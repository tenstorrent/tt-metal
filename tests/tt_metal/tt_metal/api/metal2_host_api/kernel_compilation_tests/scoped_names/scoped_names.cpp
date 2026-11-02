// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile tests for scoped names (tech_reports/NamedKernelArgs/scoped_names.md): every kind of
// kernel-visible name, scoped, lands in kernel_scopes_generated.h as a static member of its scope's
// struct, and device code can be templated on a scope (mock Wormhole).

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

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;

// The DM kernel has scopes a and b, and some unscoped names. It reads both scopes through one
// templated piece of device code, the way a fused kernel uses one micro-op twice.
constexpr const char* kScopedDmSource = R"(
#include <type_traits>
#include "api/semaphore.h"

// A scope this KernelSpec has no names in: kernel source shared across KernelSpecs can still name it.
struct not_in_this_kernel;

template <typename Scope>
struct Op {
    static constexpr uint32_t height = get_arg(Scope::Ht);
    static uint32_t tiles() { return get_arg(Scope::num_tiles); }
    // By-name lookup through a dependent scope needs the `template` disambiguator.
    static constexpr bool has_pad = Scope::template get_scratch_token_if_present<"pad">() != nullptr;
};

void kernel_main() {
    // Scoping a name doesn't change the dispatch buffer layout: RTA offsets follow declaration
    // order across unscoped and scoped names alike. (CRTAs count their offsets separately.)
    static_assert(args::num_tiles.byte_offset == 0u);
    static_assert(a::num_tiles.byte_offset == 4u);
    static_assert(b::num_tiles.byte_offset == 8u);
    static_assert(a::batch.byte_offset == 0u);

    static_assert(get_arg(args::Wt) == 4u);
    static_assert(Op<a>::height == 8u && Op<b>::height == 16u);
    static_assert(!Op<a>::has_pad && Op<b>::has_pad);
    volatile uint32_t sink = get_arg(args::num_tiles) + Op<a>::tiles() + Op<b>::tiles() + get_arg(a::batch);
    (void)sink;

    DataflowBuffer out(a::out);
    Semaphore ready(a::ready);
    TensorAccessor accessor(a::t);
    (void)accessor.get_noc_addr(0);
    static_assert(std::tuple_size_v<std::remove_cv_t<decltype(a::inputs)>> == 1);
    Scratchpad<int32_t> pad(b::pad);
    (void)out;
    (void)ready;
    (void)pad;
}

static_assert(a::get_dfb_token_if_present<"out">() == &a::out);
static_assert(a::get_dfb_token_if_present<"missing">() == nullptr);
static_assert(a::get_tensor_token_if_present<"t">() == &a::t);
static_assert(b::get_scratch_token_if_present<"pad">() == &b::pad);
static_assert(b::get_dfb_token_if_present<"out">() == nullptr);
)";

constexpr const char* kScopedComputeSource = R"(
void kernel_main() {
    static_assert(get_arg(b::Ht) == 16u);
    DataflowBuffer in(b::in);
    volatile uint32_t sink = get_arg(b::count);
    (void)sink;
    (void)in;
}
)";

TEST_F(ProgramSpecTestGen1, CPU_ScopedNamesOfEveryKindJITSmoke) {
    NodeCoord node{0, 0};

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{kScopedDmSource};
    dm_kernel.runtime_arg_schema.runtime_arg_names = {"num_tiles", "a.num_tiles", "b.num_tiles"};
    dm_kernel.runtime_arg_schema.common_runtime_arg_names = {"a.batch"};
    dm_kernel.compile_time_args = {{"Wt", 4}, {"a.Ht", 8}, {"b.Ht", 16}};
    dm_kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "a.out"));
    dm_kernel.semaphore_bindings.push_back(
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"ready"}, .accessor_name = "a.ready"});
    dm_kernel.scratchpad_bindings.push_back(
        ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "b.pad"});
    dm_kernel.advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "a.inputs", .members = {"a.t"}},
    };

    auto compute_kernel = MakeMinimalGen1ComputeKernel("compute_kernel");
    compute_kernel.source = KernelSpec::SourceCode{kScopedComputeSource};
    compute_kernel.runtime_arg_schema.runtime_arg_names = {"b.count"};
    compute_kernel.compile_time_args = {{"b.Ht", 16}};
    compute_kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_0"}, "b.in"));

    auto dfb = MakeMinimalDFB("dfb_0");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;  // required for a compute DFB endpoint

    ProgramSpec spec;
    spec.name = "scoped_names_every_kind";
    spec.kernels = {dm_kernel, compute_kernel};
    spec.dataflow_buffers = {dfb};
    spec.semaphores = {SemaphoreSpec{.unique_id = SemaphoreSpecName{"ready"}, .target_nodes = node}};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 1024}};
    spec.tensor_parameters = {MakeMinimalTensorParameter("t_0")};
    BindTensorParameterToKernel(spec.kernels[0], "t_0", "a.t");
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel", "compute_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_ScopedNameMissingFromKernelFailsToCompile) {
    // A scoped name reaches the kernel only through its scope; the unscoped spelling doesn't exist.
    NodeCoord node{0, 0};

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    static_assert(get_arg(args::Ht) == 8u);  // Should not compile: Ht is a::Ht
}
)"};
    dm_kernel.compile_time_args = {{"a.Ht", 8}};

    ProgramSpec spec;
    spec.name = "scoped_name_unscoped_spelling";
    spec.kernels = {dm_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    // MakeProgramFromSpec compiles; the bad spelling must fail that build.
    EXPECT_ANY_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_TtKernelEntryRejectsScopedArg) {
    // A TT_KERNEL entry takes its arguments as parameters, so they can't be scoped.
    NodeCoord node{0, 0};

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
#include "experimental/kernel_args.h"
TT_KERNEL void entry(uint32_t num_tiles) { (void)num_tiles; }
)"};
    dm_kernel.runtime_arg_schema.runtime_arg_names = {"op.num_tiles"};

    ProgramSpec spec;
    spec.name = "tt_kernel_scoped_arg";
    spec.kernels = {dm_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("cannot take the scoped argument 'op.num_tiles'")));
}

TEST_F(ProgramSpecTestGen1, CPU_ScopesShareABindingThroughAKernelSideAliasJITSmoke) {
    // The documented way to share a binding between scopes (tech_reports/NamedKernelArgs/scoped_names.md):
    // bind it once, in one scope, and alias it in kernel code, in a view of the other scope.
    NodeCoord node{0, 0};

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
template <typename Scope>
uint64_t op() {
    TensorAccessor accessor(Scope::input);
    return accessor.get_noc_addr(get_arg(Scope::page));
}

// Scope b's names, plus a's tensor binding under b's own name.
struct b_view : b {
    static constexpr const auto& input = a::input;
};
static_assert(&b_view::input == &a::input);  // the same token, not a copy

void kernel_main() {
    volatile uint64_t sink = op<a>() + op<b_view>();
    (void)sink;
}
)"};
    dm_kernel.runtime_arg_schema.runtime_arg_names = {"a.page", "b.page"};

    ProgramSpec spec;
    spec.name = "scopes_share_a_binding";
    spec.kernels = {dm_kernel};
    spec.tensor_parameters = {MakeMinimalTensorParameter("t_0")};
    BindTensorParameterToKernel(spec.kernels[0], "t_0", "a.input");
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
