// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Layout of a lowered kernel's CRTA buffer: named CRTAs, varargs and tensor binding runtime fields.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <hostdevcommon/tensor_accessor/arg_config.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeShardedTensorParameter;
using test_helpers::ProgramSpecTestGen1;

// ============================================================================
// Dynamic tensor shape: tensor binding CRTA slots
// ============================================================================
//
// For sharded TensorParameters, dynamic_tensor_shape moves the tensor_shape_in_pages words out of the
// kernel's CTAs and into per-binding CRTA slots, so num_runtime_field_crta_words tracks the rank when
// needed. Kernel-hash stability across shapes is in kernel_hash/tensor_spec_relaxations.cpp; the
// loosened runtime spec match is in program_run_args/tensor_args.cpp.

TEST_F(ProgramSpecTestGen1, CPU_DynamicTensorShape_ShardedBindingTracksShapeCRTASlots) {
    // Sharded + dynamic_tensor_shape: the TensorBindingHandle's num_runtime_field_crta_words
    // should equal the BufferDistributionSpec's tensor_shape_in_pages rank (one CRTA word per
    // shape dim, written at enqueue).
    //
    // Note: BDS flattens the logical_shape via its sharding scheme, so the BDS rank is not
    // generally the same as logical_shape.rank(). We derive the expected value from the BDS
    // directly to be robust against BDS-internal flattening conventions.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto tp = MakeShardedTensorParameter("input_tensor", tt::tt_metal::Shape{1, 1, 64, 32}, {32, 32}, 2);
    tp.relaxations = TensorSpecRelaxations{.dynamic_tensor_shape = true};
    spec.tensor_parameters = {tp};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    const auto& handles = kernel->tensor_binding_handles();
    ASSERT_EQ(handles.size(), 1u);

    const auto bds = tp.spec.compute_buffer_sharding_args().buffer_distribution_spec();
    ASSERT_TRUE(bds.has_value());
    const auto expected_rank = bds->tensor_shape_in_pages().rank();
    EXPECT_GT(expected_rank, 0u);
    EXPECT_EQ(handles[0].num_runtime_field_crta_words, static_cast<uint32_t>(expected_rank))
        << "Sharded + dynamic_tensor_shape: runtime-field CRTA words should equal BDS shape rank.";
}

TEST_F(ProgramSpecTestGen1, CPU_DynamicTensorShape_InterleavedRowMajorBindingTracksPageSizeCRTASlot) {
    // Row-major interleaved + dynamic_tensor_shape: the page size (= last_dim_width * elem_size) is
    // part of the varying shape, so the resolver folds it from a compile-time arg into a single
    // per-binding CRTA word ("A-collapse": the page-size CTA slot is dropped and the RuntimePageSize
    // bit is set in args_config). The binding handle must advertise exactly one runtime field word,
    // tagged as the page-size kind. MakeMinimalTensorParameter is BFLOAT16 / ROW_MAJOR / interleaved.
    auto make_spec = [](bool dynamic) {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        auto tp = MakeMinimalTensorParameter("input_tensor");
        tp.relaxations = TensorSpecRelaxations{.dynamic_tensor_shape = dynamic};
        spec.tensor_parameters = {tp};
        BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");
        return spec;
    };

    Program prog_dyn = MakeProgramFromSpec(*mesh_device_, make_spec(/*dynamic=*/true));
    auto kernel = prog_dyn.impl().get_kernel_by_spec_name("dm_kernel");
    const auto& handles = kernel->tensor_binding_handles();
    ASSERT_EQ(handles.size(), 1u);
    EXPECT_EQ(handles[0].num_runtime_field_crta_words, 1u)
        << "Row-major interleaved + dynamic_tensor_shape: page size demotes to exactly one CRTA word.";
    EXPECT_TRUE(handles[0].runtime_field_is_page_size)
        << "The runtime field must be tagged as the page-size kind (not the sharded-shape kind).";

    // The binding's args_config CTA word carries the RuntimePageSize bit.
    const std::vector<uint32_t> dyn_ctas = kernel->compile_time_args();
    ASSERT_LT(handles[0].cta_offset, dyn_ctas.size());
    const auto dyn_cfg = tensor_accessor::ArgsConfig(
        static_cast<tensor_accessor::ArgsConfig::Underlying>(dyn_ctas[handles[0].cta_offset]));
    EXPECT_TRUE(dyn_cfg.test(tensor_accessor::ArgConfig::RuntimePageSize))
        << "RuntimePageSize bit must be set in the binding's args_config word.";

    // A-collapse: the dynamic binding omits the page-size CTA word that the static (bit-off) binding
    // carries, so the whole-kernel CTA count is exactly one shorter. The two specs are identical
    // apart from the flag, so the size delta is precisely the dropped page-size slot.
    Program prog_static = MakeProgramFromSpec(*mesh_device_, make_spec(/*dynamic=*/false));
    const std::vector<uint32_t> static_ctas =
        prog_static.impl().get_kernel_by_spec_name("dm_kernel")->compile_time_args();
    EXPECT_EQ(static_ctas.size(), dyn_ctas.size() + 1u)
        << "Static binding keeps the page-size CTA; the dynamic binding drops it (A-collapse).";
    const auto static_cfg = tensor_accessor::ArgsConfig(
        static_cast<tensor_accessor::ArgsConfig::Underlying>(static_ctas[handles[0].cta_offset]));
    EXPECT_FALSE(static_cfg.test(tensor_accessor::ArgConfig::RuntimePageSize))
        << "Without the flag, the RuntimePageSize bit must NOT be set.";
}

TEST_F(ProgramSpecTestGen1, CPU_DynamicTensorShape_InterleavedTileBindingHasNoRuntimeFieldCRTAs) {
    // Interleaved + TILE layout: the page size is dtype/tile-fixed, independent of logical shape, so
    // dynamic_tensor_shape does NOT demote it to a CRTA (the fold gates on ROW_MAJOR). The flag is a
    // pure host-side validation loosening here; the binding carries no runtime field words. Guards
    // the layout gate -- a regression that demoted tile page sizes would trip this.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
    TensorParameter tp{
        .unique_id = TensorParamName{"input_tensor"},
        .spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32, 32}, std::move(tensor_layout)),
        .relaxations = TensorSpecRelaxations{.dynamic_tensor_shape = true},
    };
    spec.tensor_parameters = {tp};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    const auto& handles = kernel->tensor_binding_handles();
    ASSERT_EQ(handles.size(), 1u);
    EXPECT_EQ(handles[0].num_runtime_field_crta_words, 0u)
        << "Interleaved TILE + dynamic_tensor_shape is host-side-only; no runtime CRTA words.";
    EXPECT_FALSE(handles[0].runtime_field_is_page_size);
}

TEST_F(ProgramSpecTestGen1, CPU_KernelCrtaLayout_AllThreeSectionsConsistent) {
    // The Kernel's stored KernelCrtaLayout must equal what a fresh walk of (named CRTAs +
    // binding handles) would compute. This test exercises a Program in which ALL THREE
    // sections of the CRTA buffer are non-empty:
    //   - section 1: named CRTAs           (declared in runtime_arg_schema)
    //   - section 2: TensorBinding section (variable-size: a plain interleaved binding +
    //                                       a sharded-with-dynamic_tensor_shape binding)
    //   - section 3: varargs               (declared via num_common_runtime_varargs)
    //
    // The headergen bakes vararg_section_offset into the kernel's `get_common_vararg(idx)`
    // macro, so a wrong offset here would silently route vararg reads into the binding
    // section. The walk-based reference value is exactly what genfiles used to compute on
    // its own; the refactor moves that computation into ResolveTensorBindingsForKernel and
    // threads it through. This test guards against the threading silently producing a
    // different value than the walk would.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    // Section 1: named CRTAs on the DM kernel.
    spec.kernels[0].runtime_arg_schema.common_runtime_arg_names = {"foo", "bar"};
    // Section 3: vararg CRTAs on the DM kernel.
    spec.kernels[0].advanced_options.num_common_runtime_varargs = 3;

    // Section 2: two bindings — one plain (1 word), one sharded+dynamic_tensor_shape
    // (1 word + tensor_shape_in_pages rank words).
    auto plain_tp = MakeMinimalTensorParameter("plain_tensor");
    auto dyn_tp =
        MakeShardedTensorParameter("dyn_tensor", tt::tt_metal::Shape{1, 1, 64, 32}, {32, 32}, /*num_cores=*/2);
    dyn_tp.relaxations = TensorSpecRelaxations{.dynamic_tensor_shape = true};
    spec.tensor_parameters = {plain_tp, dyn_tp};
    BindTensorParameterToKernel(spec.kernels[0], "plain_tensor", "plain_ta");
    BindTensorParameterToKernel(spec.kernels[0], "dyn_tensor", "dyn_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    const KernelCrtaLayout layout = kernel->get_crta_layout();

    // Reference values, re-derived independently of the layout struct.
    const uint32_t expected_named_words = 2u;  // "foo", "bar"
    uint32_t expected_binding_words = 0;
    for (const auto& handle : kernel->tensor_binding_handles()) {
        expected_binding_words += 1u + handle.num_runtime_field_crta_words;
    }
    const uint32_t expected_vararg_offset = expected_named_words + expected_binding_words;

    EXPECT_EQ(layout.num_named_words, expected_named_words);
    EXPECT_EQ(layout.binding_section_words, expected_binding_words);
    EXPECT_EQ(layout.vararg_section_offset, expected_vararg_offset)
        << "vararg_section_offset must equal num_named_words + binding_section_words; this is the "
           "value baked into get_common_vararg(idx) by the kernel headergen.";

    // Sanity: the dynamic-shape binding's runtime-field word count should be > 0, so this test
    // genuinely exercises a variable-size binding (not just two 1-word bindings that would also
    // pass with the old binding-count-based math).
    ASSERT_EQ(kernel->tensor_binding_handles().size(), 2u);
    EXPECT_GT(kernel->tensor_binding_handles()[1].num_runtime_field_crta_words, 0u)
        << "Test precondition: the second binding should be variable-size; otherwise the layout "
           "calculation degenerates to the pre-refactor case.";
}

}  // namespace
}  // namespace tt::tt_metal::experimental
