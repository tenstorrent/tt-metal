// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramRunArgs::tensor_args in SetProgramRunArgs: completeness, TensorSpec matching (strict and
// relaxed), and the runtime tensor shape written into CRTAs.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"
#include "metal2_host_api/run_args_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeRunArgsForMinimalSpec;
using test_helpers::MakeShardedTensorParameter;
using test_helpers::ProgramRunArgsTestGen1;
using test_helpers::ReadBindingAddressFromCRTA;

TEST_F(ProgramRunArgsTestGen1, CPU_MissingTensorArgFails) {
    // Spec declares a TensorParameter; user supplies no TensorArgument entry for it.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor")};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Run params with no tensor_args entries.
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "TensorParameter 'input_tensor' is declared in the Program but has no TensorArgument entry")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_UnknownTensorParameterInRunArgsFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, binding.spec);

    // tensor_parameter_name doesn't match any TensorParameter in the spec.
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"ghost_tensor"}, TensorArgument{tensor}},
    };

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("TensorArgument references unknown TensorParameter 'ghost_tensor'")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_TensorSpecMismatchFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    // Binding declares spec with shape {1, 32}; runtime tensor has a different shape ({1, 64}).
    auto binding = MakeMinimalTensorParameter("input_tensor");  // default {1, 32}
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
    auto wrong_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 64}, tensor_layout);  // different shape!
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, wrong_spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };

    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "TensorArgument for binding 'input_tensor' supplied a MeshTensor whose TensorSpec does not match")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_TensorBindingOnlyKernelOmittedFromRunArgsSucceeds) {
    // A kernel with tensor bindings but an empty RTA/CRTA schema may be omitted from
    // kernel_run_args: SetProgramRunArgs fills its binding-section CRTAs (base addresses, dynamic
    // accessor fields) in a second pass over all binding-bearing kernels, so the binding address
    // still reaches the device. (Gen1 counterpart of the Quasar BindingOnlyKernelOmitted... test.)
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    // Bind to dm_kernel (compute_kernel has no bindings). dm_kernel has no named RTAs/CRTAs and no
    // varargs, so the binding is its only per-enqueue state — and no longer forces a run-args entry.
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Supply the bound tensor but no kernel_run_args entry for the binding kernel.
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, binding.spec);
    ProgramRunArgs params;
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };

    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
    EXPECT_EQ(ReadBindingAddressFromCRTA(program, "dm_kernel", "input_tensor"), static_cast<uint32_t>(tensor.address()))
        << "binding address should be written even though the kernel was omitted from kernel_run_args";
}

// ============================================================================
// Dynamic Tensor Shape Run-Params Tests (Gen1 / WH)
// ============================================================================
// Exercises the dynamic_tensor_shape opt-in on TensorParameter:
//   - validation: tensor_layout must still match exactly; logical_shape may vary per-dim
//     but rank must be preserved.
//   - CRTA contents: for sharded TPs, the runtime tensor's shape-in-pages is written into
//     CRTAs immediately after the binding's address slot, on both SetProgramRunArgs
//     and UpdateTensorArgs.

TEST_F(ProgramRunArgsTestGen1, CPU_DynamicTensorShape_InterleavedAcceptsDifferentShape) {
    // Declared shape {1, 32}; runtime tensor with shape {1, 64} is accepted because
    // dynamic_tensor_shape is set and everything else matches.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");  // shape {1, 32}
    binding.relaxations = TensorSpecRelaxations{.dynamic_tensor_shape = true};
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Different shape, same layout.
    auto wrong_spec_layout = binding.spec.tensor_layout();
    auto larger_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 64}, wrong_spec_layout);
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, larger_spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestGen1, CPU_DynamicTensorShape_DTypeMismatchStillFails) {
    // dynamic_tensor_shape loosens shape only — dtype (part of tensor_layout) must still match.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");  // BFLOAT16
    binding.relaxations = TensorSpecRelaxations{.dynamic_tensor_shape = true};
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Different dtype (UINT32 instead of BFLOAT16).
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto wrong_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::UINT32, page_config, memory_config);
    auto wrong_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 32}, wrong_layout);
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, wrong_spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("tensor_layout does not match the binding's declared layout")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_DynamicTensorShape_RankMismatchFails) {
    // dynamic_tensor_shape lets per-dim shape values vary, but the rank must remain constant.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");  // rank-2 shape {1, 32}
    binding.relaxations = TensorSpecRelaxations{.dynamic_tensor_shape = true};
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Rank-3 tensor with same layout.
    auto wrong_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32}, binding.spec.tensor_layout());
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, wrong_spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("logical_shape rank (3) differs from the declared rank (2)")));
}

// The same bind that CPU_DynamicTensorShape_RankMismatchFails rejects is ACCEPTED once
// relax_logical_rank is added -- the two tests are a pair, and the pairing is the point: the rank
// is the one shape term dynamic_tensor_shape still pins, and this flag is the only way to free it.
TEST_F(ProgramRunArgsTestGen1, CPU_RelaxLogicalRank_RankMismatchAccepted) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");  // rank-2 shape {1, 32}
    binding.relaxations = TensorSpecRelaxations{
        .dynamic_tensor_shape = true,
        .relax_logical_rank = true,
    };
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Rank-3 tensor with same layout -- rejected without relax_logical_rank.
    auto runtime_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32}, binding.spec.tensor_layout());
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, runtime_spec);
    ASSERT_NE(runtime_spec.logical_shape().rank(), binding.spec.logical_shape().rank());

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

// Read the runtime field CRTA words (immediately after the address slot) for a tensor binding.
// Returns the [rank] shape words in declaration order.
inline std::vector<uint32_t> ReadBindingShapeFromCRTA(
    const Program& program, const std::string& kernel_name, const std::string& tensor_parameter_name) {
    auto kernel = program.impl().get_kernel_by_spec_name(kernel_name);
    for (const auto& handle : kernel->tensor_binding_handles()) {
        if (handle.tensor_parameter_name == tensor_parameter_name) {
            const uint32_t base_word = handle.addr_crta_offset / sizeof(uint32_t);
            const auto* data = kernel->common_runtime_args_data().data();
            std::vector<uint32_t> shape;
            shape.reserve(handle.num_runtime_field_crta_words);
            for (uint32_t i = 0; i < handle.num_runtime_field_crta_words; ++i) {
                shape.push_back(data[base_word + 1u + i]);
            }
            return shape;
        }
    }
    ADD_FAILURE() << "No binding handle for '" << tensor_parameter_name << "' on kernel '" << kernel_name << "'";
    return {};
}

// Compute the expected tensor_shape_in_pages from a TensorSpec via its BufferDistributionSpec.
// This is the source of truth for what gets written into the runtime field CRTA section.
inline std::vector<uint32_t> ExpectedShapeInPagesFromSpec(const tt::tt_metal::TensorSpec& spec) {
    const auto bds = spec.compute_buffer_sharding_args().buffer_distribution_spec();
    if (!bds.has_value()) {
        ADD_FAILURE() << "TensorSpec has no BufferDistributionSpec; expected sharded";
        return {};
    }
    const auto& shape = bds->tensor_shape_in_pages();
    std::vector<uint32_t> out;
    out.reserve(shape.rank());
    for (size_t i = 0; i < shape.rank(); ++i) {
        out.push_back(static_cast<uint32_t>(shape[i]));
    }
    return out;
}

TEST_F(ProgramRunArgsTestGen1, CPU_DynamicTensorShape_ShardedSetWritesShapeIntoCRTAs) {
    // Sharded + dynamic_tensor_shape: SetProgramRunArgs must write the actual runtime
    // tensor's tensor_shape_in_pages into the CRTA section that follows the binding's address.
    // Layout: HEIGHT_SHARDED with shard_shape {32, 32} on 2 cores → 2 shards along height.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding =
        MakeShardedTensorParameter("input_tensor", tt::tt_metal::Shape{1, 1, 64, 32}, {32, 32}, /*num_cores=*/2);
    binding.relaxations = TensorSpecRelaxations{.dynamic_tensor_shape = true};
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, binding.spec);
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };
    SetProgramRunArgs(program, params);

    // BDS flattens shape per its sharding scheme; derive expected value from BDS directly
    // (source of truth — same path the runtime uses to populate the CRTA slots).
    EXPECT_EQ(
        ReadBindingShapeFromCRTA(program, "dm_kernel", "input_tensor"), ExpectedShapeInPagesFromSpec(binding.spec));
}

TEST_F(ProgramRunArgsTestGen1, CPU_DynamicTensorShape_ShardedUpdateRefreshesShape) {
    // Sharded + dynamic_tensor_shape: UpdateTensorArgs must refresh BOTH the address slot and
    // the runtime shape slots when bound to a tensor of different shape (but same layout).
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding =
        MakeShardedTensorParameter("input_tensor", tt::tt_metal::Shape{1, 1, 64, 32}, {32, 32}, /*num_cores=*/2);
    binding.relaxations = TensorSpecRelaxations{.dynamic_tensor_shape = true};
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // First Set: tensor of declared shape (2 shards along height).
    MeshTensor tensor1 = MeshTensor::allocate_on_device(*mesh_device_, binding.spec);
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor1}},
    };
    SetProgramRunArgs(program, params);
    ASSERT_EQ(
        ReadBindingShapeFromCRTA(program, "dm_kernel", "input_tensor"), ExpectedShapeInPagesFromSpec(binding.spec));

    // Second Update: smaller-shape tensor (1 shard along height). Same shard_spec, fewer shards.
    auto smaller_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32, 32}, binding.spec.tensor_layout());
    MeshTensor tensor2 = MeshTensor::allocate_on_device(*mesh_device_, smaller_spec);
    Table<TensorParamName, TensorArgument> tensor_args{
        {TensorParamName{"input_tensor"}, TensorArgument{tensor2}},
    };
    EXPECT_NO_THROW(UpdateTensorArgs(program, tensor_args));

    EXPECT_EQ(
        ReadBindingAddressFromCRTA(program, "dm_kernel", "input_tensor"), static_cast<uint32_t>(tensor2.address()));
    EXPECT_EQ(
        ReadBindingShapeFromCRTA(program, "dm_kernel", "input_tensor"), ExpectedShapeInPagesFromSpec(smaller_spec));
}

// ============================================================================
// match_padded_shape_only Run-Params Tests (Gen1 / WH)
// ============================================================================
// Exercises the match_padded_shape_only opt-in on TensorParameter:
//   - tensor_layout must still match exactly (dtype / page_config / memory_config / alignment).
//   - padded_shape() must match exactly across binds.
//   - logical_shape() may differ provided the resulting padded_shape is unchanged.
//   - Strictly weaker than dynamic_tensor_shape; no device-side CTA/CRTA effect.

TEST_F(ProgramRunArgsTestGen1, CPU_MatchPaddedShapeOnly_AcceptsDifferentLogicalShape) {
    // Declared logical shape {1, 1, 32, 32} on TILE layout produces padded_shape {1, 1, 32, 32}.
    // A runtime tensor with logical_shape {1, 1, 20, 20} pads up to the same {1, 1, 32, 32}, so
    // match_padded_shape_only accepts the rebind.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
    auto declared_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32, 32}, layout);
    TensorParameter binding{
        .unique_id = TensorParamName{"input_tensor"},
        .spec = declared_spec,
        .relaxations = TensorSpecRelaxations{.match_padded_shape_only = true},
    };
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Runtime tensor: smaller logical shape that pads up to the same padded_shape (one tile).
    auto runtime_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 20, 20}, layout);
    ASSERT_EQ(runtime_spec.padded_shape(), declared_spec.padded_shape())
        << "Test precondition: runtime logical {1,1,20,20} should pad to the same {1,1,32,32} as declared.";
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, runtime_spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

TEST_F(ProgramRunArgsTestGen1, CPU_MatchPaddedShapeOnly_PaddedShapeMismatchFails) {
    // Same TensorLayout, but a logical shape that pads to a DIFFERENT padded_shape is rejected.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
    auto declared_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32, 32}, layout);
    TensorParameter binding{
        .unique_id = TensorParamName{"input_tensor"},
        .spec = declared_spec,
        .relaxations = TensorSpecRelaxations{.match_padded_shape_only = true},
    };
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Runtime tensor: logical shape that pads up to a DIFFERENT padded_shape (two tiles wide).
    auto runtime_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32, 33}, layout);
    ASSERT_NE(runtime_spec.padded_shape(), declared_spec.padded_shape())
        << "Test precondition: runtime logical {1,1,32,33} should pad to a different padded_shape than declared.";
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, runtime_spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("padded_shape does not match the binding's declared padded_shape")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_MatchPaddedShapeOnly_DTypeMismatchStillFails) {
    // match_padded_shape_only loosens only along logical_shape. tensor_layout fields (dtype here)
    // must still match exactly.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");  // BFLOAT16
    binding.relaxations = TensorSpecRelaxations{.match_padded_shape_only = true};
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto wrong_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::UINT32, page_config, memory_config);
    auto wrong_spec = tt::tt_metal::TensorSpec(binding.spec.logical_shape(), wrong_layout);
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, wrong_spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("tensor_layout does not match the binding's declared layout")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_MatchPaddedShapeOnly_DynamicWinsWhenBothSet) {
    // When both match_padded_shape_only and dynamic_tensor_shape are set, dynamic is more
    // permissive and wins. A runtime tensor whose padded_shape differs from declared should
    // be accepted (which match_padded_shape_only alone would reject).
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");  // shape {1, 32}
    binding.relaxations = TensorSpecRelaxations{.match_padded_shape_only = true, .dynamic_tensor_shape = true};
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Different logical shape; for ROW_MAJOR this also gives a different padded_shape, which
    // dynamic accepts but padded_only alone would not.
    auto wrong_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 64}, binding.spec.tensor_layout());
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, wrong_spec);

    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };
    EXPECT_NO_THROW(SetProgramRunArgs(program, params));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
