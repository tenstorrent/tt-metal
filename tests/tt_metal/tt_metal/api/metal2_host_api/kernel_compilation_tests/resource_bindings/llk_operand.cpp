// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile tests for the LLK operand metadata (data format, tile and face geometry) that DFB,
// scratchpad and tensor bindings carry to device code (mock Quasar and mock Blackhole).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <array>
#include <cstdint>
#include <string>
#include <utility>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeShardedTensorParameter;
using test_helpers::ProgramSpecTestBlackhole;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, DFBCustomTileCompiles) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.dataflow_buffers[0].tile_format_metadata = Tile{{8, 32}};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

// LLKOperandFrom pulls experimental/2_0/llk_operand.h (Blackhole-only). Fold checks that name the
// alias must JIT under a Blackhole mock; WH Gen1 cannot include that header.
// Fold checks for LLKOperandFrom (SPEC Part II). Reuses ProgramSpecTestBlackhole from mock_device_fixtures.hpp.
using LLKOperandInterop = ProgramSpecTestBlackhole;

TEST_F(LLKOperandInterop, ScratchpadWithoutMetadataFailsToFormOperand) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.kernels[1].source = KernelSpec::SourceCode{R"(
#include "api/llk_operand_from_tokens.h"
void kernel_main() {
    using PadOp = LLKOperandFrom<scratch::pad>;
    (void)sizeof(PadOp);
}
)"};
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 1024,
    }};
    spec.kernels[1].scratchpad_bindings.push_back(
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"});

    EXPECT_ANY_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(LLKOperandInterop, ScratchpadFormatAloneSucceeds) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.kernels[1].source = KernelSpec::SourceCode{R"(
#include "api/llk_operand_from_tokens.h"
void kernel_main() {
    Scratchpad<uint32_t> pad(scratch::pad);
    using PadOp = LLKOperandFrom<scratch::pad>;
    static_assert(PadOp::descriptor.format == DataFormat::Float16_b);
    static_assert(PadOp::descriptor.shape.face_r_dim == 16);
    static_assert(PadOp::descriptor.shape.face_c_dim == 16);
    static_assert(PadOp::descriptor.shape.num_faces_r_dim == 2);
    static_assert(PadOp::descriptor.shape.num_faces_c_dim == 2);
    (void)pad.operand<PadOp>();
}
)"};
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 1024,
        .data_format_metadata = tt::DataFormat::Float16_b,
    }};
    spec.kernels[1].scratchpad_bindings.push_back(
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"});

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(LLKOperandInterop, ScratchpadFormatAndTileSucceeds) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.kernels[1].source = KernelSpec::SourceCode{R"(
#include "api/llk_operand_from_tokens.h"
void kernel_main() {
    Scratchpad<uint32_t> pad(scratch::pad);
    using PadOp = LLKOperandFrom<scratch::pad>;
    static_assert(PadOp::descriptor.format == DataFormat::Float16_b);
    static_assert(PadOp::descriptor.shape.face_r_dim == 16);
    static_assert(PadOp::descriptor.shape.face_c_dim == 16);
    static_assert(PadOp::descriptor.shape.num_faces_r_dim == 1);
    static_assert(PadOp::descriptor.shape.num_faces_c_dim == 2);
    (void)pad.operand<PadOp>();
}
)"};
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 1024,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .tile_format_metadata = Tile{{16, 32}},
    }};
    spec.kernels[1].scratchpad_bindings.push_back(
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"});

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(LLKOperandInterop, ScratchpadCustomFaceTileSucceeds) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.kernels[1].source = KernelSpec::SourceCode{R"(
#include "api/llk_operand_from_tokens.h"
void kernel_main() {
    Scratchpad<uint32_t> pad(scratch::pad);
    using PadOp = LLKOperandFrom<scratch::pad>;
    static_assert(PadOp::descriptor.format == DataFormat::Float16_b);
    static_assert(PadOp::descriptor.shape.face_r_dim == 1);
    static_assert(PadOp::descriptor.shape.face_c_dim == 16);
    static_assert(PadOp::descriptor.shape.num_faces_r_dim == 2);
    static_assert(PadOp::descriptor.shape.num_faces_c_dim == 2);
    (void)pad.operand<PadOp>();
}
)"};
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 1024,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .tile_format_metadata = Tile({2, 32}, {1, 16}),
    }};
    spec.kernels[1].scratchpad_bindings.push_back(
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"});

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(LLKOperandInterop, DFBDefaultTileCompiles) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.kernels[1].source = KernelSpec::SourceCode{R"(
#include "api/llk_operand_from_tokens.h"
void kernel_main() {
    DataflowBuffer in(dfb::input_dfb);
    using InOp = LLKOperandFrom<dfb::input_dfb>;
    static_assert(InOp::descriptor.format == DataFormat::Float16_b);
    static_assert(InOp::descriptor.shape.face_r_dim == 16);
    static_assert(InOp::descriptor.shape.face_c_dim == 16);
    static_assert(InOp::descriptor.shape.num_faces_r_dim == 2);
    static_assert(InOp::descriptor.shape.num_faces_c_dim == 2);
    (void)in.front<InOp>();
    (void)in.back<InOp>();
}
)"};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(LLKOperandInterop, DFBFaceGeometryCompiles) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.dataflow_buffers[0].tile_format_metadata = Tile({2, 32}, {1, 16});
    spec.kernels[1].source = KernelSpec::SourceCode{R"(
#include "api/llk_operand_from_tokens.h"
void kernel_main() {
    DataflowBuffer in(dfb::input_dfb);
    using InOp = LLKOperandFrom<dfb::input_dfb>;
    static_assert(InOp::descriptor.format == DataFormat::Float16_b);
    static_assert(InOp::descriptor.shape.face_r_dim == 1);
    static_assert(InOp::descriptor.shape.face_c_dim == 16);
    static_assert(InOp::descriptor.shape.num_faces_r_dim == 2);
    static_assert(InOp::descriptor.shape.num_faces_c_dim == 2);
    (void)in.front<InOp>();
    (void)in.back<InOp>();
}
)"};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

// An L1-sharded tensor whose page tile is exactly `tile_shape`, so LLKOperandFrom sees a
// non-default face grid.
TensorParameter MakeL1ShardedTiledTensor(std::string name, std::array<uint32_t, 2> tile_shape) {
    return MakeShardedTensorParameter(
        std::move(name),
        tt::tt_metal::Shape{1, 1, tile_shape[0], tile_shape[1]},
        tile_shape,
        /*num_cores=*/1,
        tt::tt_metal::Layout::TILE,
        Tile{tile_shape});
}

TEST_F(LLKOperandInterop, L1TensorDefaultCompiles) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.tensor_parameters = {MakeShardedTensorParameter("a", tt::tt_metal::Shape{1, 1, 32, 32}, {32, 32}, 2)};
    BindTensorParameterToKernel(spec.kernels[1], "a", "a");
    spec.kernels[1].source = KernelSpec::SourceCode{R"(
#include "api/llk_operand_from_tokens.h"
#include "api/tensor/local_tensor_accessor.h"
void kernel_main() {
    LocalTensorAccessor<uint32_t> a(tensor::a);
    using AOp = LLKOperandFrom<tensor::a>;
    static_assert(AOp::descriptor.format == DataFormat::Float16_b);
    static_assert(AOp::descriptor.shape.face_r_dim == 16);
    static_assert(AOp::descriptor.shape.face_c_dim == 16);
    static_assert(AOp::descriptor.shape.num_faces_r_dim == 2);
    static_assert(AOp::descriptor.shape.num_faces_c_dim == 2);
    (void)a.operand<AOp>();
}
)"};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(LLKOperandInterop, L1Tensor16x32Compiles) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.tensor_parameters = {MakeL1ShardedTiledTensor("a", {16, 32})};
    BindTensorParameterToKernel(spec.kernels[1], "a", "a");
    spec.kernels[1].source = KernelSpec::SourceCode{R"(
#include "api/llk_operand_from_tokens.h"
#include "api/tensor/local_tensor_accessor.h"
void kernel_main() {
    LocalTensorAccessor<uint32_t> a(tensor::a);
    using AOp = LLKOperandFrom<tensor::a>;
    static_assert(AOp::descriptor.format == DataFormat::Float16_b);
    static_assert(AOp::descriptor.shape.face_r_dim == 16);
    static_assert(AOp::descriptor.shape.face_c_dim == 16);
    static_assert(AOp::descriptor.shape.num_faces_r_dim == 1);
    static_assert(AOp::descriptor.shape.num_faces_c_dim == 2);
    (void)a.operand<AOp>();
}
)"};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(LLKOperandInterop, L1Tensor32x16Compiles) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.tensor_parameters = {MakeL1ShardedTiledTensor("a", {32, 16})};
    BindTensorParameterToKernel(spec.kernels[1], "a", "a");
    spec.kernels[1].source = KernelSpec::SourceCode{R"(
#include "api/llk_operand_from_tokens.h"
#include "api/tensor/local_tensor_accessor.h"
void kernel_main() {
    LocalTensorAccessor<uint32_t> a(tensor::a);
    using AOp = LLKOperandFrom<tensor::a>;
    static_assert(AOp::descriptor.format == DataFormat::Float16_b);
    static_assert(AOp::descriptor.shape.face_r_dim == 16);
    static_assert(AOp::descriptor.shape.face_c_dim == 16);
    static_assert(AOp::descriptor.shape.num_faces_r_dim == 2);
    static_assert(AOp::descriptor.shape.num_faces_c_dim == 1);
    (void)a.operand<AOp>();
}
)"};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
