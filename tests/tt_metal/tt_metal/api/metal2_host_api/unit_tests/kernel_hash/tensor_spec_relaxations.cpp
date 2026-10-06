// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TensorSpecRelaxations::dynamic_tensor_shape keeps the kernel's JIT cache key (compute_hash) stable
// across tensor shapes.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <utility>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeShardedTensorParameter;
using test_helpers::ProgramSpecTestGen1;

TEST_F(ProgramSpecTestGen1, CPU_DynamicTensorShape_InterleavedKernelHashStableAcrossShapes_TileLayout) {
    // Interleaved + TILE layout: page_size is constant per dtype regardless of logical_shape, so
    // the CTAs ([args_config.raw(), aligned_page_size]) are stable across shape variations.
    // The dynamic flag thus enables JIT cache reuse for tile-layout eltwise.
    //
    // (Row-major interleaved has a shape-dependent page_size — the last-dim element count. Under
    // dynamic_tensor_shape the resolver demotes that page size to a per-binding CRTA word, so its
    // CTAs become shape-stable too; that path is covered by
    // DynamicTensorShape_InterleavedRowMajorKernelHashStableAcrossWidths below. This test focuses on
    // tile, whose page size is dtype-fixed and never rode a shape-dependent CTA in the first place.)
    auto make_spec = [](tt::tt_metal::Shape shape) {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE);
        auto memory_config =
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
        auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
        TensorParameter tp{
            .unique_id = TensorParamName{"input_tensor"},
            .spec = tt::tt_metal::TensorSpec(std::move(shape), std::move(tensor_layout)),
            .relaxations = TensorSpecRelaxations{.dynamic_tensor_shape = true},
        };
        spec.tensor_parameters = {tp};
        BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");
        return spec;
    };
    Program prog_a = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::Shape{1, 1, 32, 32}));
    Program prog_b = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::Shape{1, 1, 64, 64}));
    EXPECT_EQ(
        prog_a.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash(),
        prog_b.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash())
        << "Interleaved tile + dynamic_tensor_shape: shape variations must hash equal so the "
           "same compiled kernel binary is reused.";
}

TEST_F(ProgramSpecTestGen1, CPU_DynamicTensorShape_InterleavedRowMajorKernelHashStableAcrossWidths) {
    // The payoff of the page-size fold. For a ROW-MAJOR interleaved TensorParameter the page size
    // (last_dim_width * elem_size) is shape-dependent. WITHOUT the flag it rides a compile-time arg,
    // so two different-width tensors hash differently -- distinct cache entries, and (worse) a stale
    // page size baked into a binary that gets reused on a cache hit: the exact bug this feature
    // fixes. WITH dynamic_tensor_shape the page size moves to a CRTA, the CTAs become width-
    // independent, and the two widths hash identically -- one cached binary, refreshed per-dispatch.
    auto make_spec = [](uint32_t width, bool dynamic) {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
        auto memory_config =
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
        auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
        TensorParameter tp{
            .unique_id = TensorParamName{"input_tensor"},
            .spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 1, 32, width}, std::move(tensor_layout)),
            .relaxations = TensorSpecRelaxations{.dynamic_tensor_shape = dynamic},
        };
        spec.tensor_parameters = {tp};
        BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");
        return spec;
    };

    // Baseline (flag off): different widths -> different page-size CTA -> different hash.
    Program s_a = MakeProgramFromSpec(*mesh_device_, make_spec(/*width=*/64, /*dynamic=*/false));
    Program s_b = MakeProgramFromSpec(*mesh_device_, make_spec(/*width=*/128, /*dynamic=*/false));
    EXPECT_NE(
        s_a.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash(),
        s_b.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash())
        << "Baseline: row-major page size rides a CTA, so different widths must hash differently.";

    // With the flag: page size -> CRTA, CTAs width-independent -> identical hash (cache reuse).
    Program d_a = MakeProgramFromSpec(*mesh_device_, make_spec(/*width=*/64, /*dynamic=*/true));
    Program d_b = MakeProgramFromSpec(*mesh_device_, make_spec(/*width=*/128, /*dynamic=*/true));
    EXPECT_EQ(
        d_a.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash(),
        d_b.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash())
        << "Row-major interleaved + dynamic_tensor_shape: page size moves to a CRTA, so different "
           "widths hash equally (program-cache reuse; prevents the stale-page-size bug).";
}

TEST_F(ProgramSpecTestGen1, CPU_DynamicTensorShape_ShardedKernelHashStableAcrossShapes) {
    // Sharded TensorParameters DO encode tensor_shape_in_pages in CTAs by default — so without
    // the dynamic flag, two different-shape sharded TPs hash differently. With the flag, the
    // tensor_shape words move to CRTAs and the CTAs become stable across shape variations.
    //
    // Layout: HEIGHT_SHARDED with shard_shape {32, 32} on 2 cores → 2 shards along height,
    // full width per shard. The declared (64, 32) tensor has 2 shards; the alternate (32, 32)
    // tensor needs only 1 shard (subset of the 2-core grid).
    auto make_spec = [](const tt::tt_metal::Shape& shape, bool dynamic) {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        auto tp = MakeShardedTensorParameter("input_tensor", shape, {32, 32}, /*num_cores=*/2);
        tp.relaxations = TensorSpecRelaxations{.dynamic_tensor_shape = dynamic};
        spec.tensor_parameters = {tp};
        BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");
        return spec;
    };

    // Without the flag: differently-shaped sharded TPs hash differently (regression canary).
    Program prog_static_a = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::Shape{1, 1, 64, 32}, false));
    Program prog_static_b = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::Shape{1, 1, 32, 32}, false));
    EXPECT_NE(
        prog_static_a.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash(),
        prog_static_b.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash())
        << "Baseline: without dynamic_tensor_shape, different shapes should hash differently.";

    // With the flag: same two shapes hash identically — CTAs are now stable.
    Program prog_dyn_a = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::Shape{1, 1, 64, 32}, true));
    Program prog_dyn_b = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::Shape{1, 1, 32, 32}, true));
    EXPECT_EQ(
        prog_dyn_a.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash(),
        prog_dyn_b.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash())
        << "Sharded + dynamic_tensor_shape: tensor_shape moves to CRTAs, so CTA-driven hash "
           "must be stable across shape variations.";
}

}  // namespace
}  // namespace tt::tt_metal::experimental
