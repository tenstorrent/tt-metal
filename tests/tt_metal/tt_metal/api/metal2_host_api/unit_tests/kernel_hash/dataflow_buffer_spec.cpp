// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DataflowBufferSpec fields that must change the bound kernels' JIT cache key (compute_hash).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <optional>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, DFBTileMetadataAffectsKernelHash) {
    auto make_spec = [](std::optional<Tile> tile) {
        ProgramSpec spec = MakeMinimalValidProgramSpec();
        spec.dataflow_buffers[0].tile_format_metadata = tile;
        return spec;
    };

    Program prog_default = MakeProgramFromSpec(*mesh_device_, make_spec(std::nullopt));
    Program prog_wide = MakeProgramFromSpec(*mesh_device_, make_spec(Tile{{16, 32}}));

    auto hash_default = prog_default.impl().get_kernel_by_spec_name("compute_kernel")->compute_hash();
    auto hash_wide = prog_wide.impl().get_kernel_by_spec_name("compute_kernel")->compute_hash();
    EXPECT_NE(hash_default, hash_wide);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
