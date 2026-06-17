// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for https://github.com/tenstorrent/tt-metal/issues/45331
//
// On-device ttnn::tilize crashes with an L1 OOM (statically allocated
// circular buffers grow well beyond per-core L1) when the destination
// memory_config is a wide WIDTH_SHARDED DRAM tile-layout target.
//
// The original Python reproducer hits this path via
//   ttnn.from_torch(..., layout=ttnn.TILE_LAYOUT,
//                   memory_config=<width-sharded DRAM>, device=mesh_device)
// which lands the torch tensor on device as ROW_MAJOR DRAM interleaved and
// then runs on-device tilize to convert to the sharded TILE target. This
// test exercises just that final tilize hop directly.

#include <array>

#include <gtest/gtest.h>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/shape.hpp>

#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::data_movement::test {

using TilizeWidthShardedDramFixture = ttnn::TTNNFixtureWithDevice;

TEST_F(TilizeWidthShardedDramFixture, WidthShardedDramTargetDoesNotCrash) {
    auto& device = *device_;

    // Same shape, dtype, and target sharding as the reproducer in #45331:
    //   2048 x 42752 bfloat16, width-sharded across 12 cores on row 0,
    //   per-shard shape (2048, 3584), ROW_MAJOR orientation, DRAM, TILE output.
    const ttnn::Shape shape({2048, 42752});

    auto input_tensor =
        ttnn::zeros(shape, DataType::BFLOAT16, ttnn::ROW_MAJOR_LAYOUT, std::ref(device), ttnn::DRAM_MEMORY_CONFIG);

    const CoreRangeSet shard_grid(CoreRange(CoreCoord{0, 0}, CoreCoord{11, 0}));
    const ShardSpec shard_spec(shard_grid, std::array<uint32_t, 2>{2048, 3584}, ShardOrientation::ROW_MAJOR);
    const MemoryConfig sharded_dram_cfg(TensorMemoryLayout::WIDTH_SHARDED, BufferType::DRAM, shard_spec);

    // The bug manifests as a TT_THROW during tilize program compile (CB sizing),
    // not as a wrong numerical result. We only assert the call doesn't throw.
    EXPECT_NO_THROW({ auto out = ttnn::tilize(input_tensor, sharded_dram_cfg); });
}

}  // namespace ttnn::operations::data_movement::test
