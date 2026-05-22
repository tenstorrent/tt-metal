// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/ttnn/unit_tests/gtests/accessor/test_param_sets.hpp"

#include <tt-metalium/core_coord.hpp>

namespace tensor_accessor_test_params {

using tt::tt_metal::BufferType;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::NdShardSpec;
using tt::tt_metal::Shape;
using tt::tt_metal::ShardDistributionStrategy;
using tt::tt_metal::ShardOrientation;

std::vector<InputOutputBufferParams> get_sharded_reshard_base_params() {
    return {
        InputOutputBufferParams{
            .tensor_shape = Shape{4, 64, 96},
            .layout = Layout::TILE,
            .dtype = DataType::UINT16,
            .input_buffer_type = BufferType::L1,
            .output_buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{1, 32, 64},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{2, 32, 32},
                    .grid = CoreRangeSet(CoreRange({4, 4}, {5, 5})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = Shape{18, 128, 64},
            .layout = Layout::TILE,
            .dtype = DataType::UINT8,
            .input_buffer_type = BufferType::L1,
            .output_buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{1, 64, 96},
                    .grid = CoreRangeSet(tt::stl::Span<const CoreRange>(
                        {CoreRange({4, 6}, {6, 6}), CoreRange({1, 1}, {1, 1}), CoreRange({0, 3}, {3, 3})})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{1, 32, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {5, 5})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = Shape{2, 3, 256},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .input_buffer_type = BufferType::L1,
            .output_buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{2, 4, 16},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{1, 3, 16},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {2, 2})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = Shape{3, 2, 2, 3, 4},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .input_buffer_type = BufferType::L1,
            .output_buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{1, 1, 2, 2, 4},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 4})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{3, 3, 1, 1, 4},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = Shape{5, 2, 2, 64, 96},
            .layout = Layout::TILE,
            .dtype = DataType::UINT16,
            .input_buffer_type = BufferType::L1,
            .output_buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{1, 1, 2, 64, 64},
                    .grid = CoreRangeSet(
                        tt::stl::Span<const CoreRange>({CoreRange({0, 0}, {2, 0}), CoreRange({0, 1}, {1, 1})})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{5, 1, 1, 96, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = Shape{64, 64},
            .layout = Layout::TILE,
            .dtype = DataType::UINT16,
            .input_buffer_type = BufferType::DRAM,
            .output_buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{32, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 0})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{32, 32},
                    .grid = CoreRangeSet(CoreRange({4, 4}, {5, 5})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = Shape{4, 64, 96},
            .layout = Layout::TILE,
            .dtype = DataType::UINT16,
            .input_buffer_type = BufferType::DRAM,
            .output_buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{1, 64, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 0})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{2, 32, 32},
                    .grid = CoreRangeSet(CoreRange({4, 4}, {5, 5})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = Shape{18, 128, 64},
            .layout = Layout::TILE,
            .dtype = DataType::UINT8,
            .input_buffer_type = BufferType::L1,
            .output_buffer_type = BufferType::DRAM,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{1, 64, 96},
                    .grid = CoreRangeSet(tt::stl::Span<const CoreRange>(
                        {CoreRange({4, 6}, {6, 6}), CoreRange({1, 1}, {1, 1}), CoreRange({0, 3}, {3, 3})})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{2, 32, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {0, 0})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = Shape{2, 3, 256},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .input_buffer_type = BufferType::DRAM,
            .output_buffer_type = BufferType::DRAM,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{2, 4, 16},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {0, 0})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{1, 3, 16},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {0, 0})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
        },
    };
}

std::vector<CopyParams> get_interleaved_copy_params() {
    return {
        // 2D cases - L1 buffer type
        CopyParams{
            .tensor_shape = Shape{64, 128},
            .layout = Layout::TILE,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::L1,
        },
        CopyParams{
            .tensor_shape = Shape{96, 64},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT16,
            .buffer_type = BufferType::L1,
        },
        CopyParams{
            .tensor_shape = Shape{128, 96},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
        },
        // 2D cases - DRAM buffer type
        CopyParams{
            .tensor_shape = Shape{64, 128},
            .layout = Layout::TILE,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::DRAM,
        },
        CopyParams{
            .tensor_shape = Shape{96, 64},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT16,
            .buffer_type = BufferType::DRAM,
        },
        CopyParams{
            .tensor_shape = Shape{128, 96},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::DRAM,
        },
        // 3D cases - L1 buffer type
        CopyParams{
            .tensor_shape = Shape{8, 64, 64},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
        },
        CopyParams{
            .tensor_shape = Shape{12, 96, 32},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::L1,
        },
        // 3D cases - DRAM buffer type
        CopyParams{
            .tensor_shape = Shape{8, 64, 64},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::DRAM,
        },
        CopyParams{
            .tensor_shape = Shape{12, 96, 32},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::DRAM,
        },
        // 4D cases - L1 buffer type
        CopyParams{
            .tensor_shape = Shape{4, 6, 128},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
        },
        CopyParams{
            .tensor_shape = Shape{256, 64},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::L1,
        },
        // 4D cases - DRAM buffer type
        CopyParams{
            .tensor_shape = Shape{4, 6, 128},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::DRAM,
        },
        CopyParams{
            .tensor_shape = Shape{256, 64},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::DRAM,
        },
        // Higher dimensional cases - L1 buffer type
        CopyParams{
            .tensor_shape = Shape{6, 64, 128},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
        },
        CopyParams{
            .tensor_shape = Shape{3, 2, 2, 3, 4},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
        },
        // Higher dimensional cases - DRAM buffer type
        CopyParams{
            .tensor_shape = Shape{6, 64, 128},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::DRAM,
        },
        CopyParams{
            .tensor_shape = Shape{3, 2, 2, 3, 4},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::DRAM,
        },
    };
}

std::vector<CopyParams> get_sharded_copy_params() {
    return {
        // 2D cases at the beginning
        CopyParams{
            .tensor_shape = Shape{64, 128},
            .layout = Layout::TILE,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{32, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        CopyParams{
            .tensor_shape = Shape{96, 64},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT16,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{24, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
        },
        CopyParams{
            .tensor_shape = Shape{128, 96},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{32, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {2, 2})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        // 3D cases
        CopyParams{
            .tensor_shape = Shape{4, 64, 96},
            .layout = Layout::TILE,
            .dtype = DataType::UINT16,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{1, 32, 64},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        CopyParams{
            .tensor_shape = Shape{1, 2, 32, 32},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{32, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {0, 1})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                    .shard_distribution_strategy = ShardDistributionStrategy::GRID_2D,
                },
        },
        CopyParams{
            .tensor_shape = Shape{2, 64, 128},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{32, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {7, 7})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                    .shard_distribution_strategy = ShardDistributionStrategy::GRID_2D,
                },
        },
        CopyParams{
            .tensor_shape = Shape{18, 128, 64},
            .layout = Layout::TILE,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{1, 64, 96},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 4})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        CopyParams{
            .tensor_shape = Shape{2, 3, 256},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{2, 4, 16},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        // More L1->L1 cases
        CopyParams{
            .tensor_shape = Shape{32, 192},
            .layout = Layout::TILE,
            .dtype = DataType::UINT16,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{32, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 2})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        CopyParams{
            .tensor_shape = Shape{8, 64, 64},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{2, 32, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        CopyParams{
            .tensor_shape = Shape{12, 96, 32},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{3, 32, 16},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
        },
        CopyParams{
            .tensor_shape = Shape{4, 6, 128},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{2, 3, 64},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 0})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        CopyParams{
            .tensor_shape = Shape{256, 64},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{64, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
        },
        // More BFLOAT16 cases
        CopyParams{
            .tensor_shape = Shape{6, 64, 128},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{2, 32, 64},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {2, 1})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        CopyParams{
            .tensor_shape = Shape{3, 2, 2, 3, 4},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{1, 1, 2, 2, 4},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 4})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
        },
        CopyParams{
            .tensor_shape = Shape{5, 2, 2, 64, 96},
            .layout = Layout::TILE,
            .dtype = DataType::UINT16,
            .buffer_type = BufferType::L1,
            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = Shape{1, 1, 2, 64, 64},
                    .grid = CoreRangeSet(
                        tt::stl::Span<const CoreRange>({CoreRange({0, 0}, {2, 0}), CoreRange({0, 1}, {1, 1})})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
        },
    };
}

}  // namespace tensor_accessor_test_params
