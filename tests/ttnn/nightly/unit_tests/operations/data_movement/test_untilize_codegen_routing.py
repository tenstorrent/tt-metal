# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Routing-fallback coverage: every case the codegen gate rejects must fall back to native.
# The generated block below is emitted from the port's coverage data; hand-add off-grid
# regressions beneath it.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.untilize` takes no implementation argument -- it routes on its own. The forced-native golden
# leg therefore comes from the verification-only entry in the private module; see untilize_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.untilize_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.untilize_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


_DTYPES = [ttnn.bfloat8_b]
_DTYPE_IDS = ["bfloat8_b"]

_ROUTING = [
    # port scope: non-tile-aligned last dims [64, 100] for dtype bfloat8_b (not multiples of 32)
    ([1, 10, 64, 100], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([1, 10, 64, 100], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [95, 74] for dtype bfloat8_b (not multiples of 32)
    ([1, 10, 95, 74], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([1, 10, 95, 74], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [41, 119] for dtype bfloat8_b (not multiples of 32)
    ([1, 12, 41, 119], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([1, 12, 41, 119], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [188, 200] for dtype bfloat8_b (not multiples of 32)
    ([1, 188, 200], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([1, 188, 200], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [104, 60] for dtype bfloat8_b (not multiples of 32)
    ([1, 4, 104, 60], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([1, 4, 104, 60], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [102, 165] for dtype bfloat8_b (not multiples of 32)
    ([102, 165], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([102, 165], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [63, 172] for dtype bfloat8_b (not multiples of 32)
    ([12, 63, 172], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([12, 63, 172], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [132, 246] for dtype bfloat8_b (not multiples of 32)
    ([132, 246], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([132, 246], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [147, 158] for dtype bfloat8_b (not multiples of 32)
    ([147, 158], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([147, 158], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [170, 206] for dtype bfloat8_b (not multiples of 32)
    ([170, 206], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([170, 206], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [180, 102] for dtype bfloat8_b (not multiples of 32)
    ([180, 102], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([180, 102], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [113, 256] for dtype bfloat8_b (not multiples of 32)
    ([2, 113, 256], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([2, 113, 256], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [73, 122] for dtype bfloat8_b (not multiples of 32)
    ([2, 12, 73, 122], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([2, 12, 73, 122], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [205, 225] for dtype bfloat8_b (not multiples of 32)
    ([2, 205, 225], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([2, 205, 225], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [201, 196] for dtype bfloat8_b (not multiples of 32)
    ([201, 196], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([201, 196], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [205, 182] for dtype bfloat8_b (not multiples of 32)
    ([205, 182], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([205, 182], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [211, 235] for dtype bfloat8_b (not multiples of 32)
    ([211, 235], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([211, 235], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [211, 244] for dtype bfloat8_b (not multiples of 32)
    ([211, 244], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([211, 244], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [212, 166] for dtype bfloat8_b (not multiples of 32)
    ([212, 166], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([212, 166], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [239, 92] for dtype bfloat8_b (not multiples of 32)
    ([239, 92], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([239, 92], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [243, 139] for dtype bfloat8_b (not multiples of 32)
    ([243, 139], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([243, 139], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [247, 52] for dtype bfloat8_b (not multiples of 32)
    ([247, 52], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([247, 52], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [249, 87] for dtype bfloat8_b (not multiples of 32)
    ([249, 87], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([249, 87], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [72, 110] for dtype bfloat8_b (not multiples of 32)
    ([3, 7, 72, 110], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([3, 7, 72, 110], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [103, 44] for dtype bfloat8_b (not multiples of 32)
    ([3, 8, 103, 44], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([3, 8, 103, 44], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [106, 79] for dtype bfloat8_b (not multiples of 32)
    ([4, 106, 79], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([4, 106, 79], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [117, 112] for dtype bfloat8_b (not multiples of 32)
    ([4, 12, 117, 112], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([4, 12, 117, 112], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [236, 186] for dtype bfloat8_b (not multiples of 32)
    ([4, 236, 186], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([4, 236, 186], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [237, 68] for dtype bfloat8_b (not multiples of 32)
    ([4, 237, 68], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([4, 237, 68], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [242, 79] for dtype bfloat8_b (not multiples of 32)
    ([4, 242, 79], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([4, 242, 79], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [106, 59] for dtype bfloat8_b (not multiples of 32)
    ([4, 6, 106, 59], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([4, 6, 106, 59], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [37, 65] for dtype bfloat8_b (not multiples of 32)
    ([4, 7, 37, 65], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([4, 7, 37, 65], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [65, 39] for dtype bfloat8_b (not multiples of 32)
    ([4, 9, 65, 39], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([4, 9, 65, 39], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [43, 124] for dtype bfloat8_b (not multiples of 32)
    ([5, 1, 43, 124], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([5, 1, 43, 124], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [153, 49] for dtype bfloat8_b (not multiples of 32)
    ([5, 153, 49], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([5, 153, 49], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [173, 106] for dtype bfloat8_b (not multiples of 32)
    ([5, 173, 106], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([5, 173, 106], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [68, 49] for dtype bfloat8_b (not multiples of 32)
    ([5, 3, 68, 49], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([5, 3, 68, 49], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [83, 70] for dtype bfloat8_b (not multiples of 32)
    ([5, 8, 83, 70], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([5, 8, 83, 70], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [50, 71] for dtype bfloat8_b (not multiples of 32)
    ([6, 10, 50, 71], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([6, 10, 50, 71], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [240, 170] for dtype bfloat8_b (not multiples of 32)
    ([6, 240, 170], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([6, 240, 170], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [102, 93] for dtype bfloat8_b (not multiples of 32)
    ([6, 4, 102, 93], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([6, 4, 102, 93], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [65, 256] for dtype bfloat8_b (not multiples of 32)
    ([65, 256], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([65, 256], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [113, 179] for dtype bfloat8_b (not multiples of 32)
    ([7, 113, 179], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([7, 113, 179], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [70, 41] for dtype bfloat8_b (not multiples of 32)
    ([70, 41], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([70, 41], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [55, 184] for dtype bfloat8_b (not multiples of 32)
    ([8, 55, 184], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([8, 55, 184], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [146, 55] for dtype bfloat8_b (not multiples of 32)
    ([9, 146, 55], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([9, 146, 55], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [157, 59] for dtype bfloat8_b (not multiples of 32)
    ([9, 157, 59], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([9, 157, 59], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    # port scope: non-tile-aligned last dims [182, 105] for dtype bfloat8_b (not multiples of 32)
    ([9, 182, 105], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
    ([9, 182, 105], {"memory_config": ttnn.L1_MEMORY_CONFIG}, ttnn.TILE_LAYOUT),
]
_ROUTING_IDS = [
    # port scope: non-tile-aligned last dims [64, 100] for dtype bfloat8_b (not multiples of 32)
    "[1, 10, 64, 100]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[1, 10, 64, 100]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [95, 74] for dtype bfloat8_b (not multiples of 32)
    "[1, 10, 95, 74]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[1, 10, 95, 74]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [41, 119] for dtype bfloat8_b (not multiples of 32)
    "[1, 12, 41, 119]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[1, 12, 41, 119]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [188, 200] for dtype bfloat8_b (not multiples of 32)
    "[1, 188, 200]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[1, 188, 200]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [104, 60] for dtype bfloat8_b (not multiples of 32)
    "[1, 4, 104, 60]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[1, 4, 104, 60]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [102, 165] for dtype bfloat8_b (not multiples of 32)
    "[102, 165]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[102, 165]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [63, 172] for dtype bfloat8_b (not multiples of 32)
    "[12, 63, 172]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[12, 63, 172]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [132, 246] for dtype bfloat8_b (not multiples of 32)
    "[132, 246]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[132, 246]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [147, 158] for dtype bfloat8_b (not multiples of 32)
    "[147, 158]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[147, 158]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [170, 206] for dtype bfloat8_b (not multiples of 32)
    "[170, 206]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[170, 206]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [180, 102] for dtype bfloat8_b (not multiples of 32)
    "[180, 102]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[180, 102]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [113, 256] for dtype bfloat8_b (not multiples of 32)
    "[2, 113, 256]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[2, 113, 256]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [73, 122] for dtype bfloat8_b (not multiples of 32)
    "[2, 12, 73, 122]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[2, 12, 73, 122]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [205, 225] for dtype bfloat8_b (not multiples of 32)
    "[2, 205, 225]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[2, 205, 225]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [201, 196] for dtype bfloat8_b (not multiples of 32)
    "[201, 196]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[201, 196]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [205, 182] for dtype bfloat8_b (not multiples of 32)
    "[205, 182]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[205, 182]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [211, 235] for dtype bfloat8_b (not multiples of 32)
    "[211, 235]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[211, 235]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [211, 244] for dtype bfloat8_b (not multiples of 32)
    "[211, 244]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[211, 244]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [212, 166] for dtype bfloat8_b (not multiples of 32)
    "[212, 166]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[212, 166]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [239, 92] for dtype bfloat8_b (not multiples of 32)
    "[239, 92]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[239, 92]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [243, 139] for dtype bfloat8_b (not multiples of 32)
    "[243, 139]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[243, 139]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [247, 52] for dtype bfloat8_b (not multiples of 32)
    "[247, 52]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[247, 52]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [249, 87] for dtype bfloat8_b (not multiples of 32)
    "[249, 87]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[249, 87]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [72, 110] for dtype bfloat8_b (not multiples of 32)
    "[3, 7, 72, 110]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[3, 7, 72, 110]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [103, 44] for dtype bfloat8_b (not multiples of 32)
    "[3, 8, 103, 44]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[3, 8, 103, 44]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [106, 79] for dtype bfloat8_b (not multiples of 32)
    "[4, 106, 79]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[4, 106, 79]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [117, 112] for dtype bfloat8_b (not multiples of 32)
    "[4, 12, 117, 112]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[4, 12, 117, 112]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [236, 186] for dtype bfloat8_b (not multiples of 32)
    "[4, 236, 186]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[4, 236, 186]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [237, 68] for dtype bfloat8_b (not multiples of 32)
    "[4, 237, 68]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[4, 237, 68]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [242, 79] for dtype bfloat8_b (not multiples of 32)
    "[4, 242, 79]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[4, 242, 79]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [106, 59] for dtype bfloat8_b (not multiples of 32)
    "[4, 6, 106, 59]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[4, 6, 106, 59]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [37, 65] for dtype bfloat8_b (not multiples of 32)
    "[4, 7, 37, 65]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[4, 7, 37, 65]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [65, 39] for dtype bfloat8_b (not multiples of 32)
    "[4, 9, 65, 39]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[4, 9, 65, 39]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [43, 124] for dtype bfloat8_b (not multiples of 32)
    "[5, 1, 43, 124]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[5, 1, 43, 124]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [153, 49] for dtype bfloat8_b (not multiples of 32)
    "[5, 153, 49]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[5, 153, 49]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [173, 106] for dtype bfloat8_b (not multiples of 32)
    "[5, 173, 106]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[5, 173, 106]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [68, 49] for dtype bfloat8_b (not multiples of 32)
    "[5, 3, 68, 49]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[5, 3, 68, 49]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [83, 70] for dtype bfloat8_b (not multiples of 32)
    "[5, 8, 83, 70]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[5, 8, 83, 70]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [50, 71] for dtype bfloat8_b (not multiples of 32)
    "[6, 10, 50, 71]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[6, 10, 50, 71]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [240, 170] for dtype bfloat8_b (not multiples of 32)
    "[6, 240, 170]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[6, 240, 170]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [102, 93] for dtype bfloat8_b (not multiples of 32)
    "[6, 4, 102, 93]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[6, 4, 102, 93]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [65, 256] for dtype bfloat8_b (not multiples of 32)
    "[65, 256]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[65, 256]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [113, 179] for dtype bfloat8_b (not multiples of 32)
    "[7, 113, 179]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[7, 113, 179]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [70, 41] for dtype bfloat8_b (not multiples of 32)
    "[70, 41]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[70, 41]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [55, 184] for dtype bfloat8_b (not multiples of 32)
    "[8, 55, 184]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[8, 55, 184]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [146, 55] for dtype bfloat8_b (not multiples of 32)
    "[9, 146, 55]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[9, 146, 55]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [157, 59] for dtype bfloat8_b (not multiples of 32)
    "[9, 157, 59]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[9, 157, 59]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    # port scope: non-tile-aligned last dims [182, 105] for dtype bfloat8_b (not multiples of 32)
    "[9, 182, 105]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
    "[9, 182, 105]|memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)|tile",
]


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("shape,kwargs,layout", _ROUTING, ids=_ROUTING_IDS)
def test_untilize_codegen_routing(device, shape, kwargs, dtype, layout):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    # The golden call warms the native program, so a correct fallback leaves the cache
    # unchanged; only a mis-route to codegen compiles a new program and grows it.
    entries_before = device.num_program_cache_entries()
    out = ttnn.untilize(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed an out-of-scope case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


# Hand-added: execution-control routing. Every codegen factory places work over the full
# compute-with-storage grid, so a caller that asked for single-core placement or a specific
# sub-grid must reach native under `auto` -- silently widening the placement would break the
# resource partitioning the caller asked for.
_SUB_CORE_GRIDS = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6))])

# Each control carries its own shape because this leg has to run native to have a golden: native's
# sub-core-grid factory only accepts tensors one tile row tall (UntilizeDeviceOperation validate).
# Both shapes are in scope for the codegen gate, so only the execution control can drive the route.
_NATIVE_GOLDEN_CONTROLS = [
    ({"use_multicore": False}, [64, 128], "use_multicore_false"),
    ({"sub_core_grids": _SUB_CORE_GRIDS}, [32, 128], "sub_core_grids"),
]


@pytest.mark.parametrize(
    "controls,shape,control_id", _NATIVE_GOLDEN_CONTROLS, ids=[c[2] for c in _NATIVE_GOLDEN_CONTROLS]
)
def test_untilize_execution_controls_route_to_native(device, controls, shape, control_id):
    x = torch.rand(shape, dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG, **controls}
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.untilize(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = f"auto routed {control_id} to codegen (program cache grew); codegen cannot honour it"
    assert device.num_program_cache_entries() == entries_before, msg


def test_forced_codegen_refuses_out_of_scope_case(device, expect_error):
    # The forced leg exists to be compared against native, so it has to fail loudly outside its
    # support scope: if it fell back, every bit-exactness result gathered through it would really be
    # native-vs-native. bfloat8_b needs a tile-aligned logical shape, and [64, 100] is not.
    x = torch.rand([1, 10, 64, 100], dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# Hand-added: tile-geometry routing, over both axes a tile varies on. An off-default *shape* changes
# the page count and the page size the codegen plan builds from the 32x32 constants. A transposed
# 32x32 leaves both alone -- so no page-geometry check can see it -- but it permutes the datums the
# untilize kernels lift out of each face, and nothing configures the unpacker for that. Both have to
# reach native. The shape is deliberately in scope on every other axis (bfloat16, interleaved,
# tile-aligned at 32), so only the tile drives the route.
#
# Route only, no value assertion: native does not serve either tile correctly -- two native calls on
# the same input disagree with each other, and both differ from torch. What this port owes is that it
# declines the case instead of answering it wrongly in its own way. Native's answer is not a
# reference here.
_OFF_DEFAULT_TILES = [ttnn.Tile([16, 16]), ttnn.Tile([32, 32], transpose_tile=True)]
_OFF_DEFAULT_TILE_IDS = ["shape_16x16", "transposed_32x32"]


@pytest.mark.parametrize("tile", _OFF_DEFAULT_TILES, ids=_OFF_DEFAULT_TILE_IDS)
def test_untilize_non_default_tile_routes_to_native(device, tile):
    shape = [64, 128]
    x = torch.rand(shape, dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=tile)
    kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG}
    # Primes the cache with the native program, so an unchanged count means native served the call.
    _force_native(xt, **kwargs)
    entries_before = device.num_program_cache_entries()
    ttnn.untilize(xt, **kwargs)
    msg = "auto routed a non-default tile to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


@pytest.mark.parametrize("tile", _OFF_DEFAULT_TILES, ids=_OFF_DEFAULT_TILE_IDS)
def test_forced_codegen_refuses_non_default_tile(device, expect_error, tile):
    x = torch.rand([64, 128], dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=tile)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
