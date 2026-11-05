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
    # Integer values span the dtype's whole range, with the sign-bit edge patterns planted at the
    # first three datums: 0x80000000 is the one pattern a sign-magnitude DEST cannot round-trip,
    # and a random draw never produces it.
    if dtype == ttnn.uint32:
        x = torch.randint(0, 2**32, shape, dtype=torch.uint32)
        x.view(-1)[:3] = torch.tensor([2**31, 2**31 - 1, 2**32 - 1], dtype=torch.uint32)
        return x
    if dtype == ttnn.int32:
        x = torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
        x.view(-1)[:3] = torch.tensor([-(2**31), 2**31 - 1, -1], dtype=torch.int32)
        return x
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


# Hand-added: rank > 4 inputs whose height is not a tile multiple. The ND wrapper squeezes them to 4D
# and reshapes the output back; codegen's unpadding path returns a compact tensor (padded == logical)
# where native keeps the tile padding, so the reshape back has to work for both.
_RANK5_UNPADDED_SHAPES = [[1, 2, 8, 1, 768], [1, 2, 8, 16, 768]]


@pytest.mark.parametrize("shape", _RANK5_UNPADDED_SHAPES, ids=["height_1", "height_16"])
def test_untilize_codegen_rank5_unpadded_height(device, shape):
    x = torch.rand(shape, dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG}
    for out in (_force_codegen(xt, **kwargs), ttnn.untilize(xt, **kwargs)):
        assert out.layout == ttnn.ROW_MAJOR_LAYOUT
        assert list(out.shape) == shape
        assert_equal(x, ttnn.to_torch(out))


# Hand-added: 32-bit integer dtypes, the two the codegen gate admits beyond bf16/bf8_b. Values span
# the whole 32-bit range (see _make_input) so that a kernel which quietly lowered them through a
# 16-bit path, or treated uint32 as int32, could not pass. Shapes cover every builder the program
# factory dispatches to, each with its cliff (uneven split) where it has one: single tile,
# per-tile-row split and its cliff, 2D column split, column-parallel single tile-row and its cliff
# (Wt above the core count on every grid), and the with-unpadding writer on width, height, both,
# and a height cliff.
_INT_DTYPES = [ttnn.int32, ttnn.uint32]
_INT_DTYPE_IDS = ["int32", "uint32"]
_INT_SHAPES = [
    ([1, 1, 32, 32], "aligned_single_tile"),
    ([1, 1, 64, 64], "aligned_2x2"),
    ([1, 1, 128, 32], "aligned_tile_rows"),
    ([2, 3, 96, 256], "aligned_2d_column_split"),
    ([1, 1, 32, 1024], "aligned_column_parallel"),
    ([1, 1, 32, 4800], "aligned_column_parallel_cliff"),
    ([1, 1, 7136, 32], "aligned_cliff"),
    ([1, 1, 64, 16], "unpadded_width_16"),
    ([1, 1, 50, 64], "unpadded_height"),
    ([1, 3, 50, 45], "unpadded_both"),
    ([1, 1, 7130, 32], "unpadded_height_cliff"),
    ([1, 2, 8, 1, 768], "rank5_unpadded_height_1"),
    ([1, 2, 8, 16, 768], "rank5_unpadded_height_16"),
]


def _as_int64(tensor):
    return ttnn.to_torch(tensor).to(torch.int64)


@pytest.mark.parametrize("dtype", _INT_DTYPES, ids=_INT_DTYPE_IDS)
@pytest.mark.parametrize("shape", [s for s, _ in _INT_SHAPES], ids=[i for _, i in _INT_SHAPES])
def test_untilize_codegen_int32_uint32(device, shape, dtype):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG}
    expected = x.to(torch.int64)

    golden = _force_native(xt, **kwargs)
    assert_equal(expected, _as_int64(golden))

    # Native is warm, so a program-cache growth on the auto call means auto did not serve native;
    # the forced-codegen call afterwards leaving the cache unchanged pins that program as codegen's.
    entries_before = device.num_program_cache_entries()
    auto = ttnn.untilize(xt, **kwargs)
    entries_after_auto = device.num_program_cache_entries()
    assert entries_after_auto > entries_before, "auto served an in-scope 32-bit case with native"

    forced = _force_codegen(xt, **kwargs)
    assert device.num_program_cache_entries() == entries_after_auto, "auto and forced codegen ran different programs"

    assert forced.layout == ttnn.ROW_MAJOR_LAYOUT
    assert forced.dtype == dtype
    assert list(forced.shape) == shape
    assert forced.padded_shape == golden.padded_shape
    assert_equal(expected, _as_int64(forced))
    assert_equal(expected, _as_int64(auto))


def test_forced_codegen_refuses_float32(device, expect_error):
    # float32 would need the input CB unpacked straight to the 32-bit DEST to be exact, which the
    # codegen program factory does not configure, so it stays out of scope and on native.
    x = torch.rand([1, 1, 64, 64], dtype=torch.float32)
    xt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG}
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.untilize(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    assert device.num_program_cache_entries() == entries_before, "auto routed float32 to codegen"
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, **kwargs)


# Hand-added: the perf demotion (is_demoted). A one-tile-wide int32/uint32 input written to DRAM
# routes to native from 256 tile rows on, however the rows are made up, when the input is in DRAM
# (any arch) or in L1 (blackhole only); one tile row under the bound, with an L1 output, two tiles
# wide, or through the unpadding path on either axis, the same heights stay on codegen. Forced
# codegen still serves the demoted case: only auto's choice changes.
_DEMOTION_CASES = [
    # shape, input placement, output placement, demoted on ("all" | "blackhole" | None), id
    ([1, 1, 8192, 32], ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, "all", "256_tile_rows_dram"),
    ([4, 1, 2048, 32], ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, "all", "256_tile_rows_from_batch_dram"),
    ([1, 1, 8192, 32], ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, "blackhole", "256_tile_rows_l1_input_dram"),
    ([1, 1, 8160, 32], ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, None, "255_tile_rows_dram"),
    ([1, 1, 8192, 32], ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, None, "256_tile_rows_l1"),
    ([1, 1, 8192, 64], ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, None, "two_tiles_wide_dram"),
    ([1, 1, 8190, 32], ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, None, "unpadded_height_dram"),
    ([1, 1, 8192, 16], ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, None, "unpadded_width_dram"),
]


@pytest.mark.parametrize("dtype", _INT_DTYPES, ids=_INT_DTYPE_IDS)
@pytest.mark.parametrize(
    "shape,in_mem,out_mem,demoted_on,case_id", _DEMOTION_CASES, ids=[c[4] for c in _DEMOTION_CASES]
)
def test_untilize_int32_dram_demotion(device, dtype, shape, in_mem, out_mem, demoted_on, case_id):
    demoted = demoted_on == "all" or (demoted_on == "blackhole" and ttnn.device.is_blackhole(device))
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mem)
    kwargs = {"memory_config": out_mem}
    expected = x.to(torch.int64)
    # Primes the native program, so only a route to codegen grows the cache on the auto call.
    _force_native(xt, **kwargs)
    entries_before = device.num_program_cache_entries()
    auto = ttnn.untilize(xt, **kwargs)
    routed_to_codegen = device.num_program_cache_entries() > entries_before
    assert routed_to_codegen != demoted, f"{case_id}: auto {'kept' if demoted else 'demoted'} the case"
    assert_equal(expected, _as_int64(auto))
    assert_equal(expected, _as_int64(_force_codegen(xt, **kwargs)))
