# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Contracts: (1) every case the codegen gate rejects falls back to native under ordinary
# routing; (2) an accepted case is actually routed to codegen by ordinary routing; (3) an accepted
# case dispatched twice under forced codegen stays a program-cache hit and rebinds its buffers;
# (4) the forced-codegen entry refuses an out-of-scope case rather than falling back. The block
# below is generated from the op's coverage data; hand-add off-grid regressions beneath it.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.concat` takes no implementation argument -- it routes on its own. The forced legs therefore
# come from the verification-only entries in the private module; see concat_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.concat_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.concat_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


def _inputs(shapes, dtype, layout, device):
    xs = [ttnn.from_torch(_make_input(s, dtype), dtype=dtype, layout=layout, device=device) for s in shapes]
    return xs


_ROUTING = [
    ([[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]], {"dim": -1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]], {"dim": -1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]], {"dim": -1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]], {"dim": -1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]], {"dim": -1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]], {"dim": -1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]], {"dim": -1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]], {"dim": -1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]], {"dim": -1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]], {"dim": -1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]], {"dim": -1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]], {"dim": -1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    (
        [[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]],
        {"dim": -1},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
    ),
    (
        [[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]],
        {"dim": -1},
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
    ),
    (
        [[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]],
        {"dim": -1},
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
    ),
    (
        [[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]],
        {"dim": -1},
        ttnn.float32,
        ttnn.TILE_LAYOUT,
    ),
    (
        [[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]],
        {"dim": -1},
        ttnn.int32,
        ttnn.TILE_LAYOUT,
    ),
    (
        [[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]],
        {"dim": -1},
        ttnn.uint32,
        ttnn.TILE_LAYOUT,
    ),
    ([[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]], {"dim": -1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]], {"dim": -1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]], {"dim": -1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]], {"dim": -1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]], {"dim": -1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]], {"dim": -1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]], {"dim": -1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]], {"dim": -1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]], {"dim": -1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]], {"dim": -1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]], {"dim": -1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]], {"dim": -1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]], {"dim": -1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]], {"dim": -1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]], {"dim": -1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]], {"dim": -1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]], {"dim": -1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]], {"dim": -1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]], {"dim": 0}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]], {"dim": 0}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]], {"dim": 0}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]], {"dim": 0}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]], {"dim": 0}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]], {"dim": 0}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[1, 2, 32, 64], [1, 3, 32, 64]], {"dim": 1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 2, 32, 64], [1, 3, 32, 64]], {"dim": 1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 2, 32, 64], [1, 3, 32, 64]], {"dim": 1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 2, 32, 64], [1, 3, 32, 64]], {"dim": 1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 2, 32, 64], [1, 3, 32, 64]], {"dim": 1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 2, 32, 64], [1, 3, 32, 64]], {"dim": 1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[1, 32, 64], [1, 32, 64], [2, 32, 64]], {"dim": 0}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 32, 64], [1, 32, 64], [2, 32, 64]], {"dim": 0}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[1, 32, 64], [1, 32, 64], [2, 32, 64]], {"dim": 0}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 32, 64], [1, 32, 64], [2, 32, 64]], {"dim": 0}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[1, 32, 64], [1, 32, 64], [2, 32, 64]], {"dim": 0}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[1, 32, 64], [1, 32, 64], [2, 32, 64]], {"dim": 0}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[2, 32, 64], [2, 64, 64]], {"dim": 1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[2, 32, 64], [2, 64, 64]], {"dim": 1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[2, 32, 64], [2, 64, 64]], {"dim": 1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[2, 32, 64], [2, 64, 64]], {"dim": 1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[2, 32, 64], [2, 64, 64]], {"dim": 1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[2, 32, 64], [2, 64, 64]], {"dim": 1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 32], [32, 32], [32, 32]], {"dim": 0}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 32], [32, 32], [32, 32]], {"dim": 0}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 32], [32, 32], [32, 32]], {"dim": 0}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[32, 32], [32, 32], [32, 32], [32, 32]], {"dim": 0}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 32], [32, 32], [32, 32]], {"dim": 0}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 32], [32, 32], [32, 32]], {"dim": 0}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 64], [32, 96]], {"dim": -1}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 64], [32, 96]], {"dim": -1}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 64], [32, 96]], {"dim": -1}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[32, 32], [32, 64], [32, 96]], {"dim": -1}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 64], [32, 96]], {"dim": -1}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[32, 32], [32, 64], [32, 96]], {"dim": -1}, ttnn.uint32, ttnn.TILE_LAYOUT),
    ([[32, 64], [32, 64], [32, 64]], {"dim": 0}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[32, 64], [32, 64], [32, 64]], {"dim": 0}, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ([[32, 64], [32, 64], [32, 64]], {"dim": 0}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[32, 64], [32, 64], [32, 64]], {"dim": 0}, ttnn.float32, ttnn.TILE_LAYOUT),
    ([[32, 64], [32, 64], [32, 64]], {"dim": 0}, ttnn.int32, ttnn.TILE_LAYOUT),
    ([[32, 64], [32, 64], [32, 64]], {"dim": 0}, ttnn.uint32, ttnn.TILE_LAYOUT),
]
_ROUTING_IDS = [
    "[[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]]|dim=-1|bfloat16|tile",
    "[[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]]|dim=-1|bfloat8_b|tile",
    "[[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]]|dim=-1|float32|row_major",
    "[[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]]|dim=-1|float32|tile",
    "[[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]]|dim=-1|int32|tile",
    "[[1, 1, 1024, 1024], [1, 1, 1024, 1024], [1, 1, 1024, 1024]]|dim=-1|uint32|tile",
    "[[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]]|dim=-1|bfloat16|tile",
    "[[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]]|dim=-1|bfloat8_b|tile",
    "[[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]]|dim=-1|float32|row_major",
    "[[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]]|dim=-1|float32|tile",
    "[[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]]|dim=-1|int32|tile",
    "[[1, 1, 128, 128], [1, 1, 128, 128], [1, 1, 128, 128]]|dim=-1|uint32|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|bfloat16|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|bfloat8_b|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|float32|row_major",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|float32|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|int32|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|uint32|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|bfloat16|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|bfloat8_b|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|float32|row_major",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|float32|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|int32|tile",
    "[[1, 1, 256, 256], [1, 1, 256, 256], [1, 1, 256, 256]]|dim=-1|uint32|tile",
    "[[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]]|dim=-1|bfloat16|tile",
    "[[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]]|dim=-1|bfloat8_b|tile",
    "[[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]]|dim=-1|float32|row_major",
    "[[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]]|dim=-1|float32|tile",
    "[[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]]|dim=-1|int32|tile",
    "[[1, 1, 512, 512], [1, 1, 512, 512], [1, 1, 512, 512]]|dim=-1|uint32|tile",
    "[[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]]|dim=-1|bfloat16|tile",
    "[[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]]|dim=-1|bfloat8_b|tile",
    "[[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]]|dim=-1|float32|row_major",
    "[[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]]|dim=-1|float32|tile",
    "[[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]]|dim=-1|int32|tile",
    "[[1, 1, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]]|dim=-1|uint32|tile",
    "[[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]]|dim=0|bfloat16|tile",
    "[[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]]|dim=0|bfloat8_b|tile",
    "[[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]]|dim=0|float32|row_major",
    "[[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]]|dim=0|float32|tile",
    "[[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]]|dim=0|int32|tile",
    "[[1, 1, 64, 64], [2, 1, 64, 64], [1, 1, 64, 64]]|dim=0|uint32|tile",
    "[[1, 2, 32, 64], [1, 3, 32, 64]]|dim=1|bfloat16|tile",
    "[[1, 2, 32, 64], [1, 3, 32, 64]]|dim=1|bfloat8_b|tile",
    "[[1, 2, 32, 64], [1, 3, 32, 64]]|dim=1|float32|row_major",
    "[[1, 2, 32, 64], [1, 3, 32, 64]]|dim=1|float32|tile",
    "[[1, 2, 32, 64], [1, 3, 32, 64]]|dim=1|int32|tile",
    "[[1, 2, 32, 64], [1, 3, 32, 64]]|dim=1|uint32|tile",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2|bfloat16|tile",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2|bfloat8_b|tile",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2|float32|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2|float32|tile",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2|int32|tile",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2|uint32|tile",
    "[[1, 32, 64], [1, 32, 64], [2, 32, 64]]|dim=0|bfloat16|tile",
    "[[1, 32, 64], [1, 32, 64], [2, 32, 64]]|dim=0|bfloat8_b|tile",
    "[[1, 32, 64], [1, 32, 64], [2, 32, 64]]|dim=0|float32|row_major",
    "[[1, 32, 64], [1, 32, 64], [2, 32, 64]]|dim=0|float32|tile",
    "[[1, 32, 64], [1, 32, 64], [2, 32, 64]]|dim=0|int32|tile",
    "[[1, 32, 64], [1, 32, 64], [2, 32, 64]]|dim=0|uint32|tile",
    "[[2, 32, 64], [2, 64, 64]]|dim=1|bfloat16|tile",
    "[[2, 32, 64], [2, 64, 64]]|dim=1|bfloat8_b|tile",
    "[[2, 32, 64], [2, 64, 64]]|dim=1|float32|row_major",
    "[[2, 32, 64], [2, 64, 64]]|dim=1|float32|tile",
    "[[2, 32, 64], [2, 64, 64]]|dim=1|int32|tile",
    "[[2, 32, 64], [2, 64, 64]]|dim=1|uint32|tile",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]|dim=0|bfloat16|tile",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]|dim=0|bfloat8_b|tile",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]|dim=0|float32|row_major",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]|dim=0|float32|tile",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]|dim=0|int32|tile",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]|dim=0|uint32|tile",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1|bfloat16|tile",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1|bfloat8_b|tile",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1|float32|row_major",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1|float32|tile",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1|int32|tile",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1|uint32|tile",
    "[[32, 64], [32, 64], [32, 64]]|dim=0|bfloat16|tile",
    "[[32, 64], [32, 64], [32, 64]]|dim=0|bfloat8_b|tile",
    "[[32, 64], [32, 64], [32, 64]]|dim=0|float32|row_major",
    "[[32, 64], [32, 64], [32, 64]]|dim=0|float32|tile",
    "[[32, 64], [32, 64], [32, 64]]|dim=0|int32|tile",
    "[[32, 64], [32, 64], [32, 64]]|dim=0|uint32|tile",
]


_ACCEPTED = [
    ([[1, 32, 64], [1, 32, 64]], {"dim": 0}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ([[2, 32, 64], [2, 64, 64]], {"dim": 1}, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 32, 64], [1, 32, 64]], {"dim": -1}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 32, 32], [1, 64, 32], [1, 32, 32]], {"dim": 1}, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": -1}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
]
_ACCEPTED_IDS = [
    "[[1, 32, 64], [1, 32, 64]]|dim=0|bfloat16|row_major",
    "[[2, 32, 64], [2, 64, 64]]|dim=1|int32|row_major",
    "[[1, 32, 64], [1, 32, 64]]|dim=-1|bfloat16|row_major",
    "[[1, 32, 32], [1, 64, 32], [1, 32, 32]]|dim=1|uint32|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=-1|bfloat16|row_major",
]


@pytest.mark.parametrize("shapes,kwargs,dtype,layout", _ACCEPTED, ids=_ACCEPTED_IDS)
def test_concat_routes_an_accepted_case_to_codegen(device, shapes, kwargs, dtype, layout):
    # The negative direction alone is satisfied by a gate that accepts nothing: if the predicate
    # started returning false everywhere, or a demotion grew to cover the supported set, every
    # other test here would still pass while the auto route was dead.
    #
    # The forced entry and the auto branch end in the same prim::concat_codegen call on the same
    # params, so they share a program-cache key. Warming with the forced entry turns "did ordinary
    # routing pick codegen" into "did the cache stay put" -- a native decision would have to build
    # its own program.
    xs = _inputs(shapes, dtype, layout, device)
    golden = ttnn.to_torch(_force_codegen(xs, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.concat(xs, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "routed an in-scope case to native (program cache grew); expected the codegen program"
    assert device.num_program_cache_entries() == entries_before, msg
    # "The count held" only means "codegen ran" while native has no program of its own for this
    # spec; the fixture shares one device and never clears the cache, so an earlier test holding
    # that key would make the assertion above pass on a native route. Force native last and
    # require it to build: if it were already resident, nothing above proved anything.
    assert_equal(golden, ttnn.to_torch(_force_native(xs, **kwargs)))
    vacuous = "native was already cached for this spec, so the assertion above could not see a native route"
    assert device.num_program_cache_entries() > entries_before, vacuous


@pytest.mark.parametrize("shapes,kwargs,dtype,layout", _ROUTING, ids=_ROUTING_IDS)
def test_concat_codegen_routing(device, shapes, kwargs, dtype, layout):
    xs = _inputs(shapes, dtype, layout, device)
    golden = ttnn.to_torch(_force_native(xs, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.concat(xs, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "routed an out-of-scope case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


_CACHE_HIT = [
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
]
_CACHE_HIT_IDS = [
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2|bfloat16|row_major",
]


@pytest.mark.parametrize("shapes,kwargs,dtype,layout", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_concat_codegen_program_cache_hit(device, shapes, kwargs, dtype, layout):
    xs = _inputs(shapes, dtype, layout, device)
    golden = ttnn.to_torch(_force_native(xs, **kwargs))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xs, **kwargs)))
    entries_after_miss = device.num_program_cache_entries()
    # Same spec, a distinct allocation: the cached program must rebind its Buffer*s
    # instead of reusing the first dispatch's addresses.
    ys = _inputs(shapes, dtype, layout, device)
    second_golden = ttnn.to_torch(_force_native(ys, **kwargs))
    assert_equal(second_golden, ttnn.to_torch(_force_codegen(ys, **kwargs)))
    msg = "second forced-codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_after_miss, msg


# One case per rejection clause the gate owns, so a forced leg cannot quietly serve native.
_OUT_OF_SCOPE = [
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([[1, 32, 32]], {"dim": 2}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
]
_OUT_OF_SCOPE_IDS = ["tile_layout", "unsupported_dtype", "single_input"]


@pytest.mark.parametrize("shapes,kwargs,dtype,layout", _OUT_OF_SCOPE, ids=_OUT_OF_SCOPE_IDS)
def test_forced_codegen_refuses_out_of_scope_case(device, expect_error, shapes, kwargs, dtype, layout):
    xs = _inputs(shapes, dtype, layout, device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xs, **kwargs)


# --- hand-added below the generated block ---

# The factory dispatches on input count and on whether the concat dim is the last one, and each of
# the four builders assembles its own runtime-argument list with its own buffer bindings. The
# generated leg above parametrizes over layout, which for this op is one value, so it reaches only
# the N-way width builder; a stale-address regression in any of the other three would return the
# first allocation's data on a cache hit with nothing to catch it. Those three are the cases here.
_CACHE_HIT_BRANCHES = [
    ([[1, 32, 32], [1, 64, 32]], {"dim": 1}, "two_input_nonwidth"),
    ([[1, 32, 32], [1, 32, 64]], {"dim": 2}, "two_input_width"),
    ([[1, 32, 32], [1, 64, 32], [1, 32, 32]], {"dim": 1}, "nway_nonwidth"),
]


@pytest.mark.parametrize(
    "shapes,kwargs", [(c[0], c[1]) for c in _CACHE_HIT_BRANCHES], ids=[c[2] for c in _CACHE_HIT_BRANCHES]
)
def test_concat_codegen_program_cache_hit_every_builder(device, shapes, kwargs):
    dtype, layout = ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
    xs = _inputs(shapes, dtype, layout, device)
    golden = ttnn.to_torch(_force_native(xs, **kwargs))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xs, **kwargs)))
    entries_after_miss = device.num_program_cache_entries()
    ys = _inputs(shapes, dtype, layout, device)
    second_golden = ttnn.to_torch(_force_native(ys, **kwargs))
    assert_equal(second_golden, ttnn.to_torch(_force_codegen(ys, **kwargs)))
    msg = "second forced-codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_after_miss, msg


def test_concat_codegen_declines_host_tensors(expect_error):
    # The gate reads page sizes and alignments off buffers a host tensor does not own, so it has to
    # decline before asking anything else -- otherwise routing dereferences null ahead of the
    # validation that would have named the real problem.
    xs = [ttnn.from_torch(torch.rand(1, 32, 32, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT) for _ in range(2)]
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xs, dim=2)


def test_concat_codegen_declines_zero_volume(device, expect_error):
    # No builder has a zero-work path, and a zero-width output makes the stick count a division by
    # zero, so these shapes have to stay on native.
    xs = _inputs([[1, 32, 0], [1, 32, 0]], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xs, dim=2)


def _l1_sharded_config(shard_shape):
    core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    spec = ttnn.ShardSpec(core, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def test_concat_codegen_declines_above_input_ceiling(device, expect_error):
    # The N-way readers hold 17 bytes of dataflow-RISC frame per input against a 256 B guaranteed
    # stack, so the ceiling is a memory-safety bound, not a runtime-argument one.
    n = ttnn._ttnn.operations.data_movement.CONCAT_MAX_NWAY_INPUTS + 1
    xs = _inputs([[1, 32, 32]] * n, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xs, dim=2)


def test_concat_codegen_declines_sharded_input(device, expect_error):
    # Every builder addresses its inputs as interleaved pages through TensorAccessorArgs.
    sharded = _l1_sharded_config([32, 32])
    xs = [
        ttnn.from_torch(
            torch.rand(1, 32, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=sharded,
        )
        for _ in range(2)
    ]
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xs, dim=2)


def test_concat_codegen_declines_sharded_output(device, expect_error):
    xs = _inputs([[1, 32, 32], [1, 32, 32]], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xs, dim=2, memory_config=_l1_sharded_config([32, 64]))


def test_concat_codegen_declines_mixed_memory_config_above_two_inputs(device, expect_error):
    # The N-way readers share one TensorAccessorArgs ABI across all inputs, so a DRAM input and an
    # L1 input in the same list would be addressed with the wrong bank geometry.
    def make(mem):
        return ttnn.from_torch(
            torch.rand(1, 32, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=mem,
        )

    xs = [make(ttnn.DRAM_MEMORY_CONFIG), make(ttnn.L1_MEMORY_CONFIG), make(ttnn.DRAM_MEMORY_CONFIG)]
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xs, dim=2)


def test_concat_routes_two_input_mixed_placement_staged_copy_to_codegen(device):
    # Only N > 2 is held to one memory config, so a two-input list can mix them -- and then input
    # 1's destination offset, which is input 0's stick size, has to clear input 1's own transport
    # alignment rather than input 0's. A 16 B L1 stick followed by a 64 B DRAM stick fills both
    # physical pages yet is not a legal DRAM endpoint, so input 1 stages through scratch. Native
    # reads that same offset with input 1's transport and returns shifted data, so this class is
    # held on codegen at every size instead of being demoted on staged volume, and torch is the
    # only usable reference for it.
    def make(width, mem):
        host = (torch.arange(8192 * width, dtype=torch.float32).reshape(1, 8192, width) / 64.0).to(torch.bfloat16)
        return host, ttnn.from_torch(
            host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem
        )

    h0, x0 = make(8, ttnn.L1_MEMORY_CONFIG)
    h1, x1 = make(32, ttnn.DRAM_MEMORY_CONFIG)
    xs = [x0, x1]
    want = torch.cat([h0, h1], dim=2)
    assert_equal(want, ttnn.to_torch(_force_codegen(xs, dim=2)))
    entries_before = device.num_program_cache_entries()
    out = ttnn.concat(xs, dim=2)
    assert_equal(want, ttnn.to_torch(out))
    msg = "routed a mixed-placement staged-copy case to native (program cache grew); native misreads it"
    assert device.num_program_cache_entries() == entries_before, msg


def test_concat_codegen_declines_mismatched_dtype(device, expect_error):
    xs = [
        ttnn.from_torch(
            torch.rand(1, 32, 32, dtype=torch.bfloat16), dtype=d, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        if d is ttnn.bfloat16
        else ttnn.from_torch(
            torch.randint(0, 100, (1, 32, 32), dtype=torch.int32), dtype=d, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        for d in (ttnn.bfloat16, ttnn.int32)
    ]
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xs, dim=2)


# Every CB in every plan is sized from the output's page, so an L1-resident output is the case
# where the budget the plan is measured against and the memory the output itself takes are the
# same L1. Nothing in this file passed a memory_config before, leaving that whole path unrun.
_L1_OUTPUT_BRANCHES = [
    ([[1, 32, 32], [1, 64, 32]], {"dim": 1}, "two_input_nonwidth"),
    ([[1, 32, 32], [1, 32, 64]], {"dim": 2}, "two_input_width"),
    ([[1, 32, 32], [1, 64, 32], [1, 32, 32]], {"dim": 1}, "nway_nonwidth"),
    ([[1, 32, 32], [1, 32, 64], [1, 32, 32]], {"dim": 2}, "nway_width"),
]


@pytest.mark.parametrize(
    "shapes,kwargs", [(c[0], c[1]) for c in _L1_OUTPUT_BRANCHES], ids=[c[2] for c in _L1_OUTPUT_BRANCHES]
)
def test_concat_codegen_l1_interleaved_output(device, shapes, kwargs):
    xs = _inputs(shapes, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    golden = ttnn.to_torch(_force_native(xs, **kwargs, memory_config=ttnn.L1_MEMORY_CONFIG))
    out = _force_codegen(xs, **kwargs, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    assert_equal(golden, ttnn.to_torch(out))


def test_concat_codegen_declines_execution_controls(device):
    # No builder honours sub_core_grids -- every one of them places work over the full
    # compute_with_storage_grid_size -- so a caller asking for a core subset must reach native.
    # Shapes no other case here uses: codegen carries no core-grid field, so a route that dropped
    # sub_core_grids would land on the plain (shapes, dim) codegen key. Were that key already
    # resident from another test, the dropped control would produce the same values against an
    # unchanged count and pass.
    shapes, dim = [[1, 32, 48], [1, 32, 80]], 2
    xs = _inputs(shapes, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    golden = ttnn.to_torch(_force_native(xs, dim=dim, sub_core_grids=grid))
    entries_before = device.num_program_cache_entries()
    out = ttnn.concat(xs, dim=dim, sub_core_grids=grid)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "routed a sub_core_grids request to codegen, which ignores it"
    assert device.num_program_cache_entries() == entries_before, msg
    assert_equal(golden, ttnn.to_torch(_force_codegen(xs, dim=dim)))
    vacuous = "codegen was already cached for this spec, so the assertion above could not see a codegen route"
    assert device.num_program_cache_entries() > entries_before, vacuous


def test_concat_codegen_replans_when_l1_occupancy_changes(device):
    # The CB plan is read off live L1, which the attributes do not describe. If it is not on the
    # program hash, the plan built against a clear frontier is reused after a large L1 allocation
    # and its CB addresses overlap the resident tensor.
    #
    # Sizing is the test: plan_concat_cb keeps batch = min(4, budget / (2 * out_page)), so unless
    # the occupancy pushes that quotient under 4, all three dispatches share one plan and nothing
    # about the hash is exercised. A 128 KiB output page pins the clear batch at 4 for any budget
    # at or above 1 MiB, and 640 KiB/core drops it below 4 for any budget under 1.66 MiB while
    # still leaving a page to plan against -- one window that holds on both architectures without
    # naming either. Interleaved occupancy only counts per bank, hence the grid rather than a
    # fixed total. The distinct-entry assertion is what fails if that sizing stops biting.
    grid = device.compute_with_storage_grid_size()
    hog_rows = grid.x * grid.y * 320  # x 2 KiB rows = 640 KiB/core

    shapes, dim = [[1, 4, 32768], [1, 4, 32768]], 2  # 64 KiB sticks -> a 128 KiB output page
    xs = _inputs(shapes, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    golden = ttnn.to_torch(_force_native(xs, dim=dim))

    assert_equal(golden, ttnn.to_torch(_force_codegen(xs, dim=dim)))
    entries_clear = device.num_program_cache_entries()

    hog = ttnn.from_torch(
        torch.zeros(1, 1, hog_rows, 1024, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    assert_equal(golden, ttnn.to_torch(_force_codegen(xs, dim=dim)))
    unplanned = "the occupied dispatch reused the clear frontier's program: the CB plan is not on the hash"
    assert device.num_program_cache_entries() > entries_clear, unplanned
    entries_occupied = device.num_program_cache_entries()

    ttnn.deallocate(hog)
    assert_equal(golden, ttnn.to_torch(_force_codegen(xs, dim=dim)))
    msg = "freeing the occupancy did not return to the clear frontier's program"
    assert device.num_program_cache_entries() == entries_occupied, msg
