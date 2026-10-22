# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# --- BEGIN GENERATED (agentic_port phase 8) ---
# Contracts: (1) every case the codegen gate rejects falls back to native; (2) an accepted case
# dispatched twice on the codegen path stays a program-cache hit and rebinds its buffers; (3) a
# perf-demoted case still lands on native. This block is emitted from the port's coverage ledger
# and is regenerated on every port update; hand-add off-grid regressions after its END line.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.concat` takes no implementation argument -- it routes on its own. The forced legs below are
# the verification-only entries in the private module; see concat_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.concat_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.concat_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


def _device_inputs(shapes, dtype, layout, memory_config=None, **on_device):
    # One tensor per input shape, in order: the op takes them as a list. `memory_config` is one
    # placement for every input, or a tuple holding one per input.
    configs = memory_config if isinstance(memory_config, tuple) else (memory_config,) * len(shapes)
    return [
        ttnn.from_torch(
            _make_input(shape, dtype),
            dtype=dtype,
            layout=layout,
            **on_device,
            **({} if config is None else {"memory_config": config}),
        )
        for shape, config in zip(shapes, configs)
    ]


def _reference(inputs, **kwargs):
    # Torch golden over what the device inputs read back as, so a dtype round trip cannot differ.
    return torch.cat([ttnn.to_torch(t) for t in inputs], dim=kwargs["dim"])


# 12 of the ledger's 979 gate-rejected cases, sampled to keep every refusal reason
# and the extremes of each reason's input and call sizes; the port pipeline checked the full set.
_ROUTING = [
    (
        [[1, 1, 16], [1, 1, 1]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [[1, 1, 16], [1, 1, 1]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [[1, 1, 16], [1, 1, 1]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [
            [1, 192, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
        ],
        {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        (
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ),
    ),
    (
        [
            [1, 256, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
        ],
        {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        (
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ),
    ),
    (
        [
            [1, 256, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
            [1, 32, 14, 14],
        ],
        {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
    ),
    (
        [[1, 32, 64], [1, 32, 64], [2, 32, 64]],
        {"dim": 0, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.uint32,
        ttnn.TILE_LAYOUT,
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ),
    (
        [
            [1, 384, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
        ],
        {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [
            [1, 384, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
            [1, 48, 14, 14],
        ],
        {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [
            [1, 512, 7, 7],
            [1, 32, 7, 7],
            [1, 32, 7, 7],
            [1, 32, 7, 7],
            [1, 32, 7, 7],
            [1, 32, 7, 7],
            [1, 32, 7, 7],
            [1, 32, 7, 7],
            [1, 32, 7, 7],
            [1, 32, 7, 7],
            [1, 32, 7, 7],
            [1, 32, 7, 7],
        ],
        {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.uint32,
        ttnn.TILE_LAYOUT,
        (
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
    ),
    (
        [[32, 32, 33, 7], [32, 32, 33, 16], [32, 32, 33, 16]],
        {"dim": 3, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.int32,
        ttnn.TILE_LAYOUT,
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ),
    (
        [[64, 96, 96, 32], [64, 96, 96, 64], [64, 96, 96, 64]],
        {"dim": 3, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ),
]
_ROUTING_IDS = [
    "[[1, 1, 16], [1, 1, 1]]|dim=-1&memory_config=BufferType.DRAM|bfloat16|tile",
    "[[1, 1, 16], [1, 1, 1]]|dim=-1&memory_config=BufferType.DRAM|bfloat8_b|tile",
    "[[1, 1, 16], [1, 1, 1]]|dim=-1&memory_config=BufferType.DRAM|float32|row_major",
    "[[1, 192, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28]]@L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1|dim=1&memory_config=BufferType.DRAM|uint32|row_major",
    "[[1, 256, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]]@DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1|dim=1&memory_config=BufferType.DRAM|bfloat8_b|tile",
    "[[1, 256, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14], [1, 32, 14, 14]]@L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 32, 64], [1, 32, 64], [2, 32, 64]]@DRAM,L1,DRAM|dim=0&memory_config=BufferType.DRAM|uint32|tile",
    "[[1, 384, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]]|dim=1&memory_config=BufferType.DRAM|uint32|row_major",
    "[[1, 384, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14], [1, 48, 14, 14]]|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 512, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7], [1, 32, 7, 7]]@L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM|dim=1&memory_config=BufferType.DRAM|uint32|tile",
    "[[32, 32, 33, 7], [32, 32, 33, 16], [32, 32, 33, 16]]@L1,DRAM,L1|dim=3&memory_config=BufferType.DRAM|int32|tile",
    "[[64, 96, 96, 32], [64, 96, 96, 64], [64, 96, 96, 64]]@L1,DRAM,L1|dim=3&memory_config=BufferType.DRAM|bfloat8_b|tile",
]


@pytest.mark.parametrize("shapes,kwargs,dtype,layout,placement", _ROUTING, ids=_ROUTING_IDS)
def test_concat_codegen_routing(device, shapes, kwargs, dtype, layout, placement):
    xt = _device_inputs(shapes, dtype, layout, device=device, memory_config=placement)
    device.clear_program_cache()
    _force_native(xt, **kwargs)  # warm native, so the entry count below sees only codegen
    golden = _reference(xt, **kwargs)
    entries_before = device.num_program_cache_entries()
    out = ttnn.concat(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed an out-of-scope case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


_DEMOTED = [
    (
        [[1, 32, 32], [1, 32, 64], [1, 32, 32]],
        {"dim": 2, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ),
    (
        [[1, 32, 32], [1, 32, 64], [1, 32, 32]],
        {"dim": 2, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.int32,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ),
    (
        [[1, 32, 32], [1, 32, 64], [1, 32, 32]],
        {"dim": 2, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ),
    (
        [[1, 32, 32], [1, 32, 64], [1, 32, 32]],
        {"dim": 2, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ),
    (
        [[1, 32, 32], [1, 32, 64], [1, 32, 32]],
        {"dim": 2, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.int32,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ),
    (
        [[1, 32, 32], [1, 32, 64], [1, 32, 32]],
        {"dim": 2, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ),
    (
        [[1, 32, 32], [1, 32, 64], [1, 32, 32]],
        {"dim": 2, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [[1, 32, 32], [1, 32, 64], [1, 32, 32]],
        {"dim": 2, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.int32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [[1, 32, 32], [1, 32, 64], [1, 32, 32]],
        {"dim": 2, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [[32, 32], [32, 64], [32, 96]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ),
    (
        [[32, 32], [32, 64], [32, 96]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.int32,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ),
    (
        [[32, 32], [32, 64], [32, 96]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ),
    (
        [[32, 32], [32, 64], [32, 96]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ),
    (
        [[32, 32], [32, 64], [32, 96]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.int32,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ),
    (
        [[32, 32], [32, 64], [32, 96]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ),
    (
        [[32, 32], [32, 64], [32, 96]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [[32, 32], [32, 64], [32, 96]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.int32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [[32, 32], [32, 64], [32, 96]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
]
_DEMOTED_IDS = [
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]@DRAM,L1,DRAM|dim=2&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]@DRAM,L1,DRAM|dim=2&memory_config=BufferType.DRAM|int32|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]@DRAM,L1,DRAM|dim=2&memory_config=BufferType.DRAM|uint32|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]@L1,DRAM,L1|dim=2&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]@L1,DRAM,L1|dim=2&memory_config=BufferType.DRAM|int32|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]@L1,DRAM,L1|dim=2&memory_config=BufferType.DRAM|uint32|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2&memory_config=BufferType.DRAM|int32|row_major",
    "[[1, 32, 32], [1, 32, 64], [1, 32, 32]]|dim=2&memory_config=BufferType.DRAM|uint32|row_major",
    "[[32, 32], [32, 64], [32, 96]]@DRAM,L1,DRAM|dim=-1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[32, 32], [32, 64], [32, 96]]@DRAM,L1,DRAM|dim=-1&memory_config=BufferType.DRAM|int32|row_major",
    "[[32, 32], [32, 64], [32, 96]]@DRAM,L1,DRAM|dim=-1&memory_config=BufferType.DRAM|uint32|row_major",
    "[[32, 32], [32, 64], [32, 96]]@L1,DRAM,L1|dim=-1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[32, 32], [32, 64], [32, 96]]@L1,DRAM,L1|dim=-1&memory_config=BufferType.DRAM|int32|row_major",
    "[[32, 32], [32, 64], [32, 96]]@L1,DRAM,L1|dim=-1&memory_config=BufferType.DRAM|uint32|row_major",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1&memory_config=BufferType.DRAM|int32|row_major",
    "[[32, 32], [32, 64], [32, 96]]|dim=-1&memory_config=BufferType.DRAM|uint32|row_major",
]


@pytest.mark.parametrize("shapes,kwargs,dtype,layout,placement", _DEMOTED, ids=_DEMOTED_IDS)
def test_concat_codegen_demotion(device, shapes, kwargs, dtype, layout, placement):
    xt = _device_inputs(shapes, dtype, layout, device=device, memory_config=placement)
    device.clear_program_cache()
    _force_native(xt, **kwargs)  # warm native, so the entry count below sees only codegen
    golden = _reference(xt, **kwargs)
    entries_before = device.num_program_cache_entries()
    out = ttnn.concat(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed a perf-demoted case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


_CACHE_HIT = [
    (
        [[1, 1, 16], [1, 1, 1]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ),
    (
        [[1, 1, 16], [1, 1, 1]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ),
    (
        [[1, 1, 16], [1, 1, 1]],
        {"dim": -1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [
            [1, 192, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
        ],
        {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ),
    ),
    (
        [
            [1, 192, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
            [1, 48, 28, 28],
        ],
        {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
    ),
    (
        [[1, 20, 28, 28], [1, 34, 28, 28], [1, 58, 28, 28], [1, 98, 28, 28], [1, 256, 28, 28]],
        {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
    ),
    (
        [[1, 20, 28, 28], [1, 34, 28, 28], [1, 58, 28, 28], [1, 98, 28, 28], [1, 256, 28, 28]],
        {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ),
    ),
    (
        [
            [1, 64, 56, 56],
            [1, 32, 56, 56],
            [1, 32, 56, 56],
            [1, 32, 56, 56],
            [1, 32, 56, 56],
            [1, 32, 56, 56],
            [1, 32, 56, 56],
        ],
        {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
    ),
    (
        [
            [1, 64, 56, 56],
            [1, 32, 56, 56],
            [1, 32, 56, 56],
            [1, 32, 56, 56],
            [1, 32, 56, 56],
            [1, 32, 56, 56],
            [1, 32, 56, 56],
        ],
        {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ),
    ),
    (
        [[1, 64, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56]],
        {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ),
    ),
    (
        [[1, 64, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56]],
        {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
    ),
    (
        [[32, 32], [32, 32], [32, 32], [32, 32]],
        {"dim": 0, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ),
    (
        [[32, 32], [32, 32], [32, 32], [32, 32]],
        {"dim": 0, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ),
    (
        [[32, 64], [32, 64], [32, 64]],
        {"dim": 0, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ),
    (
        [[32, 64], [32, 64], [32, 64]],
        {"dim": 0, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ),
]
_CACHE_HIT_IDS = [
    "[[1, 1, 16], [1, 1, 1]]@DRAM,L1|dim=-1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 1, 16], [1, 1, 1]]@L1,DRAM|dim=-1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 1, 16], [1, 1, 1]]|dim=-1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 192, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28]]@DRAM,L1,DRAM,L1,DRAM,L1,DRAM,L1|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 192, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28], [1, 48, 28, 28]]@L1,DRAM,L1,DRAM,L1,DRAM,L1,DRAM|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 20, 28, 28], [1, 34, 28, 28], [1, 58, 28, 28], [1, 98, 28, 28], [1, 256, 28, 28]]@DRAM,L1,DRAM,L1,DRAM|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 20, 28, 28], [1, 34, 28, 28], [1, 58, 28, 28], [1, 98, 28, 28], [1, 256, 28, 28]]@L1,DRAM,L1,DRAM,L1|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 64, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56]]@DRAM,L1,DRAM,L1,DRAM,L1,DRAM|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 64, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56]]@L1,DRAM,L1,DRAM,L1,DRAM,L1|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 64, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56]]@DRAM,L1,DRAM,L1,DRAM,L1|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[1, 64, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56], [1, 32, 56, 56]]@L1,DRAM,L1,DRAM,L1,DRAM|dim=1&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]@DRAM,L1,DRAM,L1|dim=0&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[32, 32], [32, 32], [32, 32], [32, 32]]@L1,DRAM,L1,DRAM|dim=0&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[32, 64], [32, 64], [32, 64]]@DRAM,L1,DRAM|dim=0&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[[32, 64], [32, 64], [32, 64]]@L1,DRAM,L1|dim=0&memory_config=BufferType.DRAM|bfloat16|row_major",
]


@pytest.mark.parametrize("shapes,kwargs,dtype,layout,placement", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_concat_codegen_auto_routes_to_codegen(device, shapes, kwargs, dtype, layout, placement):
    xt = _device_inputs(shapes, dtype, layout, device=device, memory_config=placement)
    device.clear_program_cache()
    _force_native(xt, **kwargs)  # warm native, so the entry count below sees only codegen
    golden = _reference(xt, **kwargs)
    entries_before = device.num_program_cache_entries()
    out = ttnn.concat(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed a supported case to native (program cache did not grow); expected codegen"
    assert device.num_program_cache_entries() > entries_before, msg


@pytest.mark.parametrize("shapes,kwargs,dtype,layout,placement", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_concat_codegen_program_cache_hit(device, shapes, kwargs, dtype, layout, placement):
    xt = _device_inputs(shapes, dtype, layout, device=device, memory_config=placement)
    golden = _reference(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    entries_after_miss = device.num_program_cache_entries()
    # Same spec, a distinct allocation: the cached program must rebind its Buffer*s
    # instead of reusing the first dispatch's addresses.
    yt = _device_inputs(shapes, dtype, layout, device=device, memory_config=placement)
    second_golden = _reference(yt, **kwargs)
    assert_equal(second_golden, ttnn.to_torch(_force_codegen(yt, **kwargs)))
    msg = "second forced-codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_after_miss, msg


_REFUSED_CASE = (
    [[1, 1, 1024], [1, 196, 1024]],
    {"dim": 1, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
    ttnn.bfloat16,
    ttnn.TILE_LAYOUT,
    (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
)


def test_concat_forced_codegen_refuses_out_of_scope_case(device, expect_error):
    shapes, kwargs, dtype, layout, placement = _REFUSED_CASE
    xt = _device_inputs(shapes, dtype, layout, device=device, memory_config=placement)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, **kwargs)


@pytest.mark.parametrize("shapes,kwargs,dtype,layout,placement", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_concat_codegen_program_key_is_frontier_invariant(device, shapes, kwargs, dtype, layout, placement):
    xt = _device_inputs(shapes, dtype, layout, device=device, memory_config=placement)
    golden = _reference(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    entries_before = device.num_program_cache_entries()
    # Trace capture compiles nothing it has not already cached, and what warm-up cached
    # depends on where the frontier stood at warm-up -- so a CB plan that reads live L1
    # occupancy is uncapturable, and the same call re-keys the cache once anything else is
    # resident. This tensor moves the frontier by many pages per bank, so a plan reading it
    # at all lands on a different key here.
    resident = ttnn.from_torch(
        torch.zeros([1, 1, 256, 4096], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    try:
        assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
        msg = "the program key moved with the L1 frontier, so the op cannot be traced"
        assert device.num_program_cache_entries() == entries_before, msg
    finally:
        ttnn.deallocate(resident)


# --- END GENERATED ---


# --- Off-grid regressions (hand-added; edit here, not the emitter) ---
