# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Knob coverage for g6_permute rows_tile_scatter: paths a full grid never takes on the acceptance shapes.

With block_in_tiles = a core's whole assignment every core runs one block, so the depth-2 CB ring,
ragged last blocks and the flush-then-refill hand-over never run. These cases cap block_in_tiles
(and turn the stagger / depth knobs) and check bit-exact results with the input padding poisoned.
The cached-descriptor case checks that a cache hit really rewrites the buffer addresses.
DO NOT DELETE.
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.g6_permute import g6_permute, g6_permute_plan

op_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute")
descriptor_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute_program_descriptor")

ROWS = (0, 2, 1, 3)


def _to_device(torch_input, device):
    return ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _bits(t):
    return t.to(torch.bfloat16).contiguous().view(torch.int16)


@pytest.fixture
def knobs(monkeypatch):
    def force(block_limit, input_depth, stagger):
        monkeypatch.setattr(descriptor_module, "BLOCK_IN_TILES_LIMIT", block_limit)
        monkeypatch.setattr(descriptor_module, "INPUT_DEPTH", input_depth)
        monkeypatch.setattr(descriptor_module, "STAGGER_ROWS", stagger)

    # The knobs are not part of the host cache key; start and end with an empty cache.
    op_module._PROGRAM_CACHE.clear()
    yield force
    op_module._PROGRAM_CACHE.clear()


# id, shape, block_in_tiles limit, input_depth, stagger
CASES = [
    ("multiblock_depth2_ragged", (1, 64, 100, 256), 3, 2, 1),
    ("multiblock_depth1", (1, 64, 100, 256), 4, 1, 1),
    ("multiblock_no_stagger", (2, 34, 47, 96), 1, 2, 0),
    ("lower_band_rows_ragged", (1, 40, 49, 64), 2, 2, 1),
]


@pytest.mark.parametrize("shape, block_limit, input_depth, stagger", [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_forced_blocks_bit_exact(shape, block_limit, input_depth, stagger, device, knobs):
    knobs(block_limit, input_depth, stagger)
    torch.manual_seed(7)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    ttnn_input = _to_device(torch_input, device)
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, float("nan"), memory_config=ttnn_input.memory_config())

    plan = g6_permute_plan(ttnn_input, ROWS)
    assert plan.block_in_tiles <= block_limit
    if plan.tiles_per_core_g1 > block_limit:
        assert plan.max_blocks_per_core > 1
        assert plan.input_depth == min(input_depth, plan.max_blocks_per_core)

    actual = ttnn.to_torch(g6_permute(ttnn_input, ROWS))
    expected = torch.permute(torch_input, ROWS).contiguous()
    assert torch.equal(_bits(actual), _bits(expected))


def test_cache_hit_patches_both_addresses(device):
    """Two inputs alive at once sit at different addresses; a stale address would replay the first."""
    torch.manual_seed(3)
    shape = (1, 6, 16, 64)
    first = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    second = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    in_first = _to_device(first, device)
    in_second = _to_device(second, device)
    assert in_first.buffer_address() != in_second.buffer_address()

    out_first = g6_permute(in_first, ROWS)
    out_second = g6_permute(in_second, ROWS)
    assert out_first.buffer_address() != out_second.buffer_address()
    assert torch.equal(_bits(ttnn.to_torch(out_first)), _bits(torch.permute(first, ROWS).contiguous()))
    assert torch.equal(_bits(ttnn.to_torch(out_second)), _bits(torch.permute(second, ROWS).contiguous()))
