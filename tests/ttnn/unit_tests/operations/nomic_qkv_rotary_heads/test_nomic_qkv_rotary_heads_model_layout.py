# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 (`model_layout`): x in the model's split layout (legacy BLOCK_SHARDED, COL_MAJOR, 8 core rows x C core
columns, shard (32 * bh, 96)), read through the gather's block-shard x source table.

- The op's own rebuild of the layout equals the model's builder (sharded_norm_split / sharded_norm_configs) for every
  M class, so validate()'s placement tag is honest.
- Correctness on the M classes that stress the table: 11 vs 10 grid columns, a padded sequence, 4 M blocks a core.
- A block shard that is not the model's (ROW_MAJOR orientation) is refused, not silently accepted.
"""

import importlib.util
from pathlib import Path

import pytest
import ttnn

from models.experimental.nomic_embed_text_v2_moe.tt.common import sharded_norm_configs, sharded_norm_split
from ttnn.operations.nomic_qkv_rotary_heads import (
    default_compute_kernel_config,
    nomic_qkv_rotary_heads,
    prepare_qkv_weights,
    prepare_rotary_tables,
)
from ttnn.operations.nomic_qkv_rotary_heads.nomic_qkv_rotary_heads import model_layout_memory_config


def _load_acceptance_module():
    """The acceptance test's reference, inputs and checks (sibling file; this test dir is not a package)."""
    path = Path(__file__).with_name("test_nomic_qkv_rotary_heads.py")
    spec = importlib.util.spec_from_file_location("_nomic_qkv_rotary_heads_acceptance_ml", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_acceptance = _load_acceptance_module()
allocate_heads = _acceptance.allocate_heads
check_outputs = _acceptance.check_outputs
make_inputs = _acceptance.make_inputs
torch_reference = _acceptance.torch_reference

HIDDEN = 768


def model_builder_layout(shape, device):
    m = shape[0] * ((shape[-2] + 31) // 32)
    split = sharded_norm_split(m, HIDDEN // 32, device.compute_with_storage_grid_size())
    assert split is not None
    return sharded_norm_configs(*split, HIDDEN // 32)[0]


LAYOUT_SHAPES = [
    (2, 1, 512, 768),
    (3, 1, 352, 768),
    (3, 1, 400, 768),
    (4, 1, 384, 768),
    (5, 1, 300, 768),
    (8, 1, 320, 768),
    (7, 1, 512, 768),
    (8, 1, 500, 768),
    (8, 1, 512, 768),
]


@pytest.mark.parametrize("shape", LAYOUT_SHAPES, ids=[f"{s[0]}x{s[2]}" for s in LAYOUT_SHAPES])
def test_layout_matches_the_model_builder(shape, device):
    assert model_layout_memory_config(shape, device) == model_builder_layout(shape, device)


SHAPES = [(2, 1, 512, 768), (3, 1, 400, 768), (8, 1, 320, 768), (8, 1, 512, 768)]


@pytest.mark.parametrize("shape", SHAPES, ids=[f"{s[0]}x{s[2]}" for s in SHAPES])
def test_model_layout(shape, device):
    x, weight, bias = make_inputs(shape)
    expected = torch_reference(x, weight, bias)
    tt_x = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=model_builder_layout(shape, device),
    )
    provided = allocate_heads(shape, device, ttnn.L1_MEMORY_CONFIG)
    outputs = nomic_qkv_rotary_heads(
        tt_x,
        prepare_qkv_weights(weight, bias, device),
        prepare_rotary_tables(shape[-2], device),
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tensors=provided,
        compute_kernel_config=default_compute_kernel_config(),
    )
    check_outputs(outputs, expected, shape, ttnn.L1_MEMORY_CONFIG, provided)


def test_row_major_block_shard_is_refused(device, expect_error):
    shape = (2, 1, 512, 768)
    x, weight, bias = make_inputs(shape)
    # 8 x 8 cores, shard (128, 96), ROW_MAJOR: a valid block shard of x, but not the model's layout.
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    other = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (128, 96), ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=other)
    with expect_error(NotImplementedError, ""):
        nomic_qkv_rotary_heads(
            tt_x, prepare_qkv_weights(weight, bias, device), prepare_rotary_tables(shape[-2], device)
        )
