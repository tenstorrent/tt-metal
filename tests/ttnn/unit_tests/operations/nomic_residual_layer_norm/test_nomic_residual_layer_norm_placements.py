# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1: the model's split layout ("model_layout") and the in-place output (output_tensor=x).

model_layout is BLOCK_SHARDED L1, COL_MAJOR, 8 core rows x C columns, shard (32 * bh, 96), bh = ceil(M / 11). The
shapes pin every bh class the INPUTS reach whose plan differs (bh 3 / 4 / 5 / 6 / 8 / 9 / 11 / 12: one block, a
multi-block shard, a prime bh, a partial last column) and both residual modes; in place is checked on both
placements (the output must land in x's buffer). Tolerances are the golden suite's.
"""

import pytest
import torch
import ttnn

from ttnn.operations.nomic_residual_layer_norm import (
    circular_buffer_bytes,
    default_compute_kernel_config,
    input_memory_config,
    nomic_residual_layer_norm,
    output_memory_config,
)
from ttnn.operations.nomic_residual_layer_norm.nomic_residual_layer_norm import model_memory_config

from .test_nomic_residual_layer_norm import EPSILON, check, make_inputs, torch_reference

HIDDEN = 768

MODEL_SHAPES = [
    (2, 1, 512, 768),  # M = 32, bh 3, C 11 (partial last column: 33 > 32)
    (3, 1, 400, 768),  # M = 39, bh 4, padded sequence, partial last column
    (3, 1, 512, 768),  # M = 48, bh 5 (prime: 5 blocks of 1 row)
    (4, 1, 512, 768),  # M = 64, bh 6 (2 blocks of 3)
    (5, 1, 512, 768),  # M = 80, bh 8 (2 blocks of 4)
    (6, 1, 512, 768),  # M = 96, bh 9 (3 blocks of 3)
    (11, 1, 352, 768),  # M = 121, bh 11 (prime: 11 blocks of 1 row)
    (8, 1, 512, 768),  # M = 128, bh 12 (3 blocks of 4), perf shape
]


def _to_device(tensor, layout, device):
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=layout)


def _params(gamma, beta, device):
    def parameter(tensor, dtype):
        return ttnn.from_torch(
            tensor.reshape(1, 1, 1, HIDDEN),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return parameter(gamma, ttnn.float32), parameter(beta, ttnn.bfloat16)


def _run(device, shape, with_residual, layout, in_place):
    x, residual, gamma, beta = make_inputs(shape, with_residual)
    expected = torch_reference(x, gamma, beta, residual)
    tt_x = _to_device(x, layout, device)
    tt_r = _to_device(residual, layout, device) if with_residual else None
    tt_g, tt_b = _params(gamma, beta, device)
    kwargs = dict(residual=tt_r, epsilon=EPSILON, compute_kernel_config=default_compute_kernel_config())
    if in_place:
        kwargs["output_tensor"] = tt_x
    output = nomic_residual_layer_norm(tt_x, tt_g, tt_b, **kwargs)
    if in_place:
        assert output.buffer_address() == tt_x.buffer_address(), "in place must write over x"
    else:
        assert output.memory_config() == output_memory_config(shape, device, tt_x.memory_config())
    check(ttnn.to_torch(output).float(), expected)


@pytest.mark.parametrize("shape", MODEL_SHAPES, ids=lambda s: f"{s[0]}x{s[2]}")
@pytest.mark.parametrize("with_residual", [True, False], ids=["residual", "no_residual"])
@pytest.mark.parametrize("in_place", [False, True], ids=["new", "in_place"])
def test_model_layout(device, shape, with_residual, in_place):
    _run(device, shape, with_residual, model_memory_config(shape, device), in_place)


@pytest.mark.parametrize(
    "shape", [(2, 1, 512, 768), (5, 1, 512, 768), (8, 1, 512, 768)], ids=lambda s: f"{s[0]}x{s[2]}"
)
@pytest.mark.parametrize("with_residual", [True, False], ids=["residual", "no_residual"])
def test_op_layout_in_place(device, shape, with_residual):
    _run(device, shape, with_residual, input_memory_config(shape, device), True)


def test_model_layout_matches_model(device):
    """The op's replica of the model layout equals the model's own config (the golden suite compares them)."""
    from models.experimental.nomic_embed_text_v2_moe.tt.common import sharded_norm_configs, sharded_norm_split

    grid = device.compute_with_storage_grid_size()
    for m in range(32, 129):
        shape = (1, 1, 32 * m, 768)
        assert model_memory_config(shape, device) == sharded_norm_configs(*sharded_norm_split(m, 24, grid), 24)[0], m


def test_cb_budget():
    for placement in ("op_layout", "model_layout"):
        for m in range(32, 129):
            assert circular_buffer_bytes((1, 1, 32 * m, 768), placement=placement) <= 256 * 1024, (placement, m)
