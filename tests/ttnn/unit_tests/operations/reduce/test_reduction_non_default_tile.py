# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The reduction kernels size every tile as 32x32, so a TILE input with another tile shape used to give wrong values
without an error (#63679). Each op now rejects it; a 32x32 tile still works. ttnn.std and ttnn.var already rejected
it (test_var_std_reject_off_default_tile)."""

import pytest
import torch
import ttnn

OPS = {
    "sum": (lambda x: ttnn.sum(x, dim=-1, keepdim=True), "ttnn.sum"),
    "sum_dim1": (lambda x: ttnn.sum(x, dim=1, keepdim=True), "ttnn.sum"),
    "mean": (lambda x: ttnn.mean(x, dim=-1, keepdim=True), "ttnn.mean"),
    "max_dim_h": (lambda x: ttnn.max(x, dim=-2, keepdim=True), "ttnn.max"),
    "min": (lambda x: ttnn.min(x, dim=-1, keepdim=True), "ttnn.min"),
    "prod": (lambda x: ttnn.prod(x, dim=-1, keepdim=True), "ttnn.prod"),
    "argmax": (lambda x: ttnn.argmax(x, dim=-1), "ttnn.argmax"),
    "topk": (lambda x: ttnn.topk(x, 8, dim=-1), "ttnn.topk"),
    "cumsum_w": (lambda x: ttnn.cumsum(x, dim=-1), "ttnn.cumsum"),
    "cumsum_dim1": (lambda x: ttnn.cumsum(x, dim=1), "ttnn.cumsum"),
    "cumprod": (lambda x: ttnn.cumprod(x, dim=-1), "ttnn.cumprod"),
}


def _input(device, tile):
    torch.manual_seed(0)
    x = (torch.randint(-64, 65, (2, 2, 64, 128)) / 8).to(torch.bfloat16)
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile(tile), device=device)


@pytest.mark.parametrize("tile", [[16, 32], [32, 16], [8, 32]], ids=lambda t: f"{t[0]}x{t[1]}")
@pytest.mark.parametrize("op", list(OPS))
def test_reduction_rejects_non_default_tile(device, op, tile, expect_error):
    call, name = OPS[op]
    x = _input(device, tile)
    with expect_error(RuntimeError, f"{name} supports only 32x32 tiles, but the input has a {tile[0]}x{tile[1]} tile"):
        call(x)


@pytest.mark.parametrize("op", list(OPS))
def test_reduction_accepts_default_tile(device, op):
    call, _ = OPS[op]
    call(_input(device, [32, 32]))


def _empty(device, shape, dtype, tile):
    return ttnn.from_torch(
        torch.zeros(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile(tile), device=device
    )


def test_topk_rejects_non_default_tile_output(device, expect_error):
    x = _input(device, [32, 32])
    values = _empty(device, (2, 2, 64, 32), ttnn.bfloat16, [16, 32])
    indices = _empty(device, (2, 2, 64, 32), ttnn.uint16, [32, 32])
    with expect_error(RuntimeError, "ttnn.topk supports only 32x32 tiles, but the output values has a 16x32 tile"):
        ttnn.topk(x, 32, dim=-1, output_tensor=(values, indices))


def test_cumsum_rejects_non_default_tile_output(device, expect_error):
    x = _input(device, [32, 32])
    out = _empty(device, (2, 2, 64, 128), ttnn.bfloat16, [16, 32])
    with expect_error(RuntimeError, "ttnn.cumsum supports only 32x32 tiles, but the output has a 16x32 tile"):
        ttnn.cumsum(x, dim=1, out=out)


def test_prod_rejects_non_default_tile_output(device, expect_error):
    x = _input(device, [32, 32])
    out = _empty(device, (1, 2, 64, 128), ttnn.bfloat16, [16, 32])
    with expect_error(RuntimeError, "ttnn.prod supports only 32x32 tiles, but the output has a 16x32 tile"):
        ttnn.prod(x, out, dims=[0])
