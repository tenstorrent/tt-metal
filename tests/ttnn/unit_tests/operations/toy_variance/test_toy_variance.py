# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for toy_variance — a Metal 2.0 ProgramSpec op with two placements.

Verifies bf16 per-row population variance against torch.var(unbiased=False):

- INTERLEAVED: one core streams the row, including the headline 32 x 64000 case which is too wide
  to fit in L1 and so exercises the streaming chunking, plus the non-aligned partial-scaler paths.
- WIDTH-SHARDED: W is split across a row of cores, so the reduced axis is split and the combine
  crosses cores (gather to root, broadcast the mean back, gather again). Deliberately narrow — the
  unsupported-placement test pins that a shape outside the gate raises instead of falling back.
"""

import pytest
import torch
import ttnn

from ttnn.operations.toy_variance import toy_variance


@pytest.mark.parametrize(
    "shape",
    [
        # Tile-aligned W
        pytest.param((1, 1, 32, 256), id="W=256_aligned"),
        pytest.param((1, 1, 32, 1024), id="W=1024_aligned"),
        pytest.param((1, 1, 32, 8192), id="W=8192_aligned"),
        pytest.param((1, 1, 32, 64000), id="W=64000_wide_aligned"),
        # Non-aligned W — exercises the partial-scaler path
        pytest.param((1, 1, 32, 33), id="W=33_partial=1"),
        pytest.param((1, 1, 32, 100), id="W=100_partial=4"),
        pytest.param((1, 1, 32, 257), id="W=257_partial=1"),
        pytest.param((1, 1, 32, 1023), id="W=1023_partial=31"),
        # Non-aligned H — output rows beyond origin_H are garbage and sliced off
        pytest.param((1, 1, 33, 64), id="H=33_W=64"),
        pytest.param((1, 1, 33, 100), id="H=33_W=100_both_partial"),
    ],
)
@pytest.mark.parametrize("std_dev", [False, True], ids=["variance", "std_dev"])
def test_toy_variance(device, shape, std_dev):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 0.5

    if std_dev:
        torch_expected = torch.std(torch_input.float(), dim=-1, keepdim=True, unbiased=False)
    else:
        torch_expected = torch.var(torch_input.float(), dim=-1, keepdim=True, unbiased=False)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Fill implicit tile padding with non-zero garbage. If the partial scaler
    # is doing its job, the contaminated values are zeroed out in the reduce
    # and the result still matches torch. If it's broken, (99 - mean)^2 ≈ 9801
    # would dominate the variance and the comparison would fail by orders of
    # magnitude — this is the actual partial-scaler correctness check.
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, 99.0)

    ttnn_output = toy_variance(ttnn_input, std_dev=std_dev)
    torch_output = ttnn.to_torch(ttnn_output)

    # Result lives in column 0 of each output tile. For non-aligned H, rows
    # beyond origin_H in the output tile are padded garbage — slice them off.
    H = shape[-2]
    actual = torch_output[..., :H, :1].float()
    expected = torch_expected.float()

    W = shape[-1]
    atol = max(0.05, 0.001 * (W / 256))
    rtol = 0.10

    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"Mismatch for shape={shape}, std_dev={std_dev}:\n"
        f"  max abs diff = {(actual - expected).abs().max().item():.6f}\n"
        f"  actual[:4]   = {actual.flatten()[:4].tolist()}\n"
        f"  expected[:4] = {expected.flatten()[:4].tolist()}"
    )


def _width_sharded(torch_input, device, num_cores, *, shard_width=None, grid=None):
    """Width-shard a (1, 1, H, W) tensor across `num_cores` cores, one grid row by default.

    `shard_width` defaults to an even split. Note the tensor layer requires the shard width itself
    to be tile-sized -- a non-tile-sized shard FATALs here, before any op sees it -- so the only way
    to hand the op a non-tile-aligned W is a logical W that PADS up to a tile-aligned one.
    """
    H, W = torch_input.shape[-2], torch_input.shape[-1]
    if grid is None:
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))])
    shard_spec = ttnn.ShardSpec(
        grid, [H, W // num_cores if shard_width is None else shard_width], ttnn.ShardOrientation.ROW_MAJOR
    )
    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    return ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )


@pytest.mark.parametrize(
    "num_cores, shape",
    [
        # W split across cores -> the reduced axis is split -> the combine crosses cores.
        pytest.param(2, (1, 1, 32, 64), id="P=2_W=64_one_tile_each"),
        pytest.param(2, (1, 1, 32, 128), id="P=2_W=128"),
        pytest.param(4, (1, 1, 32, 512), id="P=4_W=512"),
        pytest.param(8, (1, 1, 32, 1024), id="P=8_W=1024"),
        pytest.param(8, (1, 1, 32, 8192), id="P=8_W=8192_wide"),
        # Ht > 1: each core's contribution is a multi-tile block, so the gather moves Ht tiles
        # per core and the root's combine reduces num_cores blocks rather than num_cores tiles.
        pytest.param(4, (1, 1, 64, 512), id="P=4_H=64_Ht=2"),
        pytest.param(8, (1, 1, 128, 2048), id="P=8_H=128_Ht=4"),
        # An odd core count, so the reducer's odd-block seeding path runs.
        pytest.param(7, (1, 1, 32, 448), id="P=7_odd_core_count"),
    ],
)
@pytest.mark.parametrize("std_dev", [False, True], ids=["variance", "std_dev"])
def test_toy_variance_width_sharded(device, num_cores, shape, std_dev):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 0.5

    torch_expected = torch.var(torch_input.float(), dim=-1, keepdim=True, unbiased=False)
    if std_dev:
        torch_expected = torch_expected.sqrt()

    ttnn_input = _width_sharded(torch_input, device, num_cores)
    ttnn_output = toy_variance(ttnn_input, std_dev=std_dev)

    H = shape[-2]
    actual = ttnn.to_torch(ttnn_output)[..., :H, :1].float()
    expected = torch_expected.float()

    # Every core reduces with 1/N over the full width and the root adds the partials, so the error
    # is bf16 rounding on the partials rather than anything that grows with the core count.
    assert torch.allclose(actual, expected, rtol=0.10, atol=0.05), (
        f"Mismatch for num_cores={num_cores}, shape={shape}, std_dev={std_dev}:\n"
        f"  max abs diff = {(actual - expected).abs().max().item():.6f}\n"
        f"  actual[:4]   = {actual.flatten()[:4].tolist()}\n"
        f"  expected[:4] = {expected.flatten()[:4].tolist()}"
    )


@pytest.mark.parametrize(
    "label, shape, num_cores, shard_width, two_rows, reason",
    [
        # W pads up to a tile-aligned width, so the shard is legal but the op's own W is not.
        pytest.param("padded_W", (1, 1, 32, 500), 4, 128, False, "tile-aligned", id="W_not_tile_aligned"),
        # Ragged splits: the last core would own fewer tiles than the rest.
        pytest.param("ragged_5over2", (1, 1, 32, 160), 2, 96, False, "divide evenly", id="Wt5_over_2_cores"),
        pytest.param("ragged_16over3", (1, 1, 32, 512), 3, 192, False, "divide evenly", id="Wt16_over_3_cores"),
        # Nothing to combine across.
        pytest.param("single_core", (1, 1, 32, 128), 1, 128, False, "at least 2 cores", id="single_core"),
        # A 2x2 grid: the gather fan-in and the mean broadcast are both one-row shapes.
        pytest.param("two_rows", (1, 1, 32, 512), 4, 128, True, "single grid row", id="two_row_shard_grid"),
    ],
)
def test_toy_variance_width_sharded_unsupported(
    device, expect_error, label, shape, num_cores, shard_width, two_rows, reason
):
    """The sharded path is deliberately narrow: an unsupported placement raises, never falls back.

    A silent fall-back to the interleaved path would still produce the right numbers, which is
    exactly why it would be wrong -- the shape would look supported and the cross-core code would
    never run. Every case here builds a real sharded tensor first, so what is being tested is the
    op's gate and not the tensor layer's.
    """
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16) * 0.5

    grid = None
    if two_rows:
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])

    ttnn_input = _width_sharded(torch_input, device, num_cores, shard_width=shard_width, grid=grid)
    with expect_error(NotImplementedError, reason):
        toy_variance(ttnn_input)
