# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics


def _height_sharded_config(batch: int, heads: int, height: int, width: int, num_cores: int):
    total_tile_rows = batch * heads * (height // 32)
    assert total_tile_rows % num_cores == 0
    block_h = total_tile_rows // num_cores
    block_w = width // 32
    core_grid = ttnn.CoreGrid(y=1, x=num_cores)
    mem_config = ttnn.create_sharded_memory_config(
        (batch, heads, height, width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    prog_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(num_cores, 1),
        subblock_w=1,
        block_h=block_h,
        block_w=block_w,
    )
    return mem_config, prog_config


def test_causal_row_major_mask_rejected(device):
    """Causal scale_mask_softmax must reject ROW_MAJOR masks cleanly."""
    torch.manual_seed(0)
    b, h, m, k = 1, 2, 64, 64
    x = torch.randn((b, h, m, k), dtype=torch.bfloat16)
    rm_mask = torch.zeros((b, 1, k // 32, 32), dtype=torch.bfloat16)

    mem_config, prog_config = _height_sharded_config(b, h, m, k, num_cores=2)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config)
    tt_mask = ttnn.from_torch(rm_mask, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with pytest.raises(RuntimeError, match="Causal mask does not support ROW_MAJOR layout"):
        ttnn.scale_mask_softmax_in_place(
            tt_x, 0.125, tt_mask, program_config=prog_config, is_causal_mask=True
        )


def test_sharded_mask_spec_and_layout_mismatch_rejected(device):
    """Sharded mask must match the input tensor shard_spec and memory_layout."""
    torch.manual_seed(1)
    b, h, m, k = 2, 2, 64, 64
    x = torch.randn((b, h, m, k), dtype=torch.bfloat16)
    mask = torch.zeros((b, h, m, k), dtype=torch.bfloat16)

    in_mem_config, prog_config = _height_sharded_config(b, h, m, k, num_cores=4)
    wrong_mem_config, _ = _height_sharded_config(b, h, m, k, num_cores=2)

    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mem_config
    )
    tt_mask = ttnn.from_torch(
        mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=wrong_mem_config
    )

    with pytest.raises(RuntimeError, match="Sharded mask memory layout and shard spec must match input tensor"):
        ttnn.scale_mask_softmax_in_place(tt_x, 0.125, tt_mask, program_config=prog_config, is_causal_mask=True)


def test_non_sharded_tile_mask_width_mismatch_rejected(device):
    """Non-sharded TILE mask width must match input width."""
    torch.manual_seed(2)
    b, h, m, k = 1, 2, 64, 128
    x = torch.randn((b, h, m, k), dtype=torch.bfloat16)
    narrow_mask = torch.zeros((1, 1, 32, 64), dtype=torch.bfloat16)

    mem_config, prog_config = _height_sharded_config(b, h, m, k, num_cores=2)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config)
    tt_mask = ttnn.from_torch(narrow_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError, match="Non-sharded mask width"):
        ttnn.scale_mask_softmax_in_place(tt_x, 0.125, tt_mask, program_config=prog_config, is_causal_mask=False)


def test_sharded_mask_batch_mismatch_rejected(device):
    """Sharded softmax rejects masks whose batch dimension is neither 1 nor input batch size."""
    torch.manual_seed(3)
    b, h, m, k = 4, 1, 64, 64
    x = torch.randn((b, h, m, k), dtype=torch.bfloat16)
    bad_batch_mask = torch.zeros((2, 1, 1, k), dtype=torch.bfloat16)

    mem_config, prog_config = _height_sharded_config(b, h, m, k, num_cores=4)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config)
    tt_mask = ttnn.from_torch(bad_batch_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError, match="Mask batch size"):
        ttnn.scale_mask_softmax_in_place(tt_x, 0.125, tt_mask, program_config=prog_config, is_causal_mask=False)


def test_non_causal_sharded_tile_mask(device):
    """Non-causal sharded TILE mask reads resident shard correctly without broadcasting row 0."""
    torch.manual_seed(4)
    b, h, m, k = 2, 2, 64, 64
    scale = 0.25
    x = torch.randn((b, h, m, k), dtype=torch.bfloat16)
    mask = torch.where(torch.rand((b, h, m, k)) > 0.5, torch.tensor(-1e4), torch.tensor(0.0)).bfloat16()

    mem_config, prog_config = _height_sharded_config(b, h, m, k, num_cores=4)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config)
    tt_mask = ttnn.from_torch(
        mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )

    out = ttnn.scale_mask_softmax_in_place(tt_x, scale, tt_mask, program_config=prog_config, is_causal_mask=False)
    ref = torch.softmax(x.float() * scale + mask.float(), dim=-1)
    assert_numeric_metrics(ref, ttnn.to_torch(out).float(), pcc_threshold=0.999, rtol=1e-2, atol=1e-3, frobenius_threshold=1e-2)


def test_interleaved_causal_mask_non_divisible_ht_multi_batch(device):
    """Interleaved causal reader resets mask_id = mask_id_offset when ht == Ht even if Ht % mask_Ht != 0."""
    torch.manual_seed(5)
    b, h, m, k = 3, 1, 160, 64  # Ht = 5 tiles, mask_Ht = 3 tiles (5 % 3 != 0)
    mask_h = 96
    scale = 0.125
    x = torch.randn((b, h, m, k), dtype=torch.bfloat16)
    mask = torch.where(torch.rand((b, 1, mask_h, k)) > 0.6, torch.tensor(-1e4), torch.tensor(0.0)).bfloat16()

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.scale_mask_softmax_in_place(tt_x, scale, tt_mask, is_causal_mask=True)
    tiled_mask = mask.repeat(1, h, (m + mask_h - 1) // mask_h, 1)[:, :, :m, :]
    ref = torch.softmax(x.float() * scale + tiled_mask.float(), dim=-1)
    assert_numeric_metrics(ref, ttnn.to_torch(out).float(), pcc_threshold=0.999, rtol=1e-2, atol=1e-3, frobenius_threshold=1e-2)


@pytest.mark.parametrize(
    "batch, heads, height, width, mask_batch, mask_height, num_cores",
    [
        # Shard spans across batch boundaries: rows_per_batch = 6, block_h = 4 (6 % 4 != 0)
        (2, 2, 96, 64, 2, 64, 3),
        # Single core holds multiple batches: rows_per_batch = 2 < block_h = 4
        (2, 1, 64, 64, 2, 64, 1),
        # Broadcast batch=1 causal mask where core crosses head/batch boundaries
        (2, 3, 64, 64, 1, 64, 3),
    ],
)
def test_sharded_causal_mask_cross_batch_and_head_boundaries(
    device, batch, heads, height, width, mask_batch, mask_height, num_cores
):
    """Sharded causal mask handles arbitrary block_h / mask_Ht / rows_per_batch ratios across cores."""
    torch.manual_seed(6)
    scale = 0.125
    x = torch.randn((batch, heads, height, width), dtype=torch.bfloat16)
    mask = torch.where(
        torch.rand((mask_batch, 1, mask_height, width)) > 0.65, torch.tensor(-1e4), torch.tensor(0.0)
    ).bfloat16()

    mem_config, prog_config = _height_sharded_config(batch, heads, height, width, num_cores=num_cores)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config)
    tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.scale_mask_softmax_in_place(
        tt_x, scale, tt_mask, program_config=prog_config, is_causal_mask=True
    )
    expanded_mask = mask.repeat(batch // mask_batch, heads, (height + mask_height - 1) // mask_height, 1)[
        :, :, :height, :
    ]
    ref = torch.softmax(x.float() * scale + expanded_mask.float(), dim=-1)
    assert_numeric_metrics(ref, ttnn.to_torch(out).float(), pcc_threshold=0.999, rtol=1e-2, atol=1e-3, frobenius_threshold=1e-2)


@pytest.mark.parametrize("mask_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize(
    "batch, heads, height, width, num_cores",
    [
        (2, 3, 64, 64, 3),  # rows_per_batch = 6, block_h = 4 (core 1 crosses batch 0 -> 1)
        (2, 1, 64, 64, 1),  # single core spans batches 0 and 1 (rows_per_batch = 2, block_h = 4)
    ],
)
def test_sharded_broadcast_mask_cross_batch_boundaries(device, mask_layout, batch, heads, height, width, num_cores):
    """Non-causal per-batch mask updates cleanly per row when shards cross batch boundaries."""
    torch.manual_seed(7)
    scale = 0.2
    x = torch.randn((batch, heads, height, width), dtype=torch.bfloat16)
    mask_1d = torch.where(torch.rand((batch, 1, 1, width)) > 0.6, torch.tensor(-1e4), torch.tensor(0.0)).bfloat16()

    if mask_layout == ttnn.ROW_MAJOR_LAYOUT:
        mask_data = mask_1d.reshape(batch, 1, width // 32, 32)
    else:
        mask_data = mask_1d.repeat(1, 1, 32, 1)

    mem_config, prog_config = _height_sharded_config(batch, heads, height, width, num_cores=num_cores)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config)
    tt_mask = ttnn.from_torch(mask_data, dtype=ttnn.bfloat16, layout=mask_layout, device=device)

    out = ttnn.scale_mask_softmax_in_place(
        tt_x, scale, tt_mask, program_config=prog_config, is_causal_mask=False
    )
    ref = torch.softmax(x.float() * scale + mask_1d.float(), dim=-1)
    assert_numeric_metrics(ref, ttnn.to_torch(out).float(), pcc_threshold=0.999, rtol=1e-2, atol=1e-3, frobenius_threshold=1e-2)
