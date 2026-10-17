# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

# Quasar has no narrow-row pack untilize (the compute API rejects narrow_row statically), so the sharded
# row-major transpose of a non-tile-multiple height cannot run there.
skip_on_quasar = pytest.mark.skipif(
    "quasar" in ttnn.get_arch_name(), reason="Quasar does not support narrow-row pack_untilize"
)


def _height_sharded_rm_input(device, h, w, c=4):
    """[1, c, h, w] bf16 ROW_MAJOR tensor, height-sharded on one core so every H x W slab is in the shard.

    This selects the sharded row-major WH transpose factory, which packs the output rows (h datums wide when
    h < 32) through the narrow-row untilize. The native sharded factory is only chosen when the row-major
    shard holds a tile multiple (1024) of elements, hence c slabs per shard: (c*h) * w and the output's
    (c*w) * h are multiples of 1024 for every h used here.
    """
    torch.manual_seed(2005)
    torch_input = torch.rand((1, c, h, w), dtype=torch.bfloat16)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, (c * h, w), ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )
    return torch_input, tt_input


@skip_on_quasar
@pytest.mark.parametrize("h", [8, 16])
@pytest.mark.parametrize("w", [64])
def test_transpose_hw_sharded_rm_narrow_row_aligned(device, h, w):
    """Narrow output rows up to one face wide whose bf16 byte width is a multiple of 16 transpose exactly."""
    torch_input, tt_input = _height_sharded_rm_input(device, h, w)
    tt_output = ttnn.transpose(tt_input, 2, 3)
    tt_output = ttnn.to_torch(ttnn.from_device(ttnn.to_memory_config(tt_output, ttnn.L1_MEMORY_CONFIG)))
    assert torch.equal(torch_input.transpose(2, 3), tt_output)


@skip_on_quasar
@pytest.mark.parametrize(
    "h, message",
    [
        # 8- and 24-byte rows cannot be packed at the 16-byte page pitch of the sharded output.
        (4, "must be a multiple of the 16-byte buffer alignment"),
        (12, "must be a multiple of the 16-byte buffer alignment"),
        # 40- and 48-byte rows are wider than one face (checked before the alignment).
        (20, "wider than one face"),
        (24, "wider than one face"),
        # Above one tile and not a multiple of it: every tile column is clipped to H % 32 datums.
        (36, "every tile column of the output row would be clipped to 4 datums"),
        (48, "every tile column of the output row would be clipped to 16 datums"),
    ],
)
@pytest.mark.parametrize("w", [64])
def test_transpose_hw_sharded_rm_narrow_row_rejects_unsupported_heights(device, expect_error, h, message, w):
    """Heights the narrow-row untilize packs wrongly (on Wormhole and Blackhole) are rejected on the host."""
    _, tt_input = _height_sharded_rm_input(device, h, w)
    with expect_error(RuntimeError, message):
        ttnn.transpose(tt_input, 2, 3)
