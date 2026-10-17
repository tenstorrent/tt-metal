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

# ttnn.experimental.quasar.transpose keeps its own copy of the sharded row-major WH factory and kernel, and
# selects them on Wormhole and Blackhole exactly like ttnn.transpose does, so it is held to the same checks.
TRANSPOSE_OPS = [
    pytest.param(ttnn.transpose, id="transpose"),
    pytest.param(ttnn.experimental.quasar.transpose, id="quasar_transpose"),
]

DTYPES = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}


def _height_sharded_rm_input(device, h, w, dtype, c=4):
    """[1, c, h, w] ROW_MAJOR tensor, height-sharded on one core so every H x W slab is in the shard.

    This selects the sharded row-major WH transpose factory, which packs the output rows (h datums wide when
    h < 32) through the narrow-row untilize. The native sharded factory is only chosen when the row-major
    shard holds a tile multiple (1024) of elements, hence c slabs per shard: (c*h) * w and the output's
    (c*w) * h are multiples of 1024 for every h used here.
    """
    torch.manual_seed(2005)
    torch_input = torch.rand((1, c, h, w), dtype=DTYPES[dtype])
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, (c * h, w), ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )
    return torch_input, tt_input


@skip_on_quasar
@pytest.mark.parametrize("transpose_op", TRANSPOSE_OPS)
@pytest.mark.parametrize(
    "dtype, h",
    [
        # bf16: 16- and 32-byte rows.
        (ttnn.bfloat16, 8),
        (ttnn.bfloat16, 16),
        # fp32: 16-, 32-, 48- and 64-byte rows; 12 and 16 datums are only 16-byte multiples at 4 bytes a datum.
        (ttnn.float32, 4),
        (ttnn.float32, 8),
        (ttnn.float32, 12),
        (ttnn.float32, 16),
    ],
)
@pytest.mark.parametrize("w", [64])
def test_transpose_hw_sharded_rm_narrow_row_aligned(device, transpose_op, dtype, h, w):
    """Narrow output rows up to one face wide whose byte width is a multiple of 16 transpose exactly."""
    if transpose_op is ttnn.experimental.quasar.transpose and dtype == ttnn.float32:
        # The fork's sharded row-major factory binds no unpack mode for its fp32 output buffer, so program-spec
        # validation rejects every fp32 input before the narrow-row path is reached; not a row-width matter.
        pytest.skip(
            "ttnn.experimental.quasar.transpose rejects fp32 sharded row-major input in program-spec validation"
        )
    torch_input, tt_input = _height_sharded_rm_input(device, h, w, dtype)
    tt_output = transpose_op(tt_input, 2, 3)
    tt_output = ttnn.to_torch(ttnn.from_device(ttnn.to_memory_config(tt_output, ttnn.L1_MEMORY_CONFIG)))
    assert torch.equal(torch_input.transpose(2, 3), tt_output)


@skip_on_quasar
@pytest.mark.parametrize("transpose_op", TRANSPOSE_OPS)
@pytest.mark.parametrize(
    "dtype, h, message",
    [
        # 8- and 24-byte rows cannot be packed at the 16-byte page pitch of the sharded output.
        (ttnn.bfloat16, 4, "must be a multiple of the 16-byte buffer alignment"),
        (ttnn.bfloat16, 12, "must be a multiple of the 16-byte buffer alignment"),
        # 40- and 48-byte rows are wider than one face (checked before the alignment).
        (ttnn.bfloat16, 20, "wider than one face"),
        (ttnn.bfloat16, 24, "wider than one face"),
        # Above one tile and not a multiple of it: every tile column is clipped to H % 32 datums.
        (ttnn.bfloat16, 36, "every tile column of the output row would be clipped to 4 datums"),
        (ttnn.bfloat16, 48, "every tile column of the output row would be clipped to 16 datums"),
        # fp32: 20 datums are 80 bytes, 16-byte aligned, and still wider than one face.
        (ttnn.float32, 20, "wider than one face"),
    ],
)
@pytest.mark.parametrize("w", [64])
def test_transpose_hw_sharded_rm_narrow_row_rejects_unsupported_heights(
    device, expect_error, transpose_op, dtype, h, message, w
):
    """Heights the narrow-row untilize packs wrongly (on Wormhole and Blackhole) are rejected on the host."""
    _, tt_input = _height_sharded_rm_input(device, h, w, dtype)
    with expect_error(RuntimeError, message):
        transpose_op(tt_input, 2, 3)
