# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

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


def _height_sharded_rm_input(device, h, w, dtype, c=None):
    """[1, c, h, w] ROW_MAJOR tensor, height-sharded on one core so every H x W slab is in the shard.

    This selects the sharded row-major WH transpose factory, which packs the output rows (h datums wide when
    h < 32) through the narrow-row untilize. The native sharded factory is only chosen when the row-major
    shard holds a tile multiple (1024) of elements, hence c slabs per shard: c defaults to the smallest count
    (at least 4, so several slabs are always transposed) for which (c*h) * w, and so the output's (c*w) * h, is
    a multiple of 1024.
    """
    if c is None:
        c = max(4, 1024 // math.gcd(1024, h * w))
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
    "dtype, h, w",
    [
        # bf16: 16- and 32-byte rows over two and three tile columns (W a tile multiple: see the rejections below).
        (ttnn.bfloat16, 8, 64),
        (ttnn.bfloat16, 16, 64),
        (ttnn.bfloat16, 8, 96),
        (ttnn.bfloat16, 16, 96),
        # fp32: 16-, 32-, 48- and 64-byte rows; 12 and 16 datums are only 16-byte multiples at 4 bytes a datum.
        (ttnn.float32, 4, 64),
        (ttnn.float32, 8, 64),
        (ttnn.float32, 12, 64),
        (ttnn.float32, 16, 64),
    ],
)
def test_transpose_hw_sharded_rm_narrow_row_aligned(device, transpose_op, dtype, h, w):
    """Narrow output rows up to one face wide whose byte width is a multiple of 16 transpose exactly (W a tile
    multiple)."""
    if transpose_op is ttnn.experimental.quasar.transpose and dtype == ttnn.float32:
        # The fork's factory maps the compute kernel's output onto the output shard itself for H within one tile
        # (ht <= 8), binding that fp32 DFB as both producer and consumer of the compute kernel with no unpack_modes
        # entry; program-spec validation (ValidateUnpackModes: an fp32 DFB consumed with enable_32_bit_dest needs
        # an explicit entry) rejects it before the narrow-row path is reached. A fork defect, not a row-width one.
        pytest.skip(
            "ttnn.experimental.quasar.transpose binds its fp32 output shard as a consumed DFB without an unpack "
            "mode; program-spec validation rejects it before the narrow-row path"
        )
    torch_input, tt_input = _height_sharded_rm_input(device, h, w, dtype)
    tt_output = transpose_op(tt_input, 2, 3)
    tt_output = ttnn.to_torch(ttnn.from_device(ttnn.to_memory_config(tt_output, ttnn.L1_MEMORY_CONFIG)))
    assert torch.equal(torch_input.transpose(2, 3), tt_output)


@skip_on_quasar
@pytest.mark.parametrize("transpose_op", TRANSPOSE_OPS)
@pytest.mark.parametrize(
    "dtype, h, w, message",
    [
        # 8- and 24-byte rows cannot be packed at the 16-byte page pitch of the sharded output.
        (ttnn.bfloat16, 4, 64, "must be a multiple of the 16-byte buffer alignment"),
        (ttnn.bfloat16, 12, 64, "must be a multiple of the 16-byte buffer alignment"),
        # 40- and 48-byte rows are wider than one face (checked before the alignment).
        (ttnn.bfloat16, 20, 64, "wider than one face"),
        (ttnn.bfloat16, 24, 64, "wider than one face"),
        # Above one tile and not a multiple of it: every tile column is clipped to H % 32 datums.
        (ttnn.bfloat16, 36, 64, "every tile column of the output row would be clipped to 4 datums"),
        (ttnn.bfloat16, 48, 64, "every tile column of the output row would be clipped to 16 datums"),
        # fp32: 20 datums are 80 bytes, 16-byte aligned, and still wider than one face.
        (ttnn.float32, 20, 64, "wider than one face"),
        # W % 32 != 0: the last tile column is packed as 32 rows of H datums but paged as W % 32 rows, so every pack
        # of that column writes (32 - W % 32) * H * 2 bytes the pages do not cover; for the last slab of a shard
        # they land past the output buffer. The three shapes with W % 32 == H % 32 transpose exactly and still
        # write past the buffer (measured through a canary buffer allocated right above the output).
        (ttnn.bfloat16, 16, 40, "writes 768 bytes the output pages do not account for"),
        (ttnn.bfloat16, 8, 48, "writes 256 bytes the output pages do not account for"),
        (ttnn.bfloat16, 16, 8, "writes 768 bytes the output pages do not account for"),
        (ttnn.bfloat16, 8, 40, "writes 384 bytes the output pages do not account for"),
        (ttnn.bfloat16, 16, 48, "writes 512 bytes the output pages do not account for"),
        (ttnn.bfloat16, 16, 80, "writes 512 bytes the output pages do not account for"),
    ],
)
def test_transpose_hw_sharded_rm_narrow_row_rejects_unsupported_shapes(
    device, expect_error, transpose_op, dtype, h, w, message
):
    """Shapes the narrow-row untilize packs wrongly (on Wormhole and Blackhole) are rejected on the host."""
    _, tt_input = _height_sharded_rm_input(device, h, w, dtype)
    with expect_error(RuntimeError, message):
        transpose_op(tt_input, 2, 3)


@skip_on_quasar
@pytest.mark.parametrize("transpose_op", TRANSPOSE_OPS)
def test_transpose_hw_sharded_rm_narrow_row_rejects_fp32_partial_width(device, expect_error, transpose_op):
    """fp32 H 12, W 44: besides the surplus rows of the last column, a 4096-byte tile holds 2.67 pages of 32 x 12
    fp32 datums, so the factory advances the full columns by two 576-byte pages per 1536 bytes packed and the output
    comes out wrong. 192 slabs make both the input shard (a tile multiple of elements) and the output shard (a whole
    number of 576-byte pages) valid, so without the check the shape runs."""
    _, tt_input = _height_sharded_rm_input(device, 12, 44, ttnn.float32, c=192)
    with expect_error(RuntimeError, "writes 960 bytes the output pages do not account for"):
        transpose_op(tt_input, 2, 3)
