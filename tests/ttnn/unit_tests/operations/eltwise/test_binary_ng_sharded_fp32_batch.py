# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for issue 56958.

Sharded binary_ng FPU ops process several tiles per DST acquire. The batch size was a flat 8,
but under fp32 dest accumulation (any fp32 operand) DST holds only 4 tiles in half-sync mode.
The 8-tile batch wrapped past the active half; with mixed fp32/bf16 operands this corrupted
tile 4 of every batch, and tiles 0-3 when a LHS activation pass preceded the op. Same-dtype
inputs happened to survive the overspill, which is why it went unnoticed.

The fix bounds the batch by `fp32_dest_acc_en ? 4 : 8`, as every other fp32 op already does.

RHS-activation cases on mixed dtypes additionally depend on the SrcA format-state fix
(issue 56738) and are covered there.
"""

import pytest
import torch
import ttnn


def _sharded_inputs(device, shape, dtypes):
    memory = ttnn.create_sharded_memory_config(
        shape, ttnn.CoreGrid(y=1, x=1), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR
    )
    n = shape[-1] * shape[-2]
    tensors = [
        ttnn.from_torch(
            ((torch.arange(n) * 17 + offset) % 101 - 50).float().reshape(shape) / 8,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory,
        )
        for dtype, offset in zip(dtypes, (3, 13))
    ]
    return memory, tensors


def _run(device, op_name, dtypes, side, shape):
    memory, (a, b) = _sharded_inputs(device, shape, dtypes)
    ah, bh = (ttnn.to_torch(t).float() for t in (a, b))
    activation = [ttnn.UnaryWithParam(ttnn.UnaryOpType.ABS)]
    if side == "lhs":
        ah = ah.abs()
    out = getattr(ttnn, op_name)(
        a,
        b,
        dtype=ttnn.bfloat16,
        memory_config=memory,
        fast_and_approximate_mode=True,  # FPU kernel
        input_tensor_a_activations=activation if side == "lhs" else [],
    )
    expected = ah + bh if op_name == "add" else ah - bh
    got = ttnn.to_torch(out).float()
    bad_tiles = (got != expected).reshape(-1, 32, shape[-1] // 32, 32).any(dim=(1, 3)).flatten().nonzero().flatten()
    assert torch.equal(got, expected), f"{op_name} {dtypes} act={side} shape={shape}: wrong tiles {bad_tiles.tolist()}"


MIXED = [(ttnn.float32, ttnn.bfloat16), (ttnn.bfloat16, ttnn.float32)]


@pytest.mark.parametrize("op_name", ["add", "subtract"])
@pytest.mark.parametrize("dtypes", MIXED, ids=["f32_bf16", "bf16_f32"])
@pytest.mark.parametrize("side", ["none", "lhs"])
def test_sharded_mixed_dtype_fpu_batch(device, op_name, dtypes, side):
    # 9 tiles per shard: one full 8-tile batch (which used to straddle the 4-tile fp32 DST half)
    # plus a 1-tile remainder.
    _run(device, op_name, dtypes, side, shape=(1, 1, 96, 96))


@pytest.mark.parametrize("dtypes", MIXED + [(ttnn.float32, ttnn.float32)], ids=["f32_bf16", "bf16_f32", "f32_f32"])
@pytest.mark.parametrize("side", ["none", "lhs"])
@pytest.mark.parametrize("num_tiles", [4, 5, 8, 9, 12, 17])
def test_sharded_fp32_batch_boundaries(device, dtypes, side, num_tiles):
    # Sweep around the 4- and 8-tile boundaries so both the full-batch and remainder paths see
    # every batch size the factory can pick.
    _run(device, "add", dtypes, side, shape=(1, 1, 32 * num_tiles, 32))
