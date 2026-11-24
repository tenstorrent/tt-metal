# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Interleaved TILE inputs padded past the next tile boundary must be rejected.

Each op below sizes its output from the logical shape but counts its work from the
padded shape. A whole extra tile of padding makes the two disagree, so the op either
returns wrong values or writes past its output. The aligned control keeps every
logical dimension rounded to the next 32 and must keep working.
"""

import pytest
import torch

import ttnn


def _tile(device, t, padded_shape):
    return ttnn.tilize_with_val_padding(ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device), padded_shape, 7.0)


def _put(device, t, **kwargs):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, **kwargs)


def _matmul_auto(device):
    memory_config = ttnn.create_sharded_memory_config(
        [32, 32],
        ttnn.CoreGrid(y=1, x=4),
        ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=True,
    )
    a = _tile(device, torch.full((1, 1, 32, 64), 1.0), [1, 1, 32, 96])
    b = _tile(device, torch.full((1, 1, 64, 128), 1.0), [1, 1, 96, 128])
    return ttnn.matmul(a, b, memory_config=memory_config)


def _interleaved_to_sharded(device):
    t = torch.arange(1056.0).reshape(1, 1, 33, 32) % 127 + 1
    memory_config = ttnn.create_sharded_memory_config(
        [64, 32],
        ttnn.CoreGrid(y=1, x=1),
        ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.interleaved_to_sharded(
        _tile(device, t, [1, 1, 96, 32]),
        memory_config,
        preallocated_output=_put(device, 0 * t, memory_config=memory_config),
    )


def _transpose(device):
    t = torch.cat([torch.full((1, 1, 32, 32), 1.0), torch.full((1, 1, 32, 32), 2.0)], dim=1)
    return ttnn.transpose(_tile(device, t, [1, 2, 64, 32]), 1, 2)


def _typecast(device):
    t = torch.arange(4096.0).reshape(1, 1, 64, 64) % 251 - 125
    return ttnn.typecast(_tile(device, t, [1, 1, 64, 512]), ttnn.float32)


def _nlp_create_qkv_heads(device):
    t = torch.cat([torch.full((1, 1, 32, 32), h + 1.0) for h in range(6)], dim=3)
    return ttnn.experimental.nlp_create_qkv_heads(
        _tile(device, t, [1, 1, 64, 192]),
        num_heads=2,
        num_kv_heads=2,
        transpose_k_heads=False,
    )


OPS = {
    "matmul": _matmul_auto,
    "interleaved_to_sharded": _interleaved_to_sharded,
    "transpose": _transpose,
    "typecast": _typecast,
    "nlp_create_qkv_heads": _nlp_create_qkv_heads,
}


@pytest.mark.parametrize("op", list(OPS), ids=list(OPS))
def test_overpadded_tile_input_rejected(device, expect_error, op):
    with expect_error(RuntimeError, "extra tile padding"):
        OPS[op](device)


def test_tile_aligned_inputs_pass(device):
    num_dram = device.dram_grid_size().x
    n = 32 * num_dram * (1 + num_dram % 2)
    a = _tile(device, torch.full((1, 1, 32, 32), 1.0), [1, 1, 32, 32])
    b = _put(device, torch.full((1, 1, 32, n), 1.0))
    out = _put(device, torch.full((1, 1, 32, n), 0.0))
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(2, 2),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=n // 64,
        per_core_M=1,
        per_core_N=n // 64,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    result = ttnn.matmul(a, b, program_config=program_config, optional_output_tensor=out)
    assert torch.equal(ttnn.to_torch(result).float(), torch.full((1, 1, 32, n), 32.0))

    t = torch.arange(1056.0).reshape(1, 1, 33, 32) % 127 + 1
    memory_config = ttnn.create_sharded_memory_config(
        [64, 32],
        ttnn.CoreGrid(y=1, x=1),
        ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    out = ttnn.interleaved_to_sharded(
        _tile(device, t, [1, 1, 64, 32]),
        memory_config,
        preallocated_output=_put(device, 0 * t, memory_config=memory_config),
    )
    assert torch.equal(ttnn.to_torch(out), t.to(torch.bfloat16))

    t = torch.cat([torch.full((1, 1, 32, 32), 1.0), torch.full((1, 1, 32, 32), 2.0)], dim=1)
    out = ttnn.transpose(_put(device, t), 1, 2)
    assert torch.equal(ttnn.to_torch(out), t.to(torch.bfloat16).transpose(1, 2))

    t = torch.arange(4096.0).reshape(1, 1, 64, 64) % 251 - 125
    out = ttnn.typecast(_tile(device, t, [1, 1, 64, 64]), ttnn.float32)
    assert torch.equal(ttnn.to_torch(out), t.float())

    t = torch.cat([torch.full((1, 1, 32, 32), h + 1.0) for h in range(6)], dim=3)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        _tile(device, t, [1, 1, 32, 192]),
        num_heads=2,
        num_kv_heads=2,
        transpose_k_heads=False,
    )
    for index, got in enumerate((q, k, v)):
        ref = t[..., 64 * index : 64 * index + 64].reshape(1, 32, 2, 32).transpose(1, 2)
        assert torch.equal(ttnn.to_torch(got), ref.to(torch.bfloat16))
