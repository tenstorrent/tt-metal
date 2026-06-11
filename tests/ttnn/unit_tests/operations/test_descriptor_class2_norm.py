# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Class-2 descriptor cache-hit fast-path tests for the slow-path-rebuild fix (#46506).

Covers the SHARED device ops whose sharded program factories were migrated to the
descriptor cache-hit FAST PATH (create_descriptor is NOT re-run on a cache hit):

  * Softmax  (SoftmaxShardedProgramFactoryAttentionOptimized)
  * LayerNorm (LayerNormShardedProgramFactory)
  * Bcast    (BcastShardedHProgramFactory / BcastShardedHOptimisedProgramFactory)

Each op declares get_dynamic_runtime_args(...). For the sharded norms/bcast every
per-dispatch tensor address is bound as a CB `.buffer` binding or a patchable
Buffer* runtime-arg binding, and all other args are hash/shape-derived, so the
op never rebuilds the descriptor on a cache hit (CB-bound-stable pattern).

NOTE: MorehAdam (FROZEN-ARGS: hash excludes lr/step) was DEFERRED from this PR. Its
get_dynamic_runtime_args fix is structurally sound (the guard confirms no rebuild),
but it cannot be numerically validated here because moreh_adam's param_out has a
pre-existing, unrelated compute-kernel bug (param_out is broadly uncorrelated with a
torch.optim.Adam reference even with the program cache disabled; exp_avg/exp_avg_sq
are correct). moreh_adam is left on the slow-path rebuild until that bug is fixed.

We set TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT=1 BEFORE importing ttnn:
if any of these ops rebuilds its descriptor on a cache hit, the adapter raises and
the test fails.
"""

import math
import os

os.environ["TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT"] = "1"

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def _run_thrice(fn, dev):
    """Run fn 3x with the SAME config (calls 2 and 3 are cache hits). Raises if the
    op rebuilds its descriptor on a cache hit (guard env var) or binds a Buffer* at
    a wrong arg slot (resolve_bindings validation on the first/miss call)."""
    out = None
    for _ in range(3):
        out = fn()
        ttnn.synchronize_device(dev)
    return out


# ---------------------------------------------------------------------------
# Softmax sharded (SoftmaxShardedProgramFactoryAttentionOptimized)
# Non-causal interleaved (L1) mask -> exercises the mask_addr reader rt-arg that
# get_dynamic_runtime_args re-applies on a cache hit.
# ---------------------------------------------------------------------------
def test_softmax_sharded_no_rebuild(device):
    torch.manual_seed(0)
    grid_size = (1, 8)
    seq_len = 64
    fuse_head = 768 // seq_len if 768 // seq_len > 0 else 1

    batch = grid_size[0]
    num_cores_r = grid_size[1]
    input_shape = (batch, num_cores_r * fuse_head, seq_len, seq_len)
    M = input_shape[2] * input_shape[1]
    K = input_shape[3] * batch

    hidden_dim = 1024
    num_heads = num_cores_r * fuse_head
    scale = 1 / math.sqrt(hidden_dim // num_heads)

    attention_mask = torch.rand(batch, 1, 1, seq_len)
    attention_mask = (attention_mask > 0.5).float()
    attention_mask = attention_mask.reshape(batch, 1, -1, 32)
    attention_mask_t = ttnn.Tensor(attention_mask, ttnn.bfloat16).to(
        device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    )

    input_tensor = torch.randn(input_shape).bfloat16().float()
    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    in1_t = ttnn.from_torch(
        input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem_config
    )
    in1_t_shard = ttnn.interleaved_to_sharded(
        in1_t,
        grid_size,
        [M // grid_size[1], K // grid_size[0]],
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.ShardOrientation.COL_MAJOR,
    )

    block_w = seq_len // 32
    block_h = seq_len // 32 * fuse_head
    subblock_w = 1
    for i in range(min(block_w // 2, 8), 0, -1):
        if block_w % i == 0:
            subblock_w = i
            break

    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
    )

    # Re-allocate the sharded input each call so its buffer address can change, and
    # run 3x: any descriptor rebuild on a cache hit raises under the guard.
    def _call():
        inp = ttnn.interleaved_to_sharded(
            in1_t,
            grid_size,
            [M // grid_size[1], K // grid_size[0]],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )
        return ttnn.scale_mask_softmax_in_place(
            inp, scale, attention_mask_t, program_config=program_config, is_causal_mask=False
        )

    tt_out = _run_thrice(_call, device)
    tt_result = ttnn.to_torch(tt_out).float()

    ref = input_tensor * scale + attention_mask.reshape(batch, 1, 1, seq_len).repeat(
        1, num_cores_r * fuse_head, seq_len, 1
    ).float().reshape(input_shape)
    ref = torch.softmax(ref, dim=-1)
    assert_with_pcc(ref, tt_result, 0.99)


# ---------------------------------------------------------------------------
# LayerNorm sharded (LayerNormShardedProgramFactory). LN_GB exercises the
# gamma/beta Buffer* writer rt-arg bindings added by this fix.
# ---------------------------------------------------------------------------
def test_layernorm_sharded_no_rebuild(device):
    torch.manual_seed(1234)
    grid_size = (1, 8)
    seq_len = 32
    per_core_k = 128

    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
    gamma_beta_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    epsf = 1e-2
    batch = grid_size[0]

    in0_shape = (batch, 1, seq_len, per_core_k * grid_size[1])
    M = in0_shape[2] * batch
    K = in0_shape[3]

    in0 = torch.rand(in0_shape) * 2 - 0.95
    in0_t = ttnn.from_torch(
        in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem_config
    )
    in0_t_shard = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.ShardOrientation.COL_MAJOR,
    )

    gamma = torch.rand(in0_shape[3]) * 2 - 1
    beta = torch.rand(in0_shape[3]) * 2.0 - 1.1
    # ROW_MAJOR gamma/beta must be reshaped so the last (padded) dim equals the tile width (32);
    # the sharded layernorm validate requires gamma.padded_shape[-1] == TILE_WIDTH.
    gamma_t = ttnn.from_torch(
        gamma.reshape(1, 1, -1, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=gamma_beta_mem_config,
    )
    beta_t = ttnn.from_torch(
        beta.reshape(1, 1, -1, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=gamma_beta_mem_config,
    )

    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=per_core_k // 32,
        block_h=seq_len // 32,
        block_w=per_core_k // 32,
        inplace=False,
    )

    def _call():
        inp = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[0], K // grid_size[1]],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
        )
        return ttnn.layer_norm(
            inp,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
        )

    ttz = _run_thrice(_call, device)
    ttz = ttnn.sharded_to_interleaved(ttz, in0_mem_config)
    tt_result = ttnn.to_torch(ttz).float()

    ref = torch.nn.functional.layer_norm(
        in0.float(), (in0_shape[3],), weight=gamma.float(), bias=beta.float(), eps=epsf
    )
    assert_with_pcc(ref, tt_result, 0.99)


# ---------------------------------------------------------------------------
# Bcast sharded dim=H (BcastShardedHOptimisedProgramFactory). src1 (b) bound as Buffer* rt-arg.
# Both sharded-H bcast factories only support BLOCK/WIDTH sharded inputs (they TT_THROW on
# HEIGHT_SHARDED), so we use a BLOCK_SHARDED input — matching the reference test_bcast.py — which
# routes to the optimised sharded-H factory whose src1 (b) address is re-applied on a cache hit.
# ---------------------------------------------------------------------------
def test_bcast_sharded_h_no_rebuild(device):
    torch.manual_seed(0)
    grid_size = (1, 8)
    H = 32 * grid_size[1]
    W = 64
    a_shape = (1, 1, H, W)
    b_shape = (1, 1, 1, W)

    a_torch = torch.rand(a_shape, dtype=torch.bfloat16)
    b_torch = torch.rand(b_shape, dtype=torch.bfloat16)

    in_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    a_t = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mem)
    b_t = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mem)

    a_shard = ttnn.interleaved_to_sharded(
        a_t,
        grid_size,
        [H // grid_size[1], W // grid_size[0]],
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    def _call():
        inp = ttnn.interleaved_to_sharded(
            a_t,
            grid_size,
            [H // grid_size[1], W // grid_size[0]],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        return ttnn.bcast(inp, b_t, math_op=ttnn.BcastOpMath.ADD, dim=ttnn.BcastOpDim.H)

    tt_out = _run_thrice(_call, device)
    tt_result = ttnn.to_torch(tt_out).float()

    ref = a_torch.float() + b_torch.float()
    assert_with_pcc(ref, tt_result, 0.99)
