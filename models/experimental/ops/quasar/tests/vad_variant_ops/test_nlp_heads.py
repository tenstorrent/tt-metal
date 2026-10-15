# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.nlp_create_qkv_heads`` / ``nlp_concat_heads``.

Shape source (kernel cache): ``reader/writer_tm_tile_layout_nlp_create_qkv_heads`` ct
[q_tiles, kv_tiles, ...] and writer [q_Ht, q_Wt, ..., nq=8, nkv=8, ...]; the ``transpose_wh``
compute kernel is only built for ``transpose_k_heads=True``
(``nlp_create_qkv_heads_program_factory.cpp:94-117``) [H]. 8 heads x 32 head_dim on a fused
[B, 1, S, 768] qkv (L1), and a variant with separate q [B, 1, S, 256] (DRAM) + kv
[B, 1, S, 512] (L1). S <= 32 and B <= 20 [M]; S = 32 and B in {1, 6} used.
``nlp_concat_heads`` ct [Ht, Wt, heads=8, ...] on [B, 8, S, 32], input L1 or DRAM [M].
Pure data movement: checked bit-exact.
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

E = U.EMBED

H, D = U.NUM_HEADS, U.HEAD_DIM


def _split_heads(x):
    """[B, 1, S, H*D] -> [B, H, S, D]"""
    b, _, s, _ = x.shape
    return x.reshape(b, s, H, D).permute(0, 2, 1, 3).contiguous()


@U.with_default_mesh()
@pytest.mark.parametrize("batch", [1, U.NUM_CAMS], ids=lambda b: f"b{b}")
@pytest.mark.parametrize("seq", [32])
def test_nlp_create_qkv_heads_fused(ttnn_mesh_device, reset_seeds, batch, seq):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand((batch, 1, seq, 3 * E))
    x = U.to_tt_mem(x_torch, mesh, "l1")
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        x, num_heads=H, num_kv_heads=H, transpose_k_heads=True, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    U.assert_lossless(_split_heads(x_torch[..., :E]), q, mesh_device=mesh)
    U.assert_lossless(_split_heads(x_torch[..., E : 2 * E]).transpose(-2, -1).contiguous(), k, mesh_device=mesh)
    U.assert_lossless(_split_heads(x_torch[..., 2 * E :]), v, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize("batch", [1, U.NUM_CAMS], ids=lambda b: f"b{b}")
@pytest.mark.parametrize("seq", [32])
def test_nlp_create_qkv_heads_separate_kv(ttnn_mesh_device, reset_seeds, batch, seq):
    mesh = ttnn_mesh_device
    q_torch = U.torch_rand((batch, 1, seq, E))
    kv_torch = U.torch_rand((batch, 1, seq, 2 * E))
    q_in = U.to_tt_mem(q_torch, mesh, "dram")
    kv_in = U.to_tt_mem(kv_torch, mesh, "l1")
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        q_in, kv_in, num_heads=H, num_kv_heads=H, transpose_k_heads=True, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    U.assert_lossless(_split_heads(q_torch), q, mesh_device=mesh)
    U.assert_lossless(_split_heads(kv_torch[..., :E]).transpose(-2, -1).contiguous(), k, mesh_device=mesh)
    U.assert_lossless(_split_heads(kv_torch[..., E:]), v, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize("mem", ["l1", "dram"])
@pytest.mark.parametrize("batch", [1, U.NUM_CAMS], ids=lambda b: f"b{b}")
def test_nlp_concat_heads(ttnn_mesh_device, reset_seeds, mem, batch):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand((batch, H, 32, D))
    x = U.to_tt_mem(x_torch, mesh, mem)
    out = ttnn.experimental.nlp_concat_heads(x, memory_config=U.resolve_memcfg(mem, mesh))
    ref = x_torch.permute(0, 2, 1, 3).reshape(batch, 1, 32, H * D)
    U.assert_lossless(ref, out, mesh_device=mesh)
