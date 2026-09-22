# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decoder suite rows 7 and 8: the KV cache write path and its read-back.

Recipe rows ``test_kv_cache_write_vs_ref.py`` ("cache contents after a write through the
production prefill seam, read back and PCC'd against the torch reference's K/V") and
``test_kv_cache_gqa_sp_vs_ref.py`` ("write **and** read-back for the model's own cache shape on
the chunked-KV substrate, at target SP x TP"). One file: both write through
``kv_cache.write_kv_chunk`` and read back through ``kv_cache.cache_row_index``, and the second
only adds the multi-chunk / multi-layer / multi-user dimensions to the first.

What the read-back actually proves is the **layout inverse**. The cache is block-cyclic and
``cache_row_index`` claims that at chunk-aligned offsets that reduces to a contiguous SP split.
If that claim is wrong the read comes back shuffled and PCC collapses — it will not merely
degrade — so these two tests are the guard on the assumption every other SP path in the package
rests on.

The cache is bfloat8_b (the spec's ``dataformats.kv_cache``), so the measured PCC includes the
cache's own quantisation. That is the number the README reports for the cache rows.
"""

import pytest
import torch

import ttnn
from models.demos.mistral_medium_3_5_128b.reference.modeling import (
    REF_DTYPE,
    MistralAttention,
    MistralYarnRotaryEmbedding,
    causal_mask,
)
from models.demos.mistral_medium_3_5_128b.tests.device_utils import MESH_SHAPE, assert_pcc
from models.demos.mistral_medium_3_5_128b.tests.device_utils import read_kv_slot as read_cache
from models.demos.mistral_medium_3_5_128b.tests.device_utils import shard_heads_seq
from models.demos.mistral_medium_3_5_128b.tt.attention.kv_cache import allocate_kv_cache, write_kv_chunk

NKV, HEAD_DIM = 8, 128
SP_AXIS = 0
SP, TP = MESH_SHAPE


def test_kv_cache_write_vs_ref(galaxy, cfg):
    """Write the torch reference's own K/V through the production seam and read it back.

    The K here is post-RoPE and the V raw, straight out of ``MistralAttention.forward`` — the same
    convention the golden trace stores — so this checks the seam against the tensors P1 will later
    PCC per layer, not against a synthetic stand-in.
    """
    seq = 2048
    torch.manual_seed(0)
    ref = MistralAttention(cfg, REF_DTYPE).eval()
    x = (torch.randn(1, seq, cfg.hidden_size) * 0.1).to(REF_DTYPE)
    cos, sin = MistralYarnRotaryEmbedding(cfg, REF_DTYPE)(torch.arange(seq, dtype=torch.int64)[None])
    with torch.no_grad():
        _, k_ref, v_ref = ref(x, cos, sin, causal_mask(seq, seq))  # [1, NKV, seq, head_dim]

    kv = allocate_kv_cache(
        galaxy, num_layers=1, max_seq_len=seq, chunk_size=seq, sp_axis=SP_AXIS, num_kv_heads=NKV, head_dim=HEAD_DIM
    )
    write_kv_chunk(
        kv,
        shard_heads_seq(galaxy, k_ref, dtype=ttnn.bfloat8_b),
        shard_heads_seq(galaxy, v_ref, dtype=ttnn.bfloat8_b),
        slot_idx=0,
        layer_idx=0,
        kv_actual=0,
        sp_axis=SP_AXIS,
    )
    ttnn.synchronize_device(galaxy)

    got_k = read_cache(galaxy, kv.k, slot=0, cache_global=kv.max_seq_len, chunk_size=seq)
    got_v = read_cache(galaxy, kv.v, slot=0, cache_global=kv.max_seq_len, chunk_size=seq)
    assert_pcc("kv_cache_write_k", k_ref, got_k)
    assert_pcc("kv_cache_write_v", v_ref, got_v)


@pytest.mark.parametrize("chunk_size", [1024], ids=["c1024"])
def test_kv_cache_gqa_sp_vs_ref(galaxy, chunk_size):
    """Multi-chunk, multi-layer, multi-user write and read-back at the model's own cache shape.

    Distinct random K/V per (user, layer) is the point: with two users and two layers, a write or
    read that ignores the ``user_id * num_layers + layer_idx`` fold lands in the wrong slot and the
    PCC drops instead of quietly reproducing slot 0. The final cross-slot check makes that
    explicit rather than trusting the four PCCs to have been measured on different data.
    """
    num_users, num_layers, n_chunks = 2, 2, 3
    total = n_chunks * chunk_size

    kv = allocate_kv_cache(
        galaxy,
        num_layers=num_layers,
        max_seq_len=total,
        chunk_size=chunk_size,
        sp_axis=SP_AXIS,
        num_users=num_users,
        num_kv_heads=NKV,
        head_dim=HEAD_DIM,
    )
    assert kv.n_kv_local == NKV // TP == 2, "8 KV heads over TP=4 must give 2 per chip"
    assert kv.max_seq_len == total

    torch.manual_seed(1)
    expected = {}
    for user in range(num_users):
        for layer in range(num_layers):
            k = (torch.randn(1, NKV, total, HEAD_DIM) * 0.1).to(REF_DTYPE)
            v = (torch.randn(1, NKV, total, HEAD_DIM) * 0.1).to(REF_DTYPE)
            expected[user, layer] = (k, v)
            for c in range(n_chunks):
                lo, hi = c * chunk_size, (c + 1) * chunk_size
                write_kv_chunk(
                    kv,
                    shard_heads_seq(galaxy, k[:, :, lo:hi], dtype=ttnn.bfloat8_b),
                    shard_heads_seq(galaxy, v[:, :, lo:hi], dtype=ttnn.bfloat8_b),
                    slot_idx=user,
                    layer_idx=layer,
                    kv_actual=lo,
                    sp_axis=SP_AXIS,
                )
    ttnn.synchronize_device(galaxy)

    got = {}
    for (user, layer), (k, v) in expected.items():
        slot = user * num_layers + layer
        got[user, layer] = (
            read_cache(galaxy, kv.k, slot=slot, cache_global=total, chunk_size=chunk_size),
            read_cache(galaxy, kv.v, slot=slot, cache_global=total, chunk_size=chunk_size),
        )
        assert_pcc(f"kv_cache_gqa_sp_k[u{user}l{layer}]", k, got[user, layer][0])
        assert_pcc(f"kv_cache_gqa_sp_v[u{user}l{layer}]", v, got[user, layer][1])

    # Every slot must hold something different, or the four PCCs above could all be slot 0.
    slots = list(got)
    for i, a in enumerate(slots):
        for b in slots[i + 1 :]:
            assert not torch.equal(got[a][0], got[b][0]), f"slots {a} and {b} read back identical K"
