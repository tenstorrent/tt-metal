# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gated full attention + its KV cache at real dims on the 8x4 mesh, random weights.

* test_attention_vs_ref          whole block: [q|gate] proj -> head split -> QK-norm -> partial RoPE ->
                                 causal ring SDPA -> sigmoid gate -> o_proj; shared cos/sin.
                                 (pattern: minimax_m3/tests/unit/test_attention_vs_ref.py)
* test_kv_cache_write_vs_ref     the cache contents after the block's write, read back and PCC'd
                                 against the reference's post-RoPE K / raw V (right slot and offset).
                                 (pattern: minimax_m3/tests/unit/test_kv_cache_write_vs_ref.py)
* test_attention_chunked_vs_ref  a 2-chunk sequence through the SAME module; the second chunk reads
                                 the first from the cache. (pattern: test_attention_chunked_vs_ref.py)
* test_kv_cache_gqa_sp_vs_ref    write + read-back for this model's cache shape (hd 256, 1 kv head per
                                 chip) with both one-shot and chunk periods, two layers, two users.
                                 (pattern: minimax_m3/tests/unit/test_kv_cache_gqa_sp_vs_ref.py)
"""

import pytest
import torch

import ttnn
from models.demos.qwen_3_8_27b.config import QWEN38, PrefillSpec, ttnn_dtype
from models.demos.qwen_3_8_27b.reference import qwen3_8_ref as ref
from models.demos.qwen_3_8_27b.tests.common import assert_pcc, from_sp, to_sp
from models.demos.qwen_3_8_27b.tt.attention import TtAttention
from models.demos.qwen_3_8_27b.tt.context import PrefillCtx
from models.demos.qwen_3_8_27b.tt.kv_cache import allocate_caches, cache_capacity, read_attn_kv, write_kv_chunk
from models.demos.qwen_3_8_27b.tt.model import build_cache_mask
from models.demos.qwen_3_8_27b.tt.rope import TtRope

CHUNK = 5120
ATTN_ORDINAL = 1  # layer slot 1 of 2, so a wrong layer index reads the zeroed slot 0


@pytest.fixture(scope="module")
def attn_pair(mesh_config, ccl_manager):
    m = ref.init_random_(ref.Attention(QWEN38), seed=21).to(torch.bfloat16).eval()
    spec = PrefillSpec.load()
    tt = TtAttention(
        mesh_config,
        ccl_manager,
        QWEN38,
        m.state_dict(),
        attn_ordinal=ATTN_ORDINAL,
        weight_dtype=ttnn_dtype(spec.weight_dtype_attention),
    )
    return m, tt, TtRope(mesh_config, QWEN38)


@pytest.fixture(scope="module")
def caches(mesh):
    c = allocate_caches(
        mesh,
        num_attn_layers=2,
        gdn_layers=[],
        max_seq_len=cache_capacity(20480, [CHUNK, 10240]),
        head_dim=QWEN38.head_dim,
    )
    yield c
    ttnn.deallocate(c.k)
    ttnn.deallocate(c.v)


def _x(T, seed):
    return torch.randn(1, T, QWEN38.hidden_size, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


def _run(tt, rope, mesh_config, caches, x, start, total_valid_end=None):
    T = x.shape[1]
    ctx = PrefillCtx(caches=caches, user_id=0, start=start, valid_end=total_valid_end or start + T, tokens=T)
    ctx.cos, ctx.sin = rope.tables(start, T // mesh_config.sp)
    if start > 0:
        ctx.cache_mask = build_cache_mask(mesh_config, start, T)
    return from_sp(tt(to_sp(x[None], mesh_config), ctx, rope), mesh_config)[0]


@pytest.mark.parametrize("seq", [2048, 10240], ids=["2k", "10k"])
def test_attention_vs_ref(mesh_config, attn_pair, caches, seq):
    m, tt, rope = attn_pair
    x = _x(seq, 1)
    cos, sin = ref.rope_cos_sin(QWEN38, torch.arange(seq))
    with torch.no_grad():
        want, wk, wv = m(x, cos, sin)
    got = _run(tt, rope, mesh_config, caches, x, 0)
    assert_pcc(f"attention_{seq}", got, want.float())
    # the same forward wrote the cache (one-shot => block-cyclic period = seq)
    k, v = read_attn_kv(caches, mesh_config.mesh_device, user_id=0, attn_ordinal=ATTN_ORDINAL, n_tokens=seq, period=seq)
    assert_pcc(f"kv_cache_write_k_{seq}", k, wk.float())
    assert_pcc(f"kv_cache_write_v_{seq}", v, wv.float())


@pytest.mark.parametrize("mode", ["masked", "ring"])
def test_attention_chunked_vs_ref(mesh_config, attn_pair, caches, mode, monkeypatch):
    """mode: how chunk 1 reads the cached prefix — the composed masked SDPA (default) or the ring cache-read op."""
    monkeypatch.setenv("QWEN38_CACHE_ATTN", mode)
    m, tt, rope = attn_pair
    x = _x(2 * CHUNK, 2)
    cos, sin = ref.rope_cos_sin(QWEN38, torch.arange(2 * CHUNK))
    with torch.no_grad():
        want, wk, wv = m(x, cos, sin)
    o0 = _run(tt, rope, mesh_config, caches, x[:, :CHUNK], 0)
    o1 = _run(tt, rope, mesh_config, caches, x[:, CHUNK:], CHUNK)
    assert_pcc(f"attention_chunked_{mode}_chunk0", o0, want[:, :CHUNK].float())
    assert_pcc(f"attention_chunked_{mode}_chunk1", o1, want[:, CHUNK:].float())
    k, v = read_attn_kv(
        caches, mesh_config.mesh_device, user_id=0, attn_ordinal=ATTN_ORDINAL, n_tokens=2 * CHUNK, period=CHUNK
    )
    assert_pcc("kv_cache_chunked_k", k, wk.float())
    assert_pcc("kv_cache_chunked_v", v, wv.float())


@pytest.mark.parametrize("period,n_chunks", [(CHUNK, 3), (10240, 1)], ids=["chunked_3x5120", "oneshot_10240"])
def test_kv_cache_gqa_sp_vs_ref(mesh, mesh_config, period, n_chunks):
    hd, nkv = QWEN38.head_dim, QWEN38.num_key_value_heads
    c = allocate_caches(
        mesh,
        num_attn_layers=2,
        gdn_layers=[],
        max_seq_len=cache_capacity(16384, [CHUNK, 10240]),
        head_dim=hd,
        num_users=2,
    )
    g = torch.Generator().manual_seed(5)
    S = period * n_chunks
    data = {
        (u, l): (torch.randn(1, nkv, S, hd, generator=g), torch.randn(1, nkv, S, hd, generator=g))
        for u in range(2)
        for l in range(2)
    }
    for (u, l), (k, v) in data.items():
        for ci in range(n_chunks):
            sl = slice(ci * period, (ci + 1) * period)
            write_kv_chunk(
                c,
                to_sp(k[:, :, sl], mesh_config, tp_dim=1),
                to_sp(v[:, :, sl], mesh_config, tp_dim=1),
                slot_idx=u,
                layer_idx=l,
                kv_actual=ci * period,
            )
    for (u, l), (k, v) in data.items():
        gk, gv = read_attn_kv(c, mesh, user_id=u, attn_ordinal=l, n_tokens=S, period=period)
        assert_pcc(f"kv_gqa_sp_{period}_u{u}_l{l}_k", gk, k)
        assert_pcc(f"kv_gqa_sp_{period}_u{u}_l{l}_v", gv, v)
    ttnn.deallocate(c.k)
    ttnn.deallocate(c.v)
