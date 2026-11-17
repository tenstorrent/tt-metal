# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only: the Hy4 weight reader's RoPE frame reorders (``tt/runners/hy4_checkpoint.py``).

Hy4's HF attention ropes q_pe / k_pe half-split, straight off the projection, and its indexer ropes the LAST
``rope`` dims of each index key / query half-split. ttMLA ropes interleaved pairs; TtIndexer ropes the FIRST
``rope`` dims interleaved. The reader permutes projection rows so the device, with its own rope, computes the
same attention. These tests run both pipelines on random weights and check that the scores agree and that
the device caches equal the HF ones mapped by the documented frame change.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.deepseek_v3_d_p.tt.runners import hy4_checkpoint as hc

THETA = 1e7  # Hy4's rope_theta


def _randn(*shape, generator):
    return torch.randn(*shape, generator=generator, dtype=torch.float64)


def _angles(n: int, d: int) -> torch.Tensor:
    inv = 1.0 / (THETA ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    return torch.outer(torch.arange(n, dtype=torch.float64) * 37, inv)  # spread positions over a long context


def rope_half_split(x: torch.Tensor) -> torch.Tensor:
    """HF ``rotate_half`` rope: pairs (i, i + d/2) at frequency i. x: [S, ..., d]."""
    d = x.shape[-1]
    a = _angles(x.shape[0], d).reshape(x.shape[0], *([1] * (x.dim() - 2)), d // 2)
    c, s = torch.cos(a), torch.sin(a)
    x0, x1 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)


def rope_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Device (Meta) rope: pairs (2i, 2i + 1) at frequency i."""
    d = x.shape[-1]
    a = _angles(x.shape[0], d).reshape(x.shape[0], *([1] * (x.dim() - 2)), d // 2)
    c, s = torch.cos(a), torch.sin(a)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


def _layer_norm(x, w, b):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b, eps=1e-6)


def test_permutations_are_the_documented_ones(expect_error):
    assert hc.half_split_to_interleaved_perm(8).tolist() == [0, 4, 1, 5, 2, 6, 3, 7]
    # HF [nope 4 | rope 4 (half-split)] -> device [rope 4 (interleaved) | nope 4]
    assert hc.index_k_device_perm(8, 4).tolist() == [4, 6, 5, 7, 0, 1, 2, 3]
    p = hc.index_k_device_perm(128, 64)
    assert sorted(p.tolist()) == list(range(128))
    assert p[:64].tolist() == (64 + hc.half_split_to_interleaved_perm(64)).tolist() and p[64:].tolist() == list(
        range(64)
    )
    with expect_error(ValueError, "must be even"):
        hc.half_split_to_interleaved_perm(7)
    with expect_error(ValueError, "index rope width"):
        hc.index_k_device_perm(8, 9)


@pytest.mark.parametrize("d", [8, 64])
def test_half_split_perm_maps_the_rope_frame(d):
    """Roping the permuted vector interleaved == permuting the half-split-roped vector, at every position."""
    g = torch.Generator().manual_seed(d)
    x = _randn(33, d, generator=g)
    p = hc.half_split_to_interleaved_perm(d)
    torch.testing.assert_close(rope_interleaved(x[:, p]), rope_half_split(x)[:, p])
    # ... and the un-permuted vector does NOT agree: the frame change is not a no-op.
    assert not torch.allclose(rope_interleaved(x), rope_half_split(x), atol=1e-3)


def test_mla_rope_reorder_preserves_scores_and_maps_the_kv_cache():
    S, hidden, q_lora, kv_lora, nope, rope, heads = 17, 48, 24, 16, 12, 8, 3
    g = torch.Generator().manual_seed(1)
    x = _randn(S, hidden, generator=g)
    cq = _randn(S, q_lora, generator=g)  # q_a latent (q_a_proj + norm are untouched by the reorder)
    w = {
        "kv_a_proj_with_mqa.weight": _randn(kv_lora + rope, hidden, generator=g),
        "q_b_proj.weight": _randn(heads * (nope + rope), q_lora, generator=g),
        "o_proj.weight": _randn(hidden, heads * 4, generator=g),
    }
    dev = hc.mla_rope_to_device_frame(w, kv_lora_rank=kv_lora, qk_nope_head_dim=nope, qk_rope_head_dim=rope)
    assert dev["o_proj.weight"] is w["o_proj.weight"]

    def attn(weights, rope_fn):
        kv = x @ weights["kv_a_proj_with_mqa.weight"].T
        latent, k_pe = kv[:, :kv_lora], rope_fn(kv[:, kv_lora:])
        q = (cq @ weights["q_b_proj.weight"].T).reshape(S, heads, nope + rope)
        q_nope, q_pe = q[..., :nope], rope_fn(q[..., nope:])
        return latent, k_pe, q_nope, torch.einsum("shd,td->hst", q_pe, k_pe)

    hf_latent, hf_kpe, hf_qnope, hf_scores = attn(w, rope_half_split)
    d_latent, d_kpe, d_qnope, d_scores = attn(dev, rope_interleaved)
    torch.testing.assert_close(d_latent, hf_latent)  # the KVPE nope half is untouched
    torch.testing.assert_close(d_qnope, hf_qnope)
    torch.testing.assert_close(d_scores, hf_scores)  # q_pe . k_pe at every (head, query, key)
    # The device KVPE pe half == the HF one re-based to interleaved (utils.test_utils.interleave_pe): how the
    # producer / chunked driver compare the cache against the golden.
    torch.testing.assert_close(d_kpe, hf_kpe[:, hc.half_split_to_interleaved_perm(rope)])
    # Without the reorder the device would score something else entirely.
    _, _, _, wrong = attn(w, rope_interleaved)
    assert not torch.allclose(wrong, hf_scores, atol=1e-3)


@pytest.mark.parametrize("dims", [(8, 4), (128, 64)], ids=["tiny", "hy4"])
def test_indexer_reorder_preserves_scores_and_maps_the_index_keys(dims):
    D, rope = dims
    nope = D - rope
    S, hidden, q_lora, heads = 19, 40, 24, 4
    g = torch.Generator().manual_seed(D)
    x = _randn(S, hidden, generator=g)
    cq = _randn(S, q_lora, generator=g)
    w = {
        "indexer.wk.weight": _randn(D, hidden, generator=g),
        "indexer.k_norm.weight": _randn(D, generator=g),
        "indexer.k_norm_bias.weight": _randn(D, generator=g),
        "indexer.wq_b.weight": _randn(heads * D, q_lora, generator=g),
        "indexer.weights_proj.weight": _randn(heads, hidden, generator=g),
    }
    dev = hc.indexer_to_device_frame(w, index_head_dim=D, rope_dim=rope)
    assert dev["indexer.weights_proj.weight"] is w["indexer.weights_proj.weight"]

    def hf(weights):  # Hy4: [nope | rope], rope on the LAST `rope` dims, half-split
        k = _layer_norm(x @ weights["indexer.wk.weight"].T, w["indexer.k_norm.weight"], w["indexer.k_norm_bias.weight"])
        q = (cq @ weights["indexer.wq_b.weight"].T).reshape(S, heads, D)
        k = torch.cat([k[:, :nope], rope_half_split(k[:, nope:])], dim=-1)
        q = torch.cat([q[..., :nope], rope_half_split(q[..., nope:])], dim=-1)
        return k, q

    def device(weights):  # TtIndexer: [rope | nope], rope on the FIRST `rope` dims, interleaved
        k = x @ weights["indexer.wk.weight"].T
        k = _layer_norm(k, weights["indexer.k_norm.weight"], weights["indexer.k_norm_bias.weight"])
        q = (cq @ weights["indexer.wq_b.weight"].T).reshape(S, heads, D)
        k = torch.cat([rope_interleaved(k[:, :rope]), k[:, rope:]], dim=-1)
        q = torch.cat([rope_interleaved(q[..., :rope]), q[..., rope:]], dim=-1)
        return k, q

    hf_k, hf_q = hf(w)
    d_k, d_q = device(dev)
    perm = hc.index_k_device_perm(D, rope)
    torch.testing.assert_close(d_k, hf_k[:, perm])  # the golden stores index keys in this device frame
    hf_scores = torch.einsum("shd,td->hst", hf_q, hf_k)
    torch.testing.assert_close(torch.einsum("shd,td->hst", d_q, d_k), hf_scores)
    # TtIndexer also rotates q and k by the same orthonormal Hadamard; scores stay equal.
    hq, _ = torch.linalg.qr(_randn(D, D, generator=g))
    torch.testing.assert_close(torch.einsum("shd,td->hst", d_q @ hq, d_k @ hq), hf_scores)
    d_k_unperm, _ = device(w)
    assert not torch.allclose(d_k_unperm, hf_k[:, perm], atol=1e-3)


class _FakeCkpt:
    def __init__(self, tensors):
        self.t = tensors

    def _read(self, key):
        return self.t[key].to(torch.bfloat16)


class _Cfg:
    kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, index_head_dim = 16, 12, 8, 8
    index_rope_dim = 4
    index_rope_interleave = True


def _attn_tensors(prefix, owns_indexer, g):
    c, heads, hidden, q_lora = _Cfg, 2, 32, 16
    t = {
        "q_a_proj.weight": _randn(q_lora, hidden, generator=g),
        "q_a_layernorm.weight": _randn(q_lora, generator=g),
        "q_b_proj.weight": _randn(heads * (c.qk_nope_head_dim + c.qk_rope_head_dim), q_lora, generator=g),
        "kv_a_proj_with_mqa.weight": _randn(c.kv_lora_rank + c.qk_rope_head_dim, hidden, generator=g),
        "kv_a_layernorm.weight": _randn(c.kv_lora_rank, generator=g),
        "kv_b_proj.weight": _randn(heads * 20, c.kv_lora_rank, generator=g),
        "o_proj.weight": _randn(hidden, heads * 8, generator=g),
    }
    if owns_indexer:
        t.update(
            {
                "indexer.wq_b.weight": _randn(3 * c.index_head_dim, q_lora, generator=g),
                "indexer.wk.weight": _randn(c.index_head_dim, hidden, generator=g),
                "indexer.k_norm.weight": _randn(c.index_head_dim, generator=g),
                "indexer.k_norm.bias": _randn(c.index_head_dim, generator=g),
                "indexer.weights_proj.weight": _randn(3, hidden, generator=g),
            }
        )
    return {f"{prefix}.self_attn.{k}": v for k, v in t.items()}


@pytest.mark.parametrize("owns_indexer", [True, False], ids=["full", "shared"])
def test_hy4_mla_weights_applies_both_reorders_unconditionally(owns_indexer):
    g = torch.Generator().manual_seed(3)
    prefix = "model.layers.0"
    raw = _attn_tensors(prefix, owns_indexer, g)
    out = hc.hy4_mla_weights(_FakeCkpt(raw), prefix, owns_indexer=owns_indexer, gated=False, config=_Cfg)

    def src(k):
        return raw[f"{prefix}.self_attn.{k}"].to(torch.bfloat16)

    rp = hc.half_split_to_interleaved_perm(_Cfg.qk_rope_head_dim)
    kv = src("kv_a_proj_with_mqa.weight")
    L = _Cfg.kv_lora_rank
    assert torch.equal(out["kv_a_proj_with_mqa.weight"], torch.cat([kv[:L], kv[L:][rp]]))
    N = _Cfg.qk_nope_head_dim
    qb = src("q_b_proj.weight").reshape(2, N + _Cfg.qk_rope_head_dim, -1)
    want_qb = torch.cat([qb[:, :N], qb[:, N:][:, rp]], dim=1).reshape(-1, qb.shape[-1])
    assert torch.equal(out["q_b_proj.weight"], want_qb)
    for k in ("q_a_proj.weight", "kv_b_proj.weight", "o_proj.weight"):
        assert torch.equal(out[k], src(k)), k
    if not owns_indexer:
        assert not any(k.startswith("indexer.") for k in out)
        return
    perm = hc.index_k_device_perm(_Cfg.index_head_dim, _Cfg.index_rope_dim)
    assert torch.equal(out["indexer.wk.weight"], src("indexer.wk.weight")[perm])
    assert torch.equal(out["indexer.k_norm.weight"], src("indexer.k_norm.weight").float()[perm])
    assert torch.equal(out["indexer.k_norm_bias.weight"], src("indexer.k_norm.bias").float()[perm])
    wq = src("indexer.wq_b.weight").reshape(3, _Cfg.index_head_dim, -1)
    assert torch.equal(out["indexer.wq_b.weight"], wq[:, perm].reshape(3 * _Cfg.index_head_dim, -1))
    assert out["indexer.k_norm.weight"].dtype == torch.float32  # dtypes as before the reorder
    assert out["q_b_proj.weight"].dtype == torch.bfloat16


def test_reorder_rejects_mismatched_dims(expect_error):
    w = {"kv_a_proj_with_mqa.weight": torch.zeros(10, 4), "q_b_proj.weight": torch.zeros(12, 4)}
    with expect_error(ValueError, "kv_a_proj_with_mqa"):
        hc.mla_rope_to_device_frame(w, kv_lora_rank=4, qk_nope_head_dim=2, qk_rope_head_dim=4)
    with expect_error(ValueError, "head width"):
        hc.mla_rope_to_device_frame(w, kv_lora_rank=6, qk_nope_head_dim=1, qk_rope_head_dim=4)


def test_mla_cache_format_stamp_marks_old_caches_stale(tmp_path):
    def touch(name):
        (tmp_path / name).touch()

    for i in (0, 1, 2):
        touch(f"layer_{i}.attn_norm_weight_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin")
    for i in (0, 1):  # layers 0, 1 have an MLA (+ indexer) cache from an older reader; layer 2 has none
        touch(f"layer_{i}.mla.kv_a_proj_with_mqa_dtype_BFLOAT16_layout_TILE.tensorbin")
        touch(f"layer_{i}.mla_repl.wkv_b1_dtype_BFLOAT16_layout_TILE.tensorbin")
        touch(f"layer_{i}.mla.indexer_wk_dtype_BFLOAT16_layout_TILE.tensorbin")
    touch("layer_10.mla.kv_a_proj_with_mqa_dtype_BFLOAT16_layout_TILE.tensorbin")  # not layer 1

    assert hc.stale_mla_cache_layers(tmp_path, range(3)) == [0, 1]
    assert hc.purge_mla_cache(tmp_path, [0]) == 3
    assert not any(tmp_path.glob("layer_0.mla*")) and any(tmp_path.glob("layer_0.attn_norm*"))
    assert hc.stale_mla_cache_layers(tmp_path, range(3)) == [1]
    # A rebuilt layer is stamped; a layer with no MLA files is not.
    touch("layer_0.mla.kv_a_proj_with_mqa_dtype_BFLOAT16_layout_TILE.tensorbin")
    assert hc.stamp_mla_cache(tmp_path, [0, 2]) == [0]
    assert hc.mla_cache_stamp(tmp_path, 0).exists() and not hc.mla_cache_stamp(tmp_path, 2).exists()
    assert hc.stale_mla_cache_layers(tmp_path, range(3)) == [1]
    assert f"hy4_format_{hc.HY4_MLA_CACHE_FORMAT}" in hc.mla_cache_stamp(tmp_path, 0).name
    assert hc.stale_mla_cache_layers(tmp_path / "absent", range(3)) == []
