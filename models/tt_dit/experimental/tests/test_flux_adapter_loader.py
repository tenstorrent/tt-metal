# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the FLUX LoRA adapter loader.

The load path itself uploads deltas to a device, but the *hard* part — the
fused-QKV A/B stacking + head-interleave — is pure host tensor math. These
tests validate it against the ground truth: fusing the per-projection LoRA
deltas into the base weight and then merging Q/K/V exactly the way
``blocks.attention.Attention._reshape_and_merge_qkv`` does must equal the
single fused delta the loader builds for ``to_qkv``. No Tenstorrent device is
required.
"""
from __future__ import annotations

import torch

from models.tt_dit.experimental.lora.flux_adapter_loader import _SPATIAL_QKV, _register_fused, load_flux_adapter_into
from models.tt_dit.layers.lora import LoRAMixin


# --------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------
class _StubTP:
    def __init__(self, factor):
        self.tensor_parallel = type("tp", (), {"factor": factor})()


class _StubAttn:
    def __init__(self, *, n_dev, n_local_heads, head_dim):
        self.parallel_config = _StubTP(n_dev)
        self.n_local_heads = n_local_heads
        self.head_dim = head_dim


class _StubLinear(LoRAMixin):
    """Captures the (A, B, scale) that ``register_lora`` would receive."""

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.captured = None

    def register_lora(self, A, B, scale=1.0, name=""):
        self.captured = {"A": A, "B": B, "scale": scale, "name": name}
        return 0


def _merge_qkv_base(q, k, v, *, n_dev, n_local_heads, head_dim):
    """Reference: mirror Attention._reshape_and_merge_qkv (no head padding).

    Each input is a diffusers-layout weight/delta [out, in]. Returns the fused
    tt-layout weight [3*out, in] with the output dim head-interleaved.
    """

    def one(t):
        t = t.T  # [in, out]
        t = t.reshape(t.shape[0], n_dev, n_local_heads, head_dim)
        return t

    q, k, v = one(q), one(k), one(v)
    qkv = torch.cat([q, k, v], dim=2)  # [in, n_dev, 3*n_local_heads, head_dim]
    qkv = qkv.reshape(qkv.shape[0], 3 * n_dev * n_local_heads * head_dim)  # [in, 3*out]
    return qkv.T  # [3*out, in]


def test_fused_qkv_interleave_matches_base_merge():
    torch.manual_seed(0)
    n_dev, n_local_heads, head_dim = 2, 3, 8
    heads = n_dev * n_local_heads
    in_dim = heads * head_dim  # self-attn: query_dim == inner == out
    per_out = in_dim
    rank = 4
    lora_alpha = float(rank)  # → alpha/rank == 1
    lora_scale = 1.0

    # Per-projection adapter tensors (diffusers convention: A[r,in], B[out,r]).
    qkvs = {}
    deltas = {}
    for nm in _SPATIAL_QKV:
        A = torch.randn(rank, in_dim, dtype=torch.float64)
        B = torch.randn(per_out, rank, dtype=torch.float64)
        qkvs[nm] = {"A": A, "B": B}
        deltas[nm] = B @ A  # diffusers delta on weight [out, in] (alpha/rank == 1)

    attn = _StubAttn(n_dev=n_dev, n_local_heads=n_local_heads, head_dim=head_dim)
    target = _StubLinear(in_features=in_dim, out_features=3 * per_out)
    alphas = {nm: lora_alpha for nm in _SPATIAL_QKV}

    bank_idx, fused_rank = _register_fused(
        attn, target, "transformer_blocks.0.attn.to_qkv", _SPATIAL_QKV, qkvs, alphas, lora_scale, "t"
    )
    assert bank_idx == 0
    assert fused_rank == 3 * rank

    cap = target.captured
    A_fused, B_fused, eff_scale = cap["A"], cap["B"], cap["scale"]
    assert A_fused.shape == (3 * rank, in_dim)
    assert B_fused.shape == (3 * per_out, 3 * rank)
    assert eff_scale == lora_scale * (lora_alpha / rank)

    # Loader's fused delta in tt weight layout [3*out, in] = eff * B_fused @ A_fused.
    loader_delta = eff_scale * (B_fused.to(torch.float64) @ A_fused.to(torch.float64))

    # Ground truth: fuse each per-proj delta into a base weight, merge Q/K/V the
    # way the attention layer does, and take the difference. Base cancels, so we
    # can merge the deltas directly.
    ref_delta = _merge_qkv_base(
        deltas["to_q"], deltas["to_k"], deltas["to_v"], n_dev=n_dev, n_local_heads=n_local_heads, head_dim=head_dim
    )

    assert loader_delta.shape == ref_delta.shape == (3 * per_out, in_dim)
    torch.testing.assert_close(loader_delta, ref_delta, rtol=1e-9, atol=1e-9)


def test_head_padding_is_refused():
    """A fused output larger than heads*head_dim (head padding) must raise, not miswire."""
    n_dev, n_local_heads, head_dim = 1, 2, 8
    in_dim = n_dev * n_local_heads * head_dim
    rank = 2
    qkvs = {nm: {"A": torch.randn(rank, in_dim), "B": torch.randn(in_dim, rank)} for nm in _SPATIAL_QKV}
    attn = _StubAttn(n_dev=n_dev, n_local_heads=n_local_heads, head_dim=head_dim)
    # out_features implies a per-proj out of in_dim + 8 (i.e. one padded head) → mismatch.
    target = _StubLinear(in_features=in_dim, out_features=3 * (in_dim + head_dim))
    try:
        _register_fused(attn, target, "transformer_blocks.0.attn.to_qkv", _SPATIAL_QKV, qkvs, {}, 1.0, "t")
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError for head-padded fused-QKV LoRA")


# --------------------------------------------------------------------
# key routing (stub transformer records which module each key hits)
# --------------------------------------------------------------------
class _StubFF:
    def __init__(self, in_f, out_f):
        self.ff1 = _StubLinear(in_f, out_f)
        self.ff2 = _StubLinear(out_f, in_f)


class _StubAttnFull(_StubAttn):
    def __init__(self, *, n_dev, n_local_heads, head_dim, context=False):
        super().__init__(n_dev=n_dev, n_local_heads=n_local_heads, head_dim=head_dim)
        inner = n_dev * n_local_heads * head_dim
        self.to_qkv = _StubLinear(inner, 3 * inner)
        self.to_out = _StubLinear(inner, inner)
        if context:
            self.add_qkv_proj = _StubLinear(inner, 3 * inner)
            self.to_add_out = _StubLinear(inner, inner)


class _StubDoubleBlock:
    def __init__(self, **kw):
        inner = kw["n_dev"] * kw["n_local_heads"] * kw["head_dim"]
        self.attn = _StubAttnFull(context=True, **kw)
        self.ff = _StubFF(inner, 4 * inner)
        self.ff_context = _StubFF(inner, 4 * inner)


class _StubSingleBlock:
    def __init__(self, **kw):
        inner = kw["n_dev"] * kw["n_local_heads"] * kw["head_dim"]
        self.attn = _StubAttnFull(context=False, **kw)
        self.proj_mlp = _StubLinear(inner, 4 * inner)
        self.proj_out = _StubLinear(inner + 4 * inner, inner)


class _StubTransformer:
    def __init__(self, **kw):
        self.transformer_blocks = [_StubDoubleBlock(**kw)]
        self.single_transformer_blocks = [_StubSingleBlock(**kw)]


def _lora_ab(rank, in_f, out_f):
    return {"lora_A.weight": torch.randn(rank, in_f), "lora_B.weight": torch.randn(out_f, rank)}


def test_key_routing(tmp_path):
    from safetensors.torch import save_file

    kw = dict(n_dev=1, n_local_heads=4, head_dim=8)
    inner = kw["n_dev"] * kw["n_local_heads"] * kw["head_dim"]
    rank = 2

    state: dict[str, torch.Tensor] = {}

    def add(prefix, in_f, out_f):
        for k, v in _lora_ab(rank, in_f, out_f).items():
            state[f"transformer.{prefix}.{k}"] = v

    # double-stream block 0
    for p in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
        add(f"transformer_blocks.0.attn.{p}", inner, inner)
    add("transformer_blocks.0.attn.to_out.0", inner, inner)
    add("transformer_blocks.0.attn.to_add_out", inner, inner)
    add("transformer_blocks.0.ff.net.0.proj", inner, 4 * inner)
    add("transformer_blocks.0.ff.net.2", 4 * inner, inner)
    add("transformer_blocks.0.ff_context.net.0.proj", inner, 4 * inner)
    add("transformer_blocks.0.ff_context.net.2", 4 * inner, inner)
    # single-stream block 0
    for p in ("to_q", "to_k", "to_v"):
        add(f"single_transformer_blocks.0.attn.{p}", inner, inner)
    add("single_transformer_blocks.0.proj_mlp", inner, 4 * inner)
    # proj_out should be skipped (v0)
    add("single_transformer_blocks.0.proj_out", inner + 4 * inner, inner)

    path = tmp_path / "flux_lora.safetensors"
    save_file(state, str(path))

    tr = _StubTransformer(**kw)
    handle = load_flux_adapter_into(tr, str(path), scale=1.0, name="unit")

    expected = {
        "transformer_blocks.0.attn.to_qkv",
        "transformer_blocks.0.attn.add_qkv_proj",
        "transformer_blocks.0.attn.to_out",
        "transformer_blocks.0.attn.to_add_out",
        "transformer_blocks.0.ff.ff1",
        "transformer_blocks.0.ff.ff2",
        "transformer_blocks.0.ff_context.ff1",
        "transformer_blocks.0.ff_context.ff2",
        "single_transformer_blocks.0.attn.to_qkv",
        "single_transformer_blocks.0.proj_mlp",
    }
    assert set(handle.target_indices) == expected
    # proj_out must NOT be registered in v0
    assert "single_transformer_blocks.0.proj_out" not in handle.target_indices
