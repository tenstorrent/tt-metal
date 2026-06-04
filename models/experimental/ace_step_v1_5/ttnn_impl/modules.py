"""DEPRECATED: legacy unit-test scaffolding — production E2E uses ``dit_decoder_core``.

``AdaLNZeroTTNN`` / ``TransformerBlockTTNN`` do not match HF ``AceStepDiTLayer`` (wrong norm,
modulation layout, gates, cross-attn). Prefer ``TtAceStepDiTCore`` and PCC tests under
``tests/test_pcc_dit_decoder_core.py`` et al.
"""

from __future__ import annotations

import math

from ._ttnn import get_ttnn
from .config import AceConfigTTNN
from .math_perf_env import ace_step_add_one, ace_step_nlp_concat_heads, ace_step_permute_kwargs, ace_step_reshape_kwargs

# TTNN SDPA requires TILE tensors whose logical head_dim equals the padded head_dim
# (see sdpa_device_operation.cpp: logical_shape[3] == legacy_shape[3]). Sub-tile
# head sizes (e.g. 16) otherwise stay logically 16 while tiles pad to 32.
_SDPA_HEAD_DIM_ALIGN = 32


def _sdpa_head_dim_tile_padding(d_head: int) -> int:
    return (int(d_head) + _SDPA_HEAD_DIM_ALIGN - 1) // _SDPA_HEAD_DIM_ALIGN * _SDPA_HEAD_DIM_ALIGN


def _require_ttnn():
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for ace_step_v1_5.ttnn_impl")
    return ttnn


class AdaLNZeroTTNN:
    """
    TTNN AdaLN-Zero.

    Contract:
    - x:    TTNN tensor shaped [B, 1, S, D]
    - cond: TTNN tensor shaped [B, 1, 1, C]
    """

    def __init__(self, cfg: AceConfigTTNN, *, mesh_device, dtype=None, weights=None):
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.d_model = int(cfg.d_model)
        self.eps = float(cfg.eps)
        self.dtype = dtype or ttnn.bfloat16

        if weights is None:
            raise ValueError("AdaLNZeroTTNN requires weights={'w','b'} from Torch module")

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
        self.w = ttnn.as_tensor(
            weights["w"],
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.b = ttnn.as_tensor(
            weights["b"].reshape(1, 1, 1, -1),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

    def __call__(self, x, cond):
        ttnn = self.ttnn
        x_norm = ttnn.layer_norm(x, epsilon=self.eps)
        gb = ttnn.linear(cond, self.w, bias=self.b, transpose_b=True)
        d2 = int(gb.shape[-1])
        d = d2 // 2
        gamma = ttnn.slice(gb, (0, 0, 0, 0), (int(gb.shape[0]), 1, 1, d))
        beta = ttnn.slice(gb, (0, 0, 0, d), (int(gb.shape[0]), 1, 1, d2))
        ttnn.deallocate(gb)

        gamma1 = ace_step_add_one(ttnn, gamma)
        ttnn.deallocate(gamma)
        y = ttnn.multiply(x_norm, gamma1)
        ttnn.deallocate(x_norm)
        ttnn.deallocate(gamma1)
        y = ttnn.add(y, beta)
        ttnn.deallocate(beta)
        return y


class GEGLUMLPTTNN:
    """
    TTNN GEGLU MLP.

    Contract:
    - x: [B, 1, S, D]
    """

    def __init__(self, cfg: AceConfigTTNN, *, mesh_device, dtype=None, weights=None):
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.d_model = int(cfg.d_model)
        self.d_ff = int(cfg.d_ff)
        self.dtype = dtype or ttnn.bfloat16

        if weights is None:
            raise ValueError("GEGLUMLPTTNN requires weights from Torch module")

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None

        self.w_up = ttnn.as_tensor(
            weights["w_up"],
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.b_up = ttnn.as_tensor(
            weights["b_up"].reshape(1, 1, 1, -1),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.w_down = ttnn.as_tensor(
            weights["w_down"],
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.b_down = ttnn.as_tensor(
            weights["b_down"].reshape(1, 1, 1, -1),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

    def __call__(self, x):
        ttnn = self.ttnn
        up = ttnn.linear(x, self.w_up, bias=self.b_up, transpose_b=True)
        d2 = int(up.shape[-1])
        d = d2 // 2
        a = ttnn.slice(up, (0, 0, 0, 0), (int(up.shape[0]), 1, int(up.shape[2]), d))
        b = ttnn.slice(up, (0, 0, 0, d), (int(up.shape[0]), 1, int(up.shape[2]), d2))
        ttnn.deallocate(up)
        a = ttnn.gelu(a)
        y = ttnn.multiply(a, b)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        out = ttnn.linear(y, self.w_down, bias=self.b_down, transpose_b=True)
        ttnn.deallocate(y)
        return out


class MultiHeadSelfAttentionTTNN:
    """
    TTNN explicit QKV self-attention.

    NOTE: This is a minimal correctness-first implementation and uses TTNN ops only.
    """

    def __init__(self, cfg: AceConfigTTNN, *, mesh_device, dtype=None, weights=None):
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.d_model = int(cfg.d_model)
        self.n_heads = int(cfg.n_heads)
        self.d_head = int(cfg.d_head if cfg.d_head is not None else cfg.d_model // cfg.n_heads)
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.dtype = dtype or ttnn.bfloat16

        if weights is None:
            raise ValueError("MultiHeadSelfAttentionTTNN requires weights from Torch module")

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None

        def as_w(t):
            return ttnn.as_tensor(
                t, device=mesh_device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, memory_config=mem, mesh_mapper=mapper
            )

        def as_b(t):
            return ttnn.as_tensor(
                t.reshape(1, 1, 1, -1),
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        self.wq, self.bq = as_w(weights["wq"]), as_b(weights["bq"])
        self.wk, self.bk = as_w(weights["wk"]), as_b(weights["bk"])
        self.wv, self.bv = as_w(weights["wv"]), as_b(weights["bv"])
        self.wo, self.bo = as_w(weights["wo"]), as_b(weights["bo"])
        self._causal_neg_mask_by_s: dict[int, object] = {}

    def _causal_neg_mask(self, S: int):
        """Lower-triangular causal mask as large negative logits; cached per sequence length."""
        cached = self._causal_neg_mask_by_s.get(S)
        if cached is not None:
            return cached
        ttnn = self.ttnn
        _sr = ace_step_reshape_kwargs(ttnn)
        if not hasattr(ttnn, "tril"):
            raise RuntimeError("TTNN build missing ttnn.tril; cannot build causal mask without host ops.")
        keep = ttnn.tril(ttnn.ones((S, S), device=self.mesh_device, dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT))
        keep = ttnn.to_layout(keep, ttnn.TILE_LAYOUT)
        keep = ttnn.reshape(keep, (1, 1, S, S), **_sr)
        inv = ttnn.subtract(1.0, keep)
        ttnn.deallocate(keep)
        neg = ttnn.multiply(inv, -1.0e9)
        ttnn.deallocate(inv)
        self._causal_neg_mask_by_s[S] = neg
        return neg

    def __call__(self, x):
        ttnn = self.ttnn
        _sr = ace_step_reshape_kwargs(ttnn)
        _pk = ace_step_permute_kwargs(ttnn)
        q = ttnn.linear(x, self.wq, bias=self.bq, transpose_b=True)
        k = ttnn.linear(x, self.wk, bias=self.bk, transpose_b=True)
        v = ttnn.linear(x, self.wv, bias=self.bv, transpose_b=True)

        B = int(q.shape[0])
        S = int(q.shape[2])
        H = self.n_heads
        Dh = self.d_head

        # [B,1,S,H*Dh] -> [B,S,H,Dh] -> [B,H,S,Dh]
        q = ttnn.reshape(q, (B, S, H, Dh), **_sr)
        k = ttnn.reshape(k, (B, S, H, Dh), **_sr)
        v = ttnn.reshape(v, (B, S, H, Dh), **_sr)
        q = ttnn.permute(q, (0, 2, 1, 3), **_pk)
        k = ttnn.permute(k, (0, 2, 1, 3), **_pk)
        v = ttnn.permute(v, (0, 2, 1, 3), **_pk)

        kt = ttnn.transpose(k, -2, -1)
        ttnn.deallocate(k)
        scores = ttnn.matmul(q, kt)
        ttnn.deallocate(q)
        ttnn.deallocate(kt)
        scores = ttnn.multiply(scores, self.scale)

        neg = self._causal_neg_mask(S)
        scores = ttnn.add(scores, neg)

        probs = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)
        ctx = ttnn.matmul(probs, v)
        ttnn.deallocate(probs)
        ttnn.deallocate(v)

        # [B,H,S,Dh] -> [B,1,S,H*Dh] (permute + reshape view)
        ctx = ace_step_nlp_concat_heads(ttnn, ctx)
        out = ttnn.linear(ctx, self.wo, bias=self.bo, transpose_b=True)
        ttnn.deallocate(ctx)
        return out


class MultiHeadSelfAttentionSDPATTNN:
    """
    Same linear QKV / output projection contract as :class:`MultiHeadSelfAttentionTTNN`,
    but attention uses ``ttnn.transformer.scaled_dot_product_attention`` (device fused SDPA).

    Input ``x``: ``[B, 1, S, D]``; QKV layouts match TTNN SDPA: ``[B, H, S, Dh]``.
    """

    def __init__(self, cfg: AceConfigTTNN, *, mesh_device, dtype=None, weights=None):
        ttnn = _require_ttnn()
        transformer = getattr(ttnn, "transformer", None)
        sdpa = getattr(transformer, "scaled_dot_product_attention", None) if transformer is not None else None
        if sdpa is None:
            raise RuntimeError(
                "This TTNN build does not expose ttnn.transformer.scaled_dot_product_attention; "
                "use attention_impl='explicit' or rebuild TTNN with SDPA bindings."
            )

        self.ttnn = ttnn
        self._sdpa = sdpa
        self.mesh_device = mesh_device
        self.d_model = int(cfg.d_model)
        self.n_heads = int(cfg.n_heads)
        self.d_head = int(cfg.d_head if cfg.d_head is not None else cfg.d_model // cfg.n_heads)
        self.dtype = dtype or ttnn.bfloat16

        if weights is None:
            raise ValueError("MultiHeadSelfAttentionSDPATTNN requires weights from Torch module")

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None

        def as_w(t):
            return ttnn.as_tensor(
                t, device=mesh_device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, memory_config=mem, mesh_mapper=mapper
            )

        def as_b(t):
            return ttnn.as_tensor(
                t.reshape(1, 1, 1, -1),
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        self.wq, self.bq = as_w(weights["wq"]), as_b(weights["bq"])
        self.wk, self.bk = as_w(weights["wk"]), as_b(weights["bk"])
        self.wv, self.bv = as_w(weights["wv"]), as_b(weights["bv"])
        self.wo, self.bo = as_w(weights["wo"]), as_b(weights["bo"])
        self._sdpa_d = _sdpa_head_dim_tile_padding(self.d_head)
        self._sdpa_pad = self._sdpa_d - self.d_head

    def __call__(self, x):
        ttnn = self.ttnn
        _sr = ace_step_reshape_kwargs(ttnn)
        _pk = ace_step_permute_kwargs(ttnn)
        q = ttnn.linear(x, self.wq, bias=self.bq, transpose_b=True)
        k = ttnn.linear(x, self.wk, bias=self.bk, transpose_b=True)
        v = ttnn.linear(x, self.wv, bias=self.bv, transpose_b=True)

        B = int(q.shape[0])
        S = int(q.shape[2])
        H = self.n_heads
        Dh = self.d_head

        # [B,1,S,H*Dh] -> [B,S,H,Dh] -> [B,H,S,Dh]
        q = ttnn.reshape(q, (B, S, H, Dh), **_sr)
        k = ttnn.reshape(k, (B, S, H, Dh), **_sr)
        v = ttnn.reshape(v, (B, S, H, Dh), **_sr)
        q = ttnn.permute(q, (0, 2, 1, 3), **_pk)
        k = ttnn.permute(k, (0, 2, 1, 3), **_pk)
        v = ttnn.permute(v, (0, 2, 1, 3), **_pk)

        if self._sdpa_pad > 0:
            pad4 = ((0, 0), (0, 0), (0, 0), (0, self._sdpa_pad))
            q = ttnn.pad(q, padding=pad4, value=0.0)
            k = ttnn.pad(k, padding=pad4, value=0.0)
            v = ttnn.pad(v, padding=pad4, value=0.0)

        # Explicit scale: TTNN may derive scale from padded last dim; keep PyTorch 1/sqrt(d_head).
        attn_scale = 1.0 / math.sqrt(float(Dh))
        ctx = self._sdpa(q, k, v, attn_mask=None, is_causal=True, scale=attn_scale)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        if self._sdpa_pad > 0:
            ctx = ttnn.slice(ctx, (0, 0, 0, 0), (B, H, S, Dh))
        # [B,H,S,Dh] -> [B,1,S,H*Dh] (permute + reshape view)
        ctx = ace_step_nlp_concat_heads(ttnn, ctx)
        out = ttnn.linear(ctx, self.wo, bias=self.bo, transpose_b=True)
        ttnn.deallocate(ctx)
        return out


class TransformerBlockTTNN:
    """
    TTNN Transformer block:
      x -> AdaLN -> Attn -> Residual -> AdaLN -> MLP -> Residual
    """

    def __init__(self, cfg: AceConfigTTNN, *, mesh_device, dtype=None, weights=None):
        if weights is None:
            raise ValueError("TransformerBlockTTNN requires a nested weights dict")
        self.adaln_attn = AdaLNZeroTTNN(cfg, mesh_device=mesh_device, dtype=dtype, weights=weights["adaln_attn"])
        if cfg.attention_impl == "sdpa":
            self.attn = MultiHeadSelfAttentionSDPATTNN(
                cfg, mesh_device=mesh_device, dtype=dtype, weights=weights["attn"]
            )
        else:
            self.attn = MultiHeadSelfAttentionTTNN(cfg, mesh_device=mesh_device, dtype=dtype, weights=weights["attn"])
        self.adaln_mlp = AdaLNZeroTTNN(cfg, mesh_device=mesh_device, dtype=dtype, weights=weights["adaln_mlp"])
        self.mlp = GEGLUMLPTTNN(cfg, mesh_device=mesh_device, dtype=dtype, weights=weights["mlp"])
        self.ttnn = self.adaln_attn.ttnn

    def __call__(self, x, cond):
        ttnn = self.ttnn
        h = self.adaln_attn(x, cond)
        a = self.attn(h)
        ttnn.deallocate(h)
        x = ttnn.add(x, a)
        ttnn.deallocate(a)
        h2 = self.adaln_mlp(x, cond)
        m = self.mlp(h2)
        ttnn.deallocate(h2)
        x = ttnn.add(x, m)
        ttnn.deallocate(m)
        return x
