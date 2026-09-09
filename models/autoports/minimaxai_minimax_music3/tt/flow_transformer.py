# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-Music3's flow-matching DiT (``MiniMaxMusic3Transformer1DModel``) on one Blackhole chip.

Reference: diffusers ``models/transformers/transformer_minimax_music3.py`` (vendored as
``reference/flow_transformer_ref.py``). 36 pre-LayerNorm blocks of dim 2048 (32 heads x 64, partial
RoPE on the first 32 dims of every head, theta 1e4), gated MLP ``ff_out(a * silu(g))`` with
``(a, g) = chunk(ff_in(x), 2)``, one prepended timestep token (position 0) that is dropped again after
the blocks, batch 2 = (conditional row, zero-condition row) so one forward serves both CFG branches.

Device layout
-------------
Both batch rows share ONE tile-aligned activation ``[1, 1, B * S_pad, 2048]`` (bf16, tile layout,
DRAM) where ``S = T + 1`` (timestep token + ``T`` latent frames) and ``S_pad`` is ``S`` rounded up to
a multiple of ``SEQ_PAD`` (128, the SDPA chunk size). Row ``b * S_pad + s`` is position ``s`` of batch
row ``b``; rows past ``S`` are zero on input, are masked out as attention KEYS (additive ``-1e9`` mask on
the padded key columns) and are sliced away on the host, so a logical ``T`` of anything up to
``MAX_LATENTS`` runs through the same code path. Head ops see the same buffer as ``[B, 1, S_pad, 2048]``
via zero-copy ``ttnn.experimental.view`` (the row split is tile aligned).

Algebraic folds (done in fp32 on the host at load time, exact up to one bf16 rounding):

* input: the reference concatenates ``[latents, zeros, condition]`` over channels, applies the 1x1
  ``preprocess_conv`` with a residual and then ``proj_in``. That is ``x @ ((I + Wc^T) @ Wp^T)``; the
  128 zero channels contribute nothing, so the folded matrix splits into ``w_in_latent`` ``[128, 2048]``
  and ``w_in_condition`` ``[2048, 2048]``. The condition is constant over a chunk's 30 steps, so
  ``prepare_condition`` computes ``condition @ w_in_condition`` once and ``forward`` only runs the
  latent half per step.
* output: ``proj_out`` then the 1x1 ``postprocess_conv`` with a residual is ``h @ (Wo^T @ (I + Wpost^T))``
  = ``w_out`` ``[2048, 128]``.
* the timestep embedding (Fourier features -> 2 linears, 1.3 M parameters) is evaluated on the host in
  fp32 and scattered into row ``b * S_pad`` with a one-hot ``[B * S_pad, 32] @ [32, 2048]`` matmul.

Partial RoPE uses ``ttnn.experimental.rotary_embedding_llama`` (prefill mode) with cos/sin tables that
are 1 / 0 on head dims 32..63 and a 32x32 "rotate-half" transformation matrix; because the op applies
the matrix tile by tile, the first 32-wide tile of every head is rotated exactly like the reference's
``rotate_half`` over the 32 rotary dims and the second tile is left untouched (probe:
``scripts/probe_dit_ops.py``, PCC 0.99999 vs torch).

Weights are bf16 on device (about 4.8 GB), matmuls run HiFi2 with fp32 accumulation, LayerNorm HiFi4
fp32, SDPA HiFi4. The weight tensors live in ``models.tt_dit`` ``Parameter`` objects so
``models.tt_dit.utils.cache.load_model`` can cache the converted tensors under ``TT_DIT_CACHE_DIR``.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple, Union

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_dit.layers.module import Module, ModuleList, Parameter
from models.tt_dit.parallel.config import DiTParallelConfig
from models.tt_dit.utils.cache import load_model
from models.tt_dit.utils.substate import pop_substate

TILE = 32
IN_CHANNELS = 128
CONDITION_DIM = 2048
DIM = 2048
HEADS = 32
HEAD_DIM = 64
ROTARY_DIM = 32
ROPE_THETA = 10000.0
FF_INNER = 8192
FOURIER_DIM = 256
NUM_LAYERS = 36
NORM_EPS = 1e-5
BATCH = 2  # row 0 = conditional, row 1 = zero condition (CFG)
SEQ_PAD = 128  # SDPA q/k chunk; S = T + 1 is padded to a multiple of this
# 200 frames -> int(200 * 44100 / 24000 * 960 / 512) = 689 latents (+1 timestep token = 690 -> 768).
MAX_LATENTS = 9000
MASK_VALUE = -1e9

TensorLike = Union[torch.Tensor, ttnn.Tensor]


def padded_seq_len(num_latents: int) -> int:
    """``S_pad`` for ``T = num_latents`` (timestep token included)."""
    assert 1 <= num_latents <= MAX_LATENTS, num_latents
    return -(-(num_latents + 1) // SEQ_PAD) * SEQ_PAD


# ----------------------------------------------------------------------------- host-side pieces
def fold_input_weights(sd: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(w_in_latent [128, 2048], w_in_condition [2048, 2048])`` = rows of ``(I + Wc^T) @ Wp^T`` (fp32)."""
    wc = sd["preprocess_conv.weight"].float().squeeze(-1)  # [2304, 2304] (out, in)
    wp = sd["proj_in.weight"].float()  # [2048, 2304]
    c = wc.shape[0]
    folded = (torch.eye(c) + wc.t()) @ wp.t()  # [2304, 2048]
    return folded[:IN_CHANNELS].contiguous(), folded[2 * IN_CHANNELS :].contiguous()


def fold_output_weights(sd: Dict[str, torch.Tensor]) -> torch.Tensor:
    """``w_out [2048, 128]`` = ``Wo^T @ (I + Wpost^T)`` (fp32)."""
    wo = sd["proj_out.weight"].float()  # [128, 2048]
    wpost = sd["postprocess_conv.weight"].float().squeeze(-1)  # [128, 128]
    return (wo.t() @ (torch.eye(IN_CHANNELS) + wpost.t())).contiguous()


class TimestepEmbedder:
    """``time_embed(time_proj(t))`` on the host in fp32 (Fourier features -> linear -> silu -> linear)."""

    def __init__(self, sd: Dict[str, torch.Tensor]):
        self.fourier = sd["time_proj.weight"].float().reshape(-1)  # [128]
        self.w1 = sd["time_embed.linear_1.weight"].float()
        self.b1 = sd["time_embed.linear_1.bias"].float()
        self.w2 = sd["time_embed.linear_2.weight"].float()
        self.b2 = sd["time_embed.linear_2.bias"].float()

    def __call__(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.reshape(-1).float()
        angles = 2.0 * math.pi * t[:, None] * self.fourier[None, :]
        feats = torch.cat((angles.cos(), angles.sin()), dim=-1)
        h = torch.nn.functional.silu(feats @ self.w1.t() + self.b1)
        return h @ self.w2.t() + self.b2  # [B, 2048]


def rope_tables(seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos / sin ``[1, 1, seq_len, 64]``: the reference's partial tables, padded with (1, 0) on dims 32..63."""
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    freqs = torch.cat((freqs, freqs), dim=-1)  # [S, 32]
    cos = torch.cat((freqs.cos(), torch.ones(seq_len, HEAD_DIM - ROTARY_DIM)), dim=-1)
    sin = torch.cat((freqs.sin(), torch.zeros(seq_len, HEAD_DIM - ROTARY_DIM)), dim=-1)
    return cos.reshape(1, 1, seq_len, HEAD_DIM), sin.reshape(1, 1, seq_len, HEAD_DIM)


def rope_transformation_matrix() -> torch.Tensor:
    """``[1, 1, 32, 32]`` so that ``x @ M = cat(-x[16:], x[:16])`` (rotate_half within one 32-wide tile)."""
    half = TILE // 2
    m = torch.zeros(1, 1, TILE, TILE)
    for j in range(half):
        m[..., j + half, j] = -1.0
        m[..., j, j + half] = 1.0
    return m


# ----------------------------------------------------------------------------- device weights
class _BlockWeights(Module):
    def __init__(self, mesh_device, dtype):
        super().__init__()
        p = lambda *shape, dt=dtype: Parameter(total_shape=shape, device=mesh_device, dtype=dt)  # noqa: E731
        self.ln1_w = p(1, 1, 1, DIM)
        self.ln1_b = p(1, 1, 1, DIM)
        self.wqkv = p(DIM, 3 * DIM)
        self.wo = p(DIM, DIM)
        self.ln2_w = p(1, 1, 1, DIM)
        self.ln2_b = p(1, 1, 1, DIM)
        self.w_in_a = p(DIM, FF_INNER)
        self.b_in_a = p(1, FF_INNER)
        self.w_in_g = p(DIM, FF_INNER)
        self.b_in_g = p(1, FF_INNER)
        self.w_out = p(FF_INNER, DIM)
        self.b_out = p(1, DIM)

    def _prepare_torch_state(self, state: Dict[str, torch.Tensor]) -> None:
        """diffusers block keys -> the device tensors above (transposed to ``[in, out]``)."""
        norm1, norm2 = pop_substate(state, "norm1"), pop_substate(state, "norm2")
        attn, ff_in, ff_out = pop_substate(state, "attn"), pop_substate(state, "ff_in"), pop_substate(state, "ff_out")
        if not attn:
            return
        state["ln1_w"] = norm1["weight"].reshape(1, 1, 1, DIM)
        state["ln1_b"] = norm1["bias"].reshape(1, 1, 1, DIM)
        state["ln2_w"] = norm2["weight"].reshape(1, 1, 1, DIM)
        state["ln2_b"] = norm2["bias"].reshape(1, 1, 1, DIM)
        state["wqkv"] = (
            torch.cat([attn["to_q.weight"], attn["to_k.weight"], attn["to_v.weight"]], dim=0).t().contiguous()
        )
        state["wo"] = attn["to_out.0.weight"].t().contiguous()
        w_in = ff_in["weight"].t().contiguous()  # [2048, 16384]; chunk(2) on the output -> first half a, second g
        b_in = ff_in["bias"]
        state["w_in_a"], state["w_in_g"] = w_in[:, :FF_INNER].contiguous(), w_in[:, FF_INNER:].contiguous()
        state["b_in_a"], state["b_in_g"] = b_in[:FF_INNER].reshape(1, -1), b_in[FF_INNER:].reshape(1, -1)
        state["w_out"] = ff_out["weight"].t().contiguous()
        state["b_out"] = ff_out["bias"].reshape(1, -1)

    def forward(self):  # pragma: no cover - weights only
        raise NotImplementedError


class _FlowTransformerWeights(Module):
    """All device-resident weights of the DiT, with the input/output folds applied in ``_prepare_torch_state``."""

    def __init__(self, mesh_device, dtype, num_layers: int):
        super().__init__()
        p = lambda *shape, dt=dtype: Parameter(total_shape=shape, device=mesh_device, dtype=dt)  # noqa: E731
        self.w_in_latent = p(IN_CHANNELS, DIM)
        self.w_in_condition = p(CONDITION_DIM, DIM)
        self.w_out = p(DIM, IN_CHANNELS)
        self.blocks = ModuleList(_BlockWeights(mesh_device, dtype) for _ in range(num_layers))

    def _prepare_torch_state(self, state: Dict[str, torch.Tensor]) -> None:
        if "preprocess_conv.weight" in state:
            state["w_in_latent"], state["w_in_condition"] = fold_input_weights(state)
            state["w_out"] = fold_output_weights(state)
            for k in ("preprocess_conv.weight", "proj_in.weight", "proj_out.weight", "postprocess_conv.weight"):
                state.pop(k)
        # The timestep embedder stays on the host (see TimestepEmbedder); drop its keys here.
        for k in list(state):
            if k.startswith("time_proj.") or k.startswith("time_embed."):
                state.pop(k)
        blocks = pop_substate(state, "transformer_blocks")
        n = len(self.blocks)
        for k, v in blocks.items():
            if int(k.split(".")[0]) < n:
                state[f"blocks.{k}"] = v

    def forward(self):  # pragma: no cover - weights only
        raise NotImplementedError


# ----------------------------------------------------------------------------- the model
class FlowTransformer(LightweightModule):
    """``MiniMaxMusic3Transformer1DModel`` on device. See the module docstring for the layout."""

    def __init__(
        self,
        mesh_device,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        *,
        dtype=ttnn.bfloat16,
        num_layers: int = NUM_LAYERS,
        get_state_dict=None,
        model_name: str = "minimax-music3",
    ):
        """``state_dict`` (fp32 diffusers keys) or a lazy ``get_state_dict`` callable (lets ``load_model`` skip
        reading the safetensors when the ``TT_DIT_CACHE_DIR`` cache already holds the converted tensors)."""
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.num_layers = num_layers
        self.mem = ttnn.DRAM_MEMORY_CONFIG
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.norm_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.sdpa_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        grid = mesh_device.compute_with_storage_grid_size()
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            q_chunk_size=SEQ_PAD,
            k_chunk_size=SEQ_PAD,
            exp_approx_mode=False,
        )

        t0 = time.time()
        if state_dict is None and get_state_dict is None:
            raise ValueError("pass state_dict or get_state_dict")
        get_sd = get_state_dict if get_state_dict is not None else (lambda: state_dict)
        # The host-side timestep embedder needs the torch weights even when the device cache hits, so
        # the safetensors are read once here in every case (about 1 s per shard from the page cache).
        sd = get_sd()
        self.time_embedder = TimestepEmbedder(sd)
        self.weights = _FlowTransformerWeights(mesh_device, dtype, num_layers)
        load_model(
            self.weights,
            model_name=model_name,
            subfolder=f"transformer_l{num_layers}",
            parallel_config=DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(1, 0), tp=(1, 0)),
            mesh_shape=tuple(mesh_device.shape),
            dtype="bf16" if dtype == ttnn.bfloat16 else str(dtype).split(".")[-1],
            get_torch_state_dict=lambda: sd,
        )
        self.trans_mat = self._to_device(rope_transformation_matrix())
        self._rope_cache: Dict[int, Tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        self._mask_cache: Dict[Tuple[int, int], ttnn.Tensor] = {}
        self._time_selector_cache: Dict[int, ttnn.Tensor] = {}
        ttnn.synchronize_device(mesh_device)
        self.load_seconds = time.time() - t0

    @classmethod
    def from_pretrained(cls, mesh_device, weights_dir=None, *, dtype=ttnn.bfloat16, num_layers: int = NUM_LAYERS):
        from models.autoports.minimaxai_minimax_music3.reference.flow_transformer_ref import load_transformer_state_dict

        return cls(
            mesh_device,
            dtype=dtype,
            num_layers=num_layers,
            get_state_dict=lambda: load_transformer_state_dict(weights_dir),
        )

    # ------------------------------------------------------------------ helpers
    def _to_device(self, t: torch.Tensor, dtype=None, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.contiguous(), dtype=dtype or self.dtype, layout=layout, device=self.mesh_device, memory_config=self.mem
        )

    def _linear(self, x, w, bias=None, activation=None) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            w,
            bias=bias,
            activation=activation,
            compute_kernel_config=self.compute_config,
            memory_config=self.mem,
            dtype=self.dtype,
        )

    def _rope(self, s_pad: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        if s_pad not in self._rope_cache:
            cos, sin = rope_tables(s_pad)
            self._rope_cache[s_pad] = (self._to_device(cos), self._to_device(sin))
        return self._rope_cache[s_pad]

    def _mask(self, s_pad: int, s_real: int) -> Optional[ttnn.Tensor]:
        """Additive ``[1, 1, S_pad, S_pad]`` mask that hides the padded key columns (None when nothing is padded)."""
        if s_real == s_pad:
            return None
        key = (s_pad, s_real)
        if key not in self._mask_cache:
            m = torch.zeros(1, 1, s_pad, s_pad)
            m[..., s_real:] = MASK_VALUE
            self._mask_cache[key] = self._to_device(m)
        return self._mask_cache[key]

    def _time_selector(self, s_pad: int) -> ttnn.Tensor:
        """One-hot ``[B * S_pad, 32]``: ``sel @ rows`` puts ``rows[b]`` at row ``b * S_pad`` (the timestep token)."""
        if s_pad not in self._time_selector_cache:
            e = torch.zeros(BATCH * s_pad, TILE)
            for b in range(BATCH):
                e[b * s_pad, b] = 1.0
            self._time_selector_cache[s_pad] = self._to_device(e.reshape(1, 1, BATCH * s_pad, TILE))
        return self._time_selector_cache[s_pad]

    @staticmethod
    def _pad_rows(x: torch.Tensor, s_pad: int) -> torch.Tensor:
        """Host ``[B, T, C]`` -> ``[1, 1, B * S_pad, C]`` with position 0 (timestep slot) and rows past ``T + 1`` zero."""
        b, t, c = x.shape
        out = torch.zeros(b, s_pad, c, dtype=torch.bfloat16)
        out[:, 1 : t + 1] = x.to(torch.bfloat16)
        return out.reshape(1, 1, b * s_pad, c)

    # ------------------------------------------------------------------ public API
    def prepare_condition(self, condition: torch.Tensor) -> ttnn.Tensor:
        """``condition [B, T, 2048]`` (host fp32; row 1 all zeros for CFG) -> its folded input projection
        ``[1, 1, B * S_pad, 2048]`` on device. Constant over a chunk, so compute it once per chunk."""
        assert condition.dim() == 3 and condition.shape[0] == BATCH and condition.shape[2] == CONDITION_DIM, tuple(
            condition.shape
        )
        s_pad = padded_seq_len(condition.shape[1])
        x = self._to_device(self._pad_rows(condition, s_pad))
        out = self._linear(x, self.weights.w_in_condition.data)
        ttnn.deallocate(x)
        return out

    def embed_inputs(self, latents: torch.Tensor, timestep: torch.Tensor, cond_proj: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, B * S_pad, 2048]`` block input: folded latent projection + condition projection + timestep token."""
        b, c, t = latents.shape
        assert b == BATCH and c == IN_CHANNELS, tuple(latents.shape)
        s_pad = padded_seq_len(t)
        assert cond_proj.shape[-2] == BATCH * s_pad, (cond_proj.shape, s_pad)
        x_lat = self._to_device(self._pad_rows(latents.transpose(1, 2), s_pad))
        x_lat_proj = self._linear(x_lat, self.weights.w_in_latent.data)
        ttnn.deallocate(x_lat)
        x = ttnn.add(x_lat_proj, cond_proj, memory_config=self.mem)
        ttnn.deallocate(x_lat_proj)
        temb = self.time_embedder(timestep.expand(BATCH) if timestep.numel() == 1 else timestep)  # [B, 2048] fp32
        rows = torch.zeros(TILE, DIM)
        rows[:BATCH] = temb
        rows_d = self._to_device(rows.reshape(1, 1, TILE, DIM))
        tok = ttnn.matmul(
            self._time_selector(s_pad),
            rows_d,
            compute_kernel_config=self.compute_config,
            memory_config=self.mem,
            dtype=self.dtype,
        )
        ttnn.deallocate(rows_d)
        out = ttnn.add(x, tok, memory_config=self.mem)
        ttnn.deallocate(tok)
        ttnn.deallocate(x)
        return out

    def _attention(self, x_norm: ttnn.Tensor, blk: _BlockWeights, s_pad: int, s_real: int) -> ttnn.Tensor:
        qkv = self._linear(x_norm, blk.wqkv.data)  # [1, 1, B*S_pad, 6144]
        qkv = ttnn.experimental.view(qkv, (BATCH, 1, s_pad, 3 * DIM))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=HEADS, num_kv_heads=HEADS, transpose_k_heads=False, memory_config=self.mem
        )  # [B, 32, S_pad, 64] each
        ttnn.deallocate(qkv)
        cos, sin = self._rope(s_pad)
        # rotary_embedding_llama (prefill) wants batch 1: fold the batch into the head dim (tile-aligned view).
        q = ttnn.experimental.view(q, (1, BATCH * HEADS, s_pad, HEAD_DIM))
        k = ttnn.experimental.view(k, (1, BATCH * HEADS, s_pad, HEAD_DIM))
        q_r = ttnn.experimental.rotary_embedding_llama(q, cos, sin, self.trans_mat, is_decode_mode=False)
        k_r = ttnn.experimental.rotary_embedding_llama(k, cos, sin, self.trans_mat, is_decode_mode=False)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        q_r = ttnn.experimental.view(q_r, (BATCH, HEADS, s_pad, HEAD_DIM))
        k_r = ttnn.experimental.view(k_r, (BATCH, HEADS, s_pad, HEAD_DIM))
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_r,
            k_r,
            v,
            attn_mask=self._mask(s_pad, s_real),
            is_causal=False,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_config,
            memory_config=self.mem,
        )
        ttnn.deallocate(q_r)
        ttnn.deallocate(k_r)
        ttnn.deallocate(v)
        merged = ttnn.experimental.nlp_concat_heads(attn, memory_config=self.mem)  # [B, 1, S_pad, 2048]
        ttnn.deallocate(attn)
        merged = ttnn.experimental.view(merged, (1, 1, BATCH * s_pad, DIM))
        out = self._linear(merged, blk.wo.data)
        ttnn.deallocate(merged)
        return out

    def _mlp(self, x_norm: ttnn.Tensor, blk: _BlockWeights) -> ttnn.Tensor:
        a = self._linear(x_norm, blk.w_in_a.data, bias=blk.b_in_a.data)
        g = self._linear(x_norm, blk.w_in_g.data, bias=blk.b_in_g.data, activation="silu")
        h = ttnn.multiply(a, g, memory_config=self.mem)
        ttnn.deallocate(a)
        ttnn.deallocate(g)
        out = self._linear(h, blk.w_out.data, bias=blk.b_out.data)
        ttnn.deallocate(h)
        return out

    def _layer_norm(self, x: ttnn.Tensor, w: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x,
            weight=w,
            bias=b,
            epsilon=NORM_EPS,
            compute_kernel_config=self.norm_compute_config,
            memory_config=self.mem,
        )

    def blocks(self, x: ttnn.Tensor, s_pad: int, s_real: int, num_layers: Optional[int] = None) -> ttnn.Tensor:
        """The transformer stack over the ``[1, 1, B * S_pad, 2048]`` input (``x`` is consumed)."""
        for blk in list(self.weights.blocks)[: num_layers if num_layers is not None else self.num_layers]:
            n1 = self._layer_norm(x, blk.ln1_w.data, blk.ln1_b.data)
            a = self._attention(n1, blk, s_pad, s_real)
            ttnn.deallocate(n1)
            x2 = ttnn.add(x, a, memory_config=self.mem)
            ttnn.deallocate(x)
            ttnn.deallocate(a)
            n2 = self._layer_norm(x2, blk.ln2_w.data, blk.ln2_b.data)
            m = self._mlp(n2, blk)
            ttnn.deallocate(n2)
            x = ttnn.add(x2, m, memory_config=self.mem)
            ttnn.deallocate(x2)
            ttnn.deallocate(m)
        return x

    def project_out(self, x: ttnn.Tensor, num_latents: int) -> torch.Tensor:
        """``[1, 1, B * S_pad, 2048]`` -> host velocity ``[B, 128, T]`` (fp32), dropping the timestep token and padding."""
        s_pad = padded_seq_len(num_latents)
        y = self._linear(x, self.weights.w_out.data)  # [1, 1, B*S_pad, 128]
        out = ttnn.to_torch(y).float().reshape(BATCH, s_pad, IN_CHANNELS)[:, 1 : num_latents + 1]
        ttnn.deallocate(y)
        return out.transpose(1, 2).contiguous()

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        *,
        cond_proj: Optional[ttnn.Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> torch.Tensor:
        """``latents [B, 128, T]``, ``timestep [B]`` (flow time, 0 = noise), ``condition [B, T, 2048]`` (or its
        ``prepare_condition`` result as ``cond_proj``) -> predicted velocity ``[B, 128, T]`` (host fp32)."""
        b, c, t = latents.shape
        assert b == BATCH and c == IN_CHANNELS, tuple(latents.shape)
        s_pad, s_real = padded_seq_len(t), t + 1
        own_cond = cond_proj is None
        if own_cond:
            assert condition is not None and condition.shape[1] == t, "pass condition [B, T, 2048] or cond_proj"
            cond_proj = self.prepare_condition(condition)
        x = self.embed_inputs(latents, timestep, cond_proj)
        if own_cond:
            ttnn.deallocate(cond_proj)
        x = self.blocks(x, s_pad, s_real, num_layers)
        out = self.project_out(x, t)
        ttnn.deallocate(x)
        return out

    def release(self) -> None:
        self.weights.deallocate_weights()
        for cache in (self._rope_cache, self._mask_cache, self._time_selector_cache):
            for v in cache.values():
                for tt in v if isinstance(v, tuple) else (v,):
                    ttnn.deallocate(tt)
            cache.clear()
