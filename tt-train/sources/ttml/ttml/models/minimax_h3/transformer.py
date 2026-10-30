# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 transformer (t2va, one packed sequence) in ttml.

Device sees (1, 1, S_pad, D) tiles. Everything frozen and row-structural (timestep MLP, AdaLN row selection, rotary
tables, sequence packing) is computed on the host or with raw ttnn outside autograd; gradients only reach the LoRA
adapters inside the blocks, so the packed input needs no graph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ttnn
import ttml
from ttml.modules import (
    AbstractModuleBase,
    ColumnParallelLinear,
    LinearLayer,
    ModuleList,
    Parameter,
    RowParallelLinear,
    RunMode,
)

from .packing import (
    AUDIO_CHANNELS,
    AUDIO_LATENTS,
    MODALITY_NUM,
    TILE,
    Layout,
    adaln_indices,
    attention_mask,
    rope_tables,
    rope_trans_mat,
    selection_matrix,
    tile_pad,
    time_embedding,
)


@dataclass(frozen=True)
class H3Config:
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_dim: int = 14336
    num_layers: int = 50
    num_refiner_layers: int = 2
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_dim: int = 2688
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    use_tp: bool = False
    # "composite" materialises the scores; the fused kernel's backward loses the small softmax terms next to the
    # attention-sink rows of this model and its LoRA grads come out 4x to 1000x too large.
    sdpa: str = "composite"

    @property
    def video_in_channels(self) -> int:
        return self.in_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

    @property
    def rotary_dim(self) -> int:
        return 2 * 3 * self.rope_freq_dim

    @classmethod
    def from_hf(
        cls, config_json: Path, num_layers: int | None = None, use_tp: bool = False, sdpa: str = "composite"
    ) -> "H3Config":
        hf = json.loads(Path(config_json).read_text())
        keys = [k for k in cls.__dataclass_fields__ if k in hf and k not in ("use_tp", "sdpa")]
        values = {k: (tuple(hf[k]) if isinstance(hf[k], list) else hf[k]) for k in keys}
        if num_layers is not None:
            values["num_layers"] = num_layers
        return cls(use_tp=use_tp, sdpa=sdpa, **values)


def _tp_size(config: H3Config) -> int:
    return ttml.mesh().axis_size("tp") if config.use_tp else 1


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _replicate_mapper():
    return ttml.core.distributed.replicate_tensor_to_mesh_mapper(_device())


def _empty_init(config: H3Config, tdim: int | None):
    """Uninitialised per-device weight; the checkpoint loader fills it. Skips host RNG, tilize and upload."""
    tp = _tp_size(config)

    def init(shape, mapper=None):
        per = list(shape)
        if tdim is not None:
            per[tdim] //= tp
        # ROW_MAJOR keeps the logical shape exact (no tile padding); `assign` replaces it before any use.
        return ttml.autograd.create_tensor(ttnn.empty(per, ttnn.DataType.BFLOAT16, ttnn.Layout.ROW_MAJOR, _device()))

    return init


def upload(arr: np.ndarray, dtype=ttnn.DataType.BFLOAT16, requires_grad: bool = False) -> ttml.autograd.Tensor:
    """Host (1, 1, R, C) array -> replicated TILE device tensor outside the graph."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    t = ttml.autograd.Tensor.from_numpy(arr, ttnn.Layout.TILE, dtype, _replicate_mapper())
    t.set_requires_grad(requires_grad)
    return t


def const(value: ttnn.Tensor) -> ttml.autograd.Tensor:
    return ttml.autograd.create_tensor(value, False)


def _slices(value: ttnn.Tensor, sizes, dim: int) -> tuple[ttnn.Tensor, ...]:
    """Contiguous chunks along `dim` with ttnn.slice; ttnn.split only runs on a few cores."""
    out, start = [], 0
    for size in sizes:
        begin, end = [0, 0, 0, 0], list(value.shape)
        begin[dim], end[dim] = start, start + size
        out.append(ttnn.slice(value, begin, end))
        start += size
    return tuple(out)


class Split(ttml.autograd.Function):
    """Contiguous chunks along a dim; backward is one concat."""

    @staticmethod
    def forward(ctx, input, sizes, dim):
        ctx.dim = dim
        return _slices(input.get_value(), sizes, dim)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return ttnn.concat(list(grad_outputs), ctx.dim)


class Concat(ttml.autograd.Function):
    """Concatenation along a dim; backward slices the gradient back."""

    @staticmethod
    def forward(ctx, dim, *inputs):
        ctx.dim, ctx.sizes = dim, [t.shape()[dim] for t in inputs]
        return ttnn.concat([t.get_value() for t in inputs], dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _slices(grad_output, ctx.sizes, ctx.dim)


class RMSNorm(AbstractModuleBase):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Parameter(ttml.init.ones()((1, 1, 1, dim)))

    def forward(self, x):
        return ttml.ops.rmsnorm.rmsnorm(x, self.weight.tensor, self.eps)


class H3Attention(AbstractModuleBase):
    """Full bidirectional self-attention, per-head QK RMSNorm, optional rotary on q/k.

    q, k and v come from one matmul against the frozen `[Wq; Wk; Wv]`, fused from the three projections on the
    first forward; the separate copies are freed then, so read them before. With LoRA on the three projections
    their A matrices are fused the same way: one `x @ [A_q; A_k; A_v]^T`, then the three B matmuls, so LoRA dropout
    draws one mask for all three."""

    def __init__(self, config: H3Config) -> None:
        super().__init__()
        tp = _tp_size(config)
        if config.num_attention_heads % tp:
            raise ValueError(f"{config.num_attention_heads} heads not divisible by tp {tp}")
        self.tp = tp
        self.local_heads = config.num_attention_heads // tp
        self.local_inner = self.local_heads * config.attention_head_dim
        # Not a registered parameter: it would be listed as trainable.
        object.__setattr__(self, "_qkv_weight", None)
        dim, inner = config.hidden_size, config.num_attention_heads * config.attention_head_dim
        col_init, row_init = _empty_init(config, 2), _empty_init(config, 3)
        if tp > 1:
            col = dict(has_bias=False, weight_init=col_init, gather_output=False, axis_name="tp")
            self.to_q = ColumnParallelLinear(dim, inner, **col)
            self.to_k = ColumnParallelLinear(dim, inner, **col)
            self.to_v = ColumnParallelLinear(dim, inner, **col)
            self.to_out = RowParallelLinear(
                inner, dim, has_bias=False, weight_init=row_init, input_is_parallel=True, axis_name="tp"
            )
        else:
            self.to_q = LinearLayer(dim, inner, False, weight_init=col_init)
            self.to_k = LinearLayer(dim, inner, False, weight_init=col_init)
            self.to_v = LinearLayer(dim, inner, False, weight_init=col_init)
            self.to_out = LinearLayer(inner, dim, False, weight_init=row_init)
        self.norm_q = RMSNorm(config.attention_head_dim, config.qk_norm_eps)
        self.norm_k = RMSNorm(config.attention_head_dim, config.qk_norm_eps)
        if config.sdpa not in ("fused", "composite"):
            raise ValueError(f"sdpa must be 'fused' or 'composite', got {config.sdpa!r}")
        self.sdpa = (
            ttml.ops.attention.scaled_dot_product_attention
            if config.sdpa == "fused"
            else ttml.ops.attention.scaled_dot_product_attention_composite
        )

    def _projections(self):
        return self.to_q, self.to_k, self.to_v

    def _fused_qkv_weight(self) -> ttml.autograd.Tensor:
        if self._qkv_weight is None:
            parts = [p.weight.tensor.get_value() for p in self._projections()]
            fused = ttnn.concat(parts, 2)  # per device: [q | k | v] rows of the local heads
            for part in parts:
                ttnn.deallocate(part)
            object.__setattr__(self, "_qkv_weight", ttml.autograd.create_tensor(fused, False))
        return self._qkv_weight

    def _sum_over_tp(self, lora_A):
        """Each device's grad of a replicated lora_A only carries its own lora_B shard; broadcast sums it."""
        return ttml.ops.distributed.broadcast(lora_A, self.to_q.cluster_axis) if self.tp > 1 else lora_A

    def _lora_updates(self, x):
        """The three LoRA terms `scaling * (x @ A^T) @ B^T`, or None without adapters."""
        projs = self._projections()
        adapted = [hasattr(p, "lora_A") for p in projs]
        if not any(adapted):
            return None
        if not all(adapted):
            raise ValueError("LoRA must target all of to_q, to_k and to_v, or none of them")
        lead = projs[0]
        if lead.get_run_mode() == RunMode.TRAIN and lead.dropout_prob > 0.0:
            x = ttml.ops.dropout.dropout(x, lead.dropout_prob)
        rank = lead.lora_A.tensor.shape()[2]
        if rank % TILE == 0:
            a = self._sum_over_tp(Concat.apply(2, *(p.lora_A.tensor for p in projs)))
            hs = Split.apply(ttml.ops.linear.linear(x, a, None), (rank,) * 3, 3)
        else:  # the split of x @ A^T must fall on tile boundaries
            hs = [ttml.ops.linear.linear(x, self._sum_over_tp(p.lora_A.tensor), None) for p in projs]
        return [ttml.ops.linear.linear(h, p.lora_B.tensor, None) * p.scaling for h, p in zip(hs, projs)]

    def forward(self, x, mask, rope_params=None):
        if self.tp > 1:
            x = ttml.ops.distributed.broadcast(x, self.to_q.cluster_axis)
        qkv = ttml.ops.linear.linear(x, self._fused_qkv_weight(), None)
        q, k, v = Split.apply(qkv, (self.local_inner,) * 3, 3)
        updates = self._lora_updates(x)
        if updates is not None:
            q, k, v = (base + update for base, update in zip((q, k, v), updates))
        split_heads = ttml.ops.multi_head_utils.split_heads
        q = self.norm_q(split_heads(q, self.local_heads))
        k = self.norm_k(split_heads(k, self.local_heads))
        v = split_heads(v, self.local_heads)
        if rope_params is not None:
            q = ttml.ops.rope.rope(q, rope_params, 0)
            k = ttml.ops.rope.rope(k, rope_params, 0)
        attn = self.sdpa(q, k, v, mask)
        return self.to_out(ttml.ops.multi_head_utils.heads_fusion(attn))


class H3FeedForward(AbstractModuleBase):
    """SwiGLU with the fused `proj` of the checkpoint: first half value, second half gate."""

    def __init__(self, config: H3Config) -> None:
        super().__init__()
        tp = _tp_size(config)
        if config.ffn_dim % tp:
            raise ValueError(f"ffn_dim {config.ffn_dim} not divisible by tp {tp}")
        self.local_ffn = config.ffn_dim // tp
        dim = config.hidden_size
        col_init, row_init = _empty_init(config, 2), _empty_init(config, 3)
        if tp > 1:
            self.proj = ColumnParallelLinear(
                dim, 2 * config.ffn_dim, has_bias=False, weight_init=col_init, gather_output=False, axis_name="tp"
            )
            self.out = RowParallelLinear(
                config.ffn_dim, dim, has_bias=False, weight_init=row_init, input_is_parallel=True, axis_name="tp"
            )
        else:
            self.proj = LinearLayer(dim, 2 * config.ffn_dim, False, weight_init=col_init)
            self.out = LinearLayer(config.ffn_dim, dim, False, weight_init=row_init)

    def forward(self, x):
        value, gate = Split.apply(self.proj(x), (self.local_ffn, self.local_ffn), 3)
        return self.out(value * ttml.ops.unary.silu(gate))


class H3RefinerBlock(AbstractModuleBase):
    def __init__(self, config: H3Config) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, config.norm_eps)
        self.attn = H3Attention(config)
        self.norm2 = RMSNorm(config.hidden_size, config.norm_eps)
        self.ff = H3FeedForward(config)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.ff(self.norm2(x))


class H3Block(AbstractModuleBase):
    """Pre-norm block with per-row AdaLN modulation from the (timestep, modality) table."""

    def __init__(self, config: H3Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm1 = RMSNorm(config.hidden_size, config.norm_eps)
        self.attn = H3Attention(config)
        self.norm2 = RMSNorm(config.hidden_size, config.norm_eps)
        self.ff = H3FeedForward(config)
        out = 6 * MODALITY_NUM * config.hidden_size
        w_init, b_init = _empty_init(config, 2), _empty_init(config, 3)
        if _tp_size(config) > 1:
            self.adaln = ColumnParallelLinear(
                config.time_embed_dim,
                out,
                has_bias=True,
                weight_init=w_init,
                bias_init=b_init,
                gather_output=True,
                axis_name="tp",
            )
        else:
            self.adaln = LinearLayer(config.time_embed_dim, out, True, weight_init=w_init, bias_init=b_init)

    def modulation(self, silu_temb, sel_adaln: ttnn.Tensor):
        """Six (1, 1, S_pad, D) constants, one row per sequence row. Frozen path: raw ttnn.

        `adaln(silu_temb)` is (T_pad, 3 * 6D): per timestep row, the three modality blocks of six chunks. Stacking
        the modality blocks as rows gives (3 * T_pad, 6D) addressed by `tag * T_pad + timestep_index`, which the
        one-hot `sel_adaln` gathers in one matmul."""
        table = self.adaln(silu_temb).get_value()
        width = 6 * self.hidden_size
        blocks = [
            ttnn.slice(table, [0, 0, 0, g * width], [1, 1, table.shape[2], (g + 1) * width])
            for g in range(MODALITY_NUM)
        ]
        stacked = ttnn.concat(blocks, 2)
        rows = ttnn.matmul(sel_adaln, stacked)
        shift1, scale1, gate1, shift2, scale2, gate2 = ttnn.chunk(rows, 6, 3)
        return (
            const(shift1),
            const(ttnn.add(scale1, 1.0)),
            const(gate1),
            const(shift2),
            const(ttnn.add(scale2, 1.0)),
            const(gate2),
        )

    def forward(self, x, mods, mask, rope_params):
        shift1, scale1, gate1, shift2, scale2, gate2 = mods
        n = self.norm1(x) * scale1 + shift1
        x = x + self.attn(n, mask, rope_params) * gate1
        n = self.norm2(x) * scale2 + shift2
        return x + self.ff(n) * gate2


@dataclass
class H3Inputs:
    """One packed sample, uploaded. Row order `[text L | audio A | video V | pad]`."""

    L: int
    A: int
    V: int
    S: int
    S_pad: int
    L_pad: int
    V_pad: int
    text_embeds: ttml.autograd.Tensor  # (1, 1, L_pad, text_dim)
    audio: ttml.autograd.Tensor  # (1, 1, 32, audio_in)
    x_t: ttml.autograd.Tensor  # (1, 1, V_pad, 96)
    text_mask: ttml.autograd.Tensor  # (1, 1, L_pad, L_pad)
    mask: ttml.autograd.Tensor  # (1, 1, S_pad, S_pad)
    silu_temb: ttml.autograd.Tensor  # (1, 1, 32, time_embed_dim), silu applied in fp32 on host
    sel_adaln: ttnn.Tensor  # (1, 1, S_pad, 96)
    sel_t: ttnn.Tensor  # (1, 1, S_pad, 32)
    rope_params: object
    video_row_mask: np.ndarray  # (S_pad,) 1.0 on video rows, host

    @classmethod
    def build(
        cls,
        config: H3Config,
        time_weights: dict[str, np.ndarray],
        x_t: np.ndarray,
        audio: np.ndarray,
        text_embeds: np.ndarray,
        layout: Layout,
        timesteps: np.ndarray,
        timestep_indices: np.ndarray,
    ) -> "H3Inputs":
        L, A, V = len(layout.text_indices), len(layout.audio_indices), len(layout.video_indices)
        S = layout.sequence_length
        S_pad, L_pad, V_pad = tile_pad(S), tile_pad(L), tile_pad(V)
        T = int(timesteps.shape[0])
        if T > TILE:
            raise ValueError(f"{T} distinct timesteps > {TILE}")

        def pad_rows(a: np.ndarray, rows: int) -> np.ndarray:
            out = np.zeros((1, 1, rows, a.shape[-1]), dtype=np.float32)
            out[0, 0, : a.shape[0]] = a.astype(np.float32)
            return out

        temb = time_embedding(timesteps, time_weights)  # (T, 2688) fp32
        silu_temb = temb / (1.0 + np.exp(-temb))

        idx = adaln_indices(layout, timestep_indices)  # timestep_index * 3 + tag
        tag, ti = idx % MODALITY_NUM, idx // MODALITY_NUM
        sel_adaln = selection_matrix(tag * TILE + ti, S_pad, MODALITY_NUM * TILE, fill=int(ti[-1]))
        sel_t = selection_matrix(timestep_indices, S_pad, TILE, fill=int(timestep_indices[-1]))

        cos, sin = rope_tables(
            layout.position_ids, S_pad, config.attention_head_dim, config.rope_freq_dim, config.rope_theta
        )
        params = ttml.ops.rope.RotaryEmbeddingParams()
        params.cos_cache = upload(cos[None, None]).get_value()
        params.sin_cache = upload(sin[None, None]).get_value()
        params.neg_cos_cache = upload(cos[None, None]).get_value()
        params.neg_sin_cache = upload(-sin[None, None]).get_value()
        params.trans_mat = upload(rope_trans_mat()).get_value()
        params.sequence_length = S_pad
        params.head_dim = config.attention_head_dim
        params.theta = config.rope_theta

        row_mask = np.zeros(S_pad, dtype=np.float32)
        row_mask[layout.video_indices] = 1.0
        return cls(
            L=L,
            A=A,
            V=V,
            S=S,
            S_pad=S_pad,
            L_pad=L_pad,
            V_pad=V_pad,
            text_embeds=upload(pad_rows(text_embeds, L_pad)),
            audio=upload(pad_rows(audio, TILE)),
            x_t=upload(pad_rows(x_t, V_pad)),
            text_mask=upload(attention_mask(L, L_pad)),
            mask=upload(attention_mask(S, S_pad)),
            silu_temb=upload(pad_rows(silu_temb, TILE)),
            sel_adaln=upload(sel_adaln[None, None]).get_value(),
            sel_t=upload(sel_t[None, None]).get_value(),
            rope_params=params,
            video_row_mask=row_mask,
        )


class H3Transformer(AbstractModuleBase):
    def __init__(self, config: H3Config) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_size
        self.proj_in = LinearLayer(config.video_in_channels, D, True)
        self.audio_proj_in = LinearLayer(config.audio_in_channels, D, True)
        self.context_embedder = LinearLayer(config.text_dim, D, True)
        self.refiner_blocks = ModuleList([H3RefinerBlock(config) for _ in range(config.num_refiner_layers)])
        self.refiner_norm = RMSNorm(D, config.final_norm_eps)
        self.blocks = ModuleList([H3Block(config) for _ in range(config.num_layers)])
        self.norm_out = RMSNorm(D, config.final_norm_eps)
        w_init, b_init = _empty_init(config, 2), _empty_init(config, 3)
        if _tp_size(config) > 1:
            self.norm_out_linear = ColumnParallelLinear(
                config.time_embed_dim,
                2 * D,
                has_bias=True,
                weight_init=w_init,
                bias_init=b_init,
                gather_output=True,
                axis_name="tp",
            )
        else:
            self.norm_out_linear = LinearLayer(config.time_embed_dim, 2 * D, True, weight_init=w_init, bias_init=b_init)
        self.proj_out = LinearLayer(D, config.video_in_channels, True)
        self.audio_proj_out = LinearLayer(D, config.audio_in_channels, True)

    def embed_text(self, inputs: H3Inputs):
        x = self.context_embedder(inputs.text_embeds)
        for block in self.refiner_blocks:
            x = block(x, inputs.text_mask)
        return self.refiner_norm(x)

    def pack(self, text, audio, video, inputs: H3Inputs) -> ttml.autograd.Tensor:
        """`[text L | audio A | video V | zeros]` as one (1, 1, S_pad, D) constant. Nothing upstream trains."""
        rm = ttnn.Layout.ROW_MAJOR

        def rows(t, n):
            v = ttnn.to_layout(t.get_value(), rm)
            return ttnn.slice(v, [0, 0, 0, 0], [1, 1, n, v.shape[3]])

        parts = [rows(text, inputs.L), rows(audio, inputs.A), rows(video, inputs.V)]
        packed = ttnn.concat(parts, 2)
        if inputs.S_pad > inputs.S:
            packed = ttnn.pad(packed, [(0, 0), (0, 0), (0, inputs.S_pad - inputs.S), (0, 0)], 0.0)
        return const(ttnn.to_layout(packed, ttnn.Layout.TILE))

    def run_blocks(self, h, inputs: H3Inputs):
        for block in self.blocks:
            mods = block.modulation(inputs.silu_temb, inputs.sel_adaln)
            h = block(h, mods, inputs.mask, inputs.rope_params)
        return h

    def head(self, h, inputs: H3Inputs, with_audio: bool = True):
        """`norm_out` modulated per row by its timestep; (video_out, audio_out) over all rows, audio optional."""
        table = self.norm_out_linear(inputs.silu_temb).get_value()  # (1, 1, 32, 2D): shift | scale
        shift_t, scale_t = ttnn.chunk(table, 2, 3)
        shift = const(ttnn.matmul(inputs.sel_t, shift_t))
        scale = const(ttnn.add(ttnn.matmul(inputs.sel_t, scale_t), 1.0))
        o = self.norm_out(h) * scale + shift
        return self.proj_out(o), (self.audio_proj_out(o) if with_audio else None)

    def forward(self, inputs: H3Inputs, with_audio: bool = True):
        text = self.embed_text(inputs)
        video = self.proj_in(inputs.x_t)
        audio = self.audio_proj_in(inputs.audio)
        h = self.pack(text, audio, video, inputs)
        h = self.run_blocks(h, inputs)
        return self.head(h, inputs, with_audio)
