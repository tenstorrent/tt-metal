# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Class-conditional DiT implemented on ttml (pure Python).

Mirrors reference_torch.py 1:1. The model works entirely in patch-token
space: patchify/unpatchify, the DDPM schedule, and sinusoidal timestep
features all live on the host (see diffusion.py). Rank-4 shapes throughout:

    tokens  [B, 1, T, in_dim]   noisy patch tokens
    t_feats [B, 1, 1, dim]      sinusoidal timestep features
    labels  [B, 1, 1, 1]        uint32 class ids (num_classes = CFG null)
    output  [B, 1, T, in_dim]   predicted noise tokens

adaLN modulation uses six separate zero-init linears per block (ttml has no
autograd split). Conditioning tensors are [B, 1, 1, dim] and rely on ttnn
row-broadcast in binary add/mul against [B, 1, T, dim] activations; if that
ever regresses, see broadcast_rows() below for a matmul-based fallback.
"""

from __future__ import annotations

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, Parameter
from ttml.modules.module_base import ModuleList


def broadcast_rows(c: "ttml.autograd.Tensor", num_tokens: int) -> "ttml.autograd.Tensor":
    """[B,1,1,D] -> [B,1,T,D] via ones[T,1] @ c[1,D]; autograd-correct
    broadcast fallback for platforms where binary-op row broadcast fails."""
    import numpy as np

    ones = ttml.autograd.Tensor.from_numpy(np.ones((1, 1, num_tokens, 1), dtype=np.float32))
    return ttml.ops.matmul.matmul_op(ones, c)


class LayerNorm(AbstractModuleBase):
    """LayerNorm with gamma/beta initialized to identity (ones/zeros).

    The torch reference uses no-affine LN; identical at init, and the
    modulation supplies the effective affine either way.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        shape = (1, 1, 1, dim)
        self.gamma = Parameter(ttml.init.ones()(shape))
        self.beta = Parameter(ttml.init.zeros()(shape))

    def forward(self, x):
        return ttml.ops.layernorm.layernorm(x, self.gamma.tensor, self.beta.tensor)


class Modulation(AbstractModuleBase):
    """One adaLN branch: SiLU(c) -> Linear(dim, dim), zero-init."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = LinearLayer(dim, dim, True, weight_init=ttml.init.zeros(), bias_init=ttml.init.zeros())

    def forward(self, c):
        return self.linear(ttml.ops.unary.silu(c))


class SliceLastDim(ttml.autograd.Function):
    """Autograd slice on the last dim: ttnn.slice fwd, ttnn.pad bwd.

    Fills the no-autograd-split gap so adaLN can use one fused D->n*D linear
    (canonical DiT form) instead of n separate linears. Slice boundaries must
    be tile-aligned (multiples of 32), which holds for any dim % 32 == 0.
    """

    @staticmethod
    def forward(ctx, x, start, size):
        import ttnn

        v = x.get_value()
        shape = list(v.shape)
        ctx.start, ctx.size, ctx.total = start, size, shape[-1]
        begins = [0] * len(shape)
        ends = shape[:-1] + [0]
        begins[-1], ends[-1] = start, start + size
        return ttnn.slice(v, begins, ends)

    @staticmethod
    def backward(ctx, grad_output):
        import ttnn

        # ttnn.pad on TILE layout cannot front-pad, so place the grad at its
        # offset via concat with zero blocks. All widths are tile-aligned.
        def zeros(width):
            shape = list(grad_output.shape)
            shape[-1] = width
            return ttnn.zeros(shape, dtype=grad_output.dtype, layout=grad_output.layout, device=grad_output.device())

        parts = []
        if ctx.start > 0:
            parts.append(zeros(ctx.start))
        parts.append(grad_output)
        back = ctx.total - ctx.start - ctx.size
        if back > 0:
            parts.append(zeros(back))
        # Exactly one gradient: only x is a tensor input (start/size are ints,
        # which Function.apply filters out of the tensor-input list).
        return ttnn.concat(parts, dim=-1) if len(parts) > 1 else grad_output


class FusedModulation(AbstractModuleBase):
    """n adaLN branches sharing one SiLU: SiLU(c) -> n zero-init linears.

    NOTE(perf): a truly fused form (1 linear D->n*D + n slices via SliceLastDim)
    measured 6.5x SLOWER — the training step is dispatch-bound and eager
    ttnn.slice/concat/zeros cost more than the small matmuls they replace.
    Sharing the SiLU keeps op count strictly lower than n full branches.
    A single fused modulation kernel (tt-lang) is the real fix later.
    """

    def __init__(self, dim: int, n: int, scale_slots: tuple[int, ...] = ()) -> None:
        super().__init__()
        self.dim, self.n = dim, n
        # Scale branches get bias init = 1 so the linear emits (1 + scale)
        # directly — saves a broadcast-add (+ its backward) per modulate site.
        self.branches = ModuleList(
            [
                LinearLayer(
                    dim, dim, True,
                    weight_init=ttml.init.zeros(),
                    bias_init=ttml.init.ones() if i in scale_slots else ttml.init.zeros(),
                )
                for i in range(n)
            ]
        )

    def forward(self, c):
        s = ttml.ops.unary.silu(c)
        return [branch(s) for branch in self.branches]


class Attention(AbstractModuleBase):
    """Non-causal MHA.

    Two paths: composite SDPA with mask=None (true non-causal, multi-dispatch)
    or the fused SDPA kernel with an explicit all-ones Arbitrary mask
    (validated numerically equal, max err 0.006). The fused kernel treats
    mask=None as CAUSAL, hence the ones mask.
    """

    _ones_masks: dict = {}

    def __init__(self, dim: int, num_heads: int, use_fused: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.use_fused = use_fused
        self.qkv = LinearLayer(dim, dim * 3, True, weight_init=ttml.init.normal(0.0, 0.02), bias_init=ttml.init.zeros())
        self.proj = LinearLayer(dim, dim, True, weight_init=ttml.init.normal(0.0, 0.02), bias_init=ttml.init.zeros())

    def forward(self, x):
        import numpy as np

        qkv = self.qkv(x)
        q, k, v = ttml.ops.multi_head_utils.heads_creation(qkv, self.num_heads)
        if self.use_fused:
            seq = x.shape()[2]
            mask = Attention._ones_masks.get(seq)
            if mask is None:
                mask = ttml.autograd.Tensor.from_numpy(np.ones((1, 1, seq, seq), dtype=np.float32))
                Attention._ones_masks[seq] = mask
            out = ttml.ops.attention.scaled_dot_product_attention(q, k, v, mask)
        else:
            out = ttml.ops.attention.scaled_dot_product_attention_composite(q, k, v, None)
        return self.proj(ttml.ops.multi_head_utils.heads_fusion(out))


class MLP(AbstractModuleBase):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = LinearLayer(dim, hidden, True, weight_init=ttml.init.normal(0.0, 0.02), bias_init=ttml.init.zeros())
        self.fc2 = LinearLayer(hidden, dim, True, weight_init=ttml.init.normal(0.0, 0.02), bias_init=ttml.init.zeros())

    def forward(self, x):
        return self.fc2(ttml.ops.unary.gelu(self.fc1(x)))


def _modulate(x_norm, scale1p, shift):
    """x_norm * scale1p + shift, with scale1p/shift [B,1,1,D].

    scale1p is (1 + scale): the scale branch's bias is initialized to ones,
    so no explicit +1 op is needed.
    """
    add, mul = ttml.ops.binary.add, ttml.ops.binary.mul
    return add(mul(x_norm, scale1p), shift)


class DiTBlock(AbstractModuleBase):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, use_fused_sdpa: bool = False) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, use_fused=use_fused_sdpa)
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        # order: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.modulation = FusedModulation(dim, 6, scale_slots=(1, 4))

    def forward(self, x, c):
        add, mul = ttml.ops.binary.add, ttml.ops.binary.mul
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(c)
        h = _modulate(self.norm1(x), scale_msa, shift_msa)
        x = add(x, mul(self.attn(h), gate_msa))
        h = _modulate(self.norm2(x), scale_mlp, shift_mlp)
        x = add(x, mul(self.mlp(h), gate_mlp))
        return x


class DiT(AbstractModuleBase):
    def __init__(
        self,
        in_dim: int,
        dim: int,
        depth: int,
        num_heads: int,
        num_tokens: int,
        num_classes: int,
        mlp_ratio: float = 4.0,
        use_fused_sdpa: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.num_classes = num_classes

        self.patch_proj = LinearLayer(
            in_dim, dim, True, weight_init=ttml.init.normal(0.0, 0.02), bias_init=ttml.init.zeros()
        )
        self.pos_emb = Parameter(ttml.init.normal(0.0, 0.02)((1, 1, num_tokens, dim)))
        self.t_fc1 = LinearLayer(dim, dim, True, weight_init=ttml.init.normal(0.0, 0.02), bias_init=ttml.init.zeros())
        self.t_fc2 = LinearLayer(dim, dim, True, weight_init=ttml.init.normal(0.0, 0.02), bias_init=ttml.init.zeros())
        # One-hot labels through a bias-free linear (== an embedding table).
        # ttnn's embedding_backward needs ids' last dim % 32 == 0, which a
        # single per-image label can't satisfy; one-hot @ W has no such limit.
        self.label_emb = LinearLayer(num_classes + 1, dim, False, weight_init=ttml.init.normal(0.0, 0.02))
        self.blocks = ModuleList([DiTBlock(dim, num_heads, mlp_ratio, use_fused_sdpa) for _ in range(depth)])
        self.final_norm = LayerNorm(dim)
        self.final_modulation = FusedModulation(dim, 2, scale_slots=(1,))  # (shift, scale1p)
        self.final_proj = LinearLayer(dim, in_dim, True, weight_init=ttml.init.zeros(), bias_init=ttml.init.zeros())

    def forward(self, tokens, t_feats, labels_onehot):
        add = ttml.ops.binary.add
        x = add(self.patch_proj(tokens), self.pos_emb.tensor)
        t_emb = self.t_fc2(ttml.ops.unary.silu(self.t_fc1(t_feats)))
        c = add(t_emb, self.label_emb(labels_onehot))
        for block in self.blocks:
            x = block(x, c)
        final_shift, final_scale = self.final_modulation(c)
        x = _modulate(self.final_norm(x), final_scale, final_shift)
        return self.final_proj(x)


def dit_s(in_dim: int, num_tokens: int, num_classes: int) -> DiT:
    return DiT(in_dim=in_dim, dim=384, depth=12, num_heads=6, num_tokens=num_tokens, num_classes=num_classes)


def dit_tiny(in_dim: int, num_tokens: int, num_classes: int) -> DiT:
    return DiT(in_dim=in_dim, dim=128, depth=4, num_heads=4, num_tokens=num_tokens, num_classes=num_classes)
