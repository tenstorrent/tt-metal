# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the XTTS-v2 audio conditioning path.

Mirrors ``reference/xtts_conditioning.py``: ``ConditioningEncoder`` (init conv +
6 attention blocks) followed by ``PerceiverResampler`` (32 latents, depth 2),
producing the GPT conditioning latents ``[b, 1024, 32]`` from a mel ``[b, 80, s]``.

Everything runs in ``[batch, seq, channels]`` (tokens x channels) layout so the
per-timestep ``Conv1d(k=1)`` layers become plain ``ttnn.linear`` and attention is
``ttnn.transformer.scaled_dot_product_attention``. Key equivalences used:

  * ConditioningEncoder QKV scale ``1/sqrt(sqrt(ch))`` on both q and k == the
    standard ``1/sqrt(head_dim)`` SDPA scale (default, non-causal).
  * The perceiver's ``F.normalize(x, dim=-1) * sqrt(dim) * gamma`` == ``ttnn.rms_norm``.

GroupNorm(32, 1024) is computed manually: reshaping ``[1024, s] -> [32, 32*s]``
makes both dims tile-aligned (no reduction-over-padding), so no group_norm masks /
sharding are needed. Stats are taken on centered values for stability.
"""

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_gpt_block import HIDDEN_SIZE
from models.experimental.xtts.reference.xtts_conditioning import (
    NUM_ATTN_HEADS,
    NUM_LATENTS,
)

GROUP_NORM_GROUPS = 32
GROUP_NORM_EPS = 1e-5
ENC_HEAD_DIM = HIDDEN_SIZE // NUM_ATTN_HEADS  # 64
PERCEIVER_HEADS = 8
PERCEIVER_HEAD_DIM = 64
PERCEIVER_DEPTH = 2


def _lin(torch_tensor, device):
    """torch [out, in] (or conv [out, in, 1]) -> ttnn linear weight [in, out] on device."""
    w = torch_tensor
    if w.dim() == 3:  # conv1d kernel-1 -> [out, in]
        w = w.squeeze(-1)
    return ttnn.from_torch(
        w.t().contiguous().to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )


def _vec(torch_tensor, device):
    """torch [n] -> ttnn tile [n] on device (bias / affine params)."""
    return ttnn.from_torch(torch_tensor.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)


class TtXttsConditioning(LightweightModule):
    def __init__(self, state_dict, device):
        super().__init__()
        self.device = device
        e = "gpt.conditioning_encoder."
        p = "gpt.conditioning_perceiver."

        # Block-diagonal group-averaging matrix E [1024, 1024] (E[c,c'] = 1/cpg iff channels c,c'
        # share a group) used by _group_norm to reduce per-group WITHOUT a reshape to [1,32,32s]
        # (that reshape needed ROW_MAJOR<->TILE conversions = Tilize/Untilize ops every block).
        cpg = HIDDEN_SIZE // GROUP_NORM_GROUPS
        e_mat = torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE)
        for gi in range(GROUP_NORM_GROUPS):
            e_mat[gi * cpg : (gi + 1) * cpg, gi * cpg : (gi + 1) * cpg] = 1.0 / cpg
        self._gn_expand = ttnn.from_torch(
            e_mat.reshape(1, HIDDEN_SIZE, HIDDEN_SIZE).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16,
        )

        # --- ConditioningEncoder ---
        self.init_w = _lin(state_dict[e + "init.weight"], device)  # [80 -> 1024]
        self.init_b = _vec(state_dict[e + "init.bias"], device)

        self.blocks = []
        i = 0
        while (e + f"attn.{i}.qkv.weight") in state_dict:
            self.blocks.append(
                {
                    "gn_w": _vec(state_dict[e + f"attn.{i}.norm.weight"], device),
                    "gn_b": _vec(state_dict[e + f"attn.{i}.norm.bias"], device),
                    "qkv_w": _lin(state_dict[e + f"attn.{i}.qkv.weight"], device),  # [1024 -> 3072]
                    "qkv_b": _vec(state_dict[e + f"attn.{i}.qkv.bias"], device),
                    "proj_w": _lin(state_dict[e + f"attn.{i}.proj_out.weight"], device),  # [1024 -> 1024]
                    "proj_b": _vec(state_dict[e + f"attn.{i}.proj_out.bias"], device),
                }
            )
            i += 1

        # --- PerceiverResampler ---
        self.latents = ttnn.from_torch(
            state_dict[p + "latents"].to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
        )  # [32, 1024]
        self.layers = []
        for j in range(PERCEIVER_DEPTH):
            self.layers.append(
                {
                    "to_q": _lin(state_dict[p + f"layers.{j}.0.to_q.weight"], device),  # [1024 -> 512]
                    "to_kv": _lin(state_dict[p + f"layers.{j}.0.to_kv.weight"], device),  # [1024 -> 1024]
                    "to_out": _lin(state_dict[p + f"layers.{j}.0.to_out.weight"], device),  # [512 -> 1024]
                    "ff0_w": _lin(state_dict[p + f"layers.{j}.1.0.weight"], device),  # [1024 -> 5460]
                    "ff0_b": _vec(state_dict[p + f"layers.{j}.1.0.bias"], device),
                    "ff2_w": _lin(state_dict[p + f"layers.{j}.1.2.weight"], device),  # [2730 -> 1024]
                    "ff2_b": _vec(state_dict[p + f"layers.{j}.1.2.bias"], device),
                }
            )
        self.perc_norm_gamma = _vec(state_dict[p + "norm.gamma"], device)

    # ------------------------------------------------------------------ #
    def _group_norm(self, x, gamma, beta):
        """GroupNorm(32, 1024) over (channels-in-group, seq). x: [1, s, 1024] -> [1, s, 1024].

        Reshape-FREE: a full group mean/var is order-independent, so per-group stats == the group
        average of per-channel stats. Compute the per-channel mean over seq (``mean(dim=-1)`` — a
        TILE reduction), expand to per-group via a matmul with the block-diagonal averaging matrix
        ``self._gn_expand``, and likewise for the (centered) variance. Everything stays TILE, so
        this avoids the old reshape-to-[1,32,32s] round trip and its four Tilize/Untilize ops."""
        xt = ttnn.permute(x, (0, 2, 1))  # [1, 1024, s]
        cmean = ttnn.mean(xt, dim=-1, keepdim=True)  # [1, 1024, 1] per-channel mean over seq
        mu = ttnn.matmul(self._gn_expand, cmean)  # [1, 1024, 1] group mean, expanded per channel
        xc = ttnn.subtract(xt, mu)  # center by group mean (stable variance)
        cvar = ttnn.mean(ttnn.multiply(xc, xc), dim=-1, keepdim=True)  # [1, 1024, 1] per-channel var
        var = ttnn.matmul(self._gn_expand, cvar)  # [1, 1024, 1] group variance
        xn = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, GROUP_NORM_EPS)))  # [1, 1024, s]
        xn = ttnn.permute(xn, (0, 2, 1))  # [1, s, 1024]
        return ttnn.add(ttnn.multiply(xn, gamma), beta)  # gamma/beta broadcast over seq

    def _split_heads(self, x, heads, head_dim):  # [1, n, heads*head_dim] -> [1, heads, n, head_dim]
        b, n, _ = x.shape
        x = ttnn.reshape(x, (b, n, heads, head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _merge_heads(self, x):  # [1, heads, n, head_dim] -> [1, n, heads*head_dim]
        b, h, n, d = x.shape
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.reshape(x, (b, n, h * d))

    def _attn_block(self, x, blk):
        """One ConditioningEncoder AttentionBlock: y = gn(x); y + proj(attn(qkv(y)))."""
        y = self._group_norm(x, blk["gn_w"], blk["gn_b"])
        qkv = ttnn.linear(y, blk["qkv_w"], bias=blk["qkv_b"])  # [1, s, 3072]
        b, s, _ = qkv.shape
        # channels are laid out per head as [h0:q,k,v | h1:q,k,v | ...]
        qkv = ttnn.reshape(qkv, (b, s, NUM_ATTN_HEADS, 3 * ENC_HEAD_DIM))
        q = ttnn.slice(qkv, [0, 0, 0, 0], [b, s, NUM_ATTN_HEADS, ENC_HEAD_DIM])
        k = ttnn.slice(qkv, [0, 0, 0, ENC_HEAD_DIM], [b, s, NUM_ATTN_HEADS, 2 * ENC_HEAD_DIM])
        v = ttnn.slice(qkv, [0, 0, 0, 2 * ENC_HEAD_DIM], [b, s, NUM_ATTN_HEADS, 3 * ENC_HEAD_DIM])
        q = ttnn.permute(q, (0, 2, 1, 3))  # [1, heads, s, head_dim]
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = self._merge_heads(attn)  # [1, s, 1024]
        h = ttnn.linear(out, blk["proj_w"], bias=blk["proj_b"])
        return ttnn.add(y, h)

    def _perceiver_attn(self, latents, context, layer):
        """Cross-attention: latents attend to [latents ; context]."""
        ctx = ttnn.concat([latents, context], dim=1)  # [1, 32+s, 1024]
        q = ttnn.linear(latents, layer["to_q"])  # [1, 32, 512]
        kv = ttnn.linear(ctx, layer["to_kv"])  # [1, 32+s, 1024]
        n_kv = kv.shape[1]
        k = ttnn.slice(kv, [0, 0, 0], [1, n_kv, PERCEIVER_HEADS * PERCEIVER_HEAD_DIM])
        v = ttnn.slice(
            kv, [0, 0, PERCEIVER_HEADS * PERCEIVER_HEAD_DIM], [1, n_kv, 2 * PERCEIVER_HEADS * PERCEIVER_HEAD_DIM]
        )
        q = self._split_heads(q, PERCEIVER_HEADS, PERCEIVER_HEAD_DIM)  # [1, 8, 32, 64]
        k = self._split_heads(k, PERCEIVER_HEADS, PERCEIVER_HEAD_DIM)  # [1, 8, 32+s, 64]
        v = self._split_heads(v, PERCEIVER_HEADS, PERCEIVER_HEAD_DIM)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)  # [1, 8, 32, 64]
        out = self._merge_heads(attn)  # [1, 32, 512]
        return ttnn.linear(out, layer["to_out"])  # [1, 32, 1024]

    def _perceiver_ff(self, x, layer):
        h = ttnn.linear(x, layer["ff0_w"], bias=layer["ff0_b"])  # [1, 32, 5460]
        inner = h.shape[-1] // 2
        a = ttnn.slice(h, [0, 0, 0], [1, h.shape[1], inner])  # [1,32,2730]
        gate = ttnn.slice(h, [0, 0, inner], [1, h.shape[1], 2 * inner])
        h = ttnn.multiply(ttnn.gelu(gate, fast_and_approximate_mode=False), a)  # GEGLU (exact gelu)
        return ttnn.linear(h, layer["ff2_w"], bias=layer["ff2_b"])  # [1, 32, 1024]

    # ------------------------------------------------------------------ #
    def mel_to_device(self, mel):
        """Host log-mel ``[1, 80, s]`` -> device bf16 TILE tensor (the ``from_torch`` host->device
        write, kept OUTSIDE any trace capture — writes are fatal inside a trace)."""
        return ttnn.from_torch(mel.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)

    def forward(self, mel):
        """mel: torch tensor ``[1, 80, s]`` -> conditioning latents ttnn ``[1, 1024, 32]``."""
        return self.forward_dev(self.mel_to_device(mel))

    def forward_dev(self, mel_tt):
        """Trace-compatible: ``mel_tt`` is an already-on-device ``[1, 80, s]`` bf16 tensor (no
        host->device write here), so this can run inside a captured trace. -> ttnn ``[1, 1024, 32]``."""
        x = ttnn.permute(mel_tt, (0, 2, 1))  # [1, s, 80]
        x = ttnn.linear(x, self.init_w, bias=self.init_b)  # [1, s, 1024]

        for blk in self.blocks:
            x = self._attn_block(x, blk)  # ConditioningEncoder output [1, s, 1024]

        # PerceiverResampler
        latents = ttnn.reshape(self.latents, (1, NUM_LATENTS, HIDDEN_SIZE))
        for layer in self.layers:
            latents = ttnn.add(self._perceiver_attn(latents, x, layer), latents)
            latents = ttnn.add(self._perceiver_ff(latents, layer), latents)
        latents = ttnn.rms_norm(latents, weight=self.perc_norm_gamma, epsilon=1e-12)  # [1, 32, 1024]

        return ttnn.permute(latents, (0, 2, 1))  # [1, 1024, 32]
