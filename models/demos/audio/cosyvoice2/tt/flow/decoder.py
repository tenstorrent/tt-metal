# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""CosyVoice2's flow-matching estimator: `CausalConditionalDecoder` (the U-Net that
predicts the ODE's vector field) and `CausalConditionalCFM` (the fixed-noise Euler
solver that drives it), confirmed against real upstream source directly --
`cosyvoice/flow/decoder.py`, `cosyvoice/flow/flow_matching.py`, and the `matcha-tts`
components CosyVoice imports rather than reimplements
(`matcha/models/components/{decoder,transformer,flow_matching}.py`) -- not assumed
from CosyVoice1's non-causal `ConditionalDecoder`/`ConditionalCFM`.

**Causality is real new work here, not a pass-through** (unlike the vocoder, where
the causal/non-causal split turned out not to matter structurally):

* `CausalBlock1D`/`CausalResnetBlock1D` don't just left-pad their convs -- they swap
  `GroupNorm` for `LayerNorm`. GroupNorm's statistics span the whole sequence per
  sample; a streaming decoder needs a per-timestep norm so a longer prefix's early
  output doesn't change. This is a real output-value difference from CosyVoice1's
  decoder, not just an added mask.
* `CausalConditionalCFM` draws its ODE's initial noise from a **fixed, pre-seeded
  buffer** (`set_all_random_seed(0)`; `torch.randn([1, 80, 50*300])`, sliced to
  length) instead of a fresh `torch.randn_like(mu)` per call, and it drops the
  `cache`/`flow_cache` overlap-splicing that CosyVoice1's `ConditionalCFM.forward`
  has entirely. The fixed buffer is what makes a longer chunk's noise a genuine
  continuation of a shorter chunk's -- both facts are essential to replicate exactly
  for a correctness claim, not simplifications to make on our own.
* `CausalMaskedDiffWithXvec` (the caller, a separate, later module -- see
  `tt/flow/flow.py`) has no `length_regulator` at all: the token-rate -> mel-rate
  upsampling moves inside the encoder (`UpsampleConformerEncoder`'s own `Upsample1D`),
  which is a real architecture-shape difference, not addressed here.

**What genuinely transfers unchanged** (the vocoder experience repeats for these
specific pieces): `BasicTransformerBlock` itself is identical causal or not -- only
the `attention_mask` argument differs; it is plain LayerNorm + self-attention + GELU
FFN, no AdaLN (CosyVoice never sets `norm_type="ada_norm"`, so the `timestep` kwarg
each block receives is accepted but unused -- time conditioning only enters through
`CausalResnetBlock1D`'s additive FiLM-style bias). `SinusoidalPosEmb`/
`TimestepEmbedding` are untouched. The 1x1 `res_conv` inside each resnet block is
trivially causal already (kernel=1).

**Scope of this module, verified from the real checkpoint's own config**
(`cosyvoice2.yaml`'s `flow.decoder.estimator`, not the class's generic default):
`channels=[256]` -- a SINGLE down/mid/up stage, not the `CausalConditionalDecoder`
class's own two-stage default `(256, 256)`. Tracing `CausalConditionalDecoder.__init__`'s
arithmetic at this config shows `is_last` is always `True` in both the down-block and
up-block loops (each loop runs exactly once), so the non-causal `Downsample1D`/
`matcha.Upsample1D(use_conv_transpose=True)` classes are **never instantiated** for
this checkpoint -- only `CausalConv1d` (kernel=3, stride=1, shape-preserving) is ever
used for the down/up legs. There is no real spatial down/up-sampling anywhere in this
U-Net: it is a flat-resolution stack (1 causal down-block + 12 mid-blocks + 1 causal
up-block, all at the input's own temporal resolution, `channels=[256]` throughout,
`n_blocks=4` transformer blocks per stage). This module hardcodes that verified
1-stage topology rather than the class's general N-stage form -- if a future
checkpoint ships a different `channels` list, this module needs extending, not
silently mis-porting the wrong shape.

**Attention masking, both modes, as real upstream builds it** (`CausalConditionalDecoder.forward`
-> `add_optional_chunk_mask` -> `mask_to_bias`, `cosyvoice/utils/mask.py` / `common.py`):

* `streaming=False`: `static_chunk_size=0` collapses `add_optional_chunk_mask` to the plain
  padding mask -- the same pattern a non-causal decoder uses.
* `streaming=True`: `masks & subsequent_chunk_mask(T, static_chunk_size)` -- a logical AND of
  the padding mask (per KEY) and the chunk-causal mask (query `i` sees keys
  `[0, (i // 50 + 1) * 50)`), then `(1 - m) * -1e10`. `static_chunk_size = chunk_size *
  token_mel_ratio = 50` (the real `cosyvoice2.yaml`), == `encoder.CHUNK_SIZE_UP`, and the mask
  helpers are the encoder's own (`subsequent_chunk_mask_torch`/`chunk_causal_bias_torch`), not
  re-derived. On device the AND becomes the additive sum `pad_bias[B,1,1,T] +
  chunk_bias[1,1,T,T]` (0 only where both are valid). The padding term is what keeps a VALID
  query in a partial last chunk (e.g. valid length 310 in a 384 bucket: queries 300..309 have a
  chunk window reaching key 349) off the padded keys; at chunk-aligned valid lengths it is
  redundant with the chunk term, which is why aligned test points alone cannot catch its absence.

Every causal conv here is left-padded only (reads backward, never into bucket padding), so
bucket padding needs nothing beyond the existing `mask` for the conv/resnet legs -- only the
attention bias gains the chunk-causal term.

Tensors are `[N, L, C]` throughout, per this package's existing channels-last
convention (conv.py). Upstream's `[B, C, T]` channel-dim concatenations
(`einops.pack([x, mu], "b * t")`) become last-axis concatenations here.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import ttnn

from ..hifigan.conv import TtConv1d, accurate_compute_config
from .encoder import CHUNK_SIZE_UP, chunk_causal_bias_torch, subsequent_chunk_mask_torch

# The real checkpoint's verified estimator config (cosyvoice2.yaml's
# flow.decoder.estimator) -- see module docstring for why this is hardcoded rather
# than generic.
IN_CHANNELS = 320  # x(80) + mu(80) + spks(80, broadcast) + cond(80)
OUT_CHANNELS = 80
CHANNELS = 256
N_BLOCKS = 4  # transformer blocks per resnet stage
NUM_MID_BLOCKS = 12
NUM_HEADS = 8
ATTENTION_HEAD_DIM = 64
INNER_DIM = NUM_HEADS * ATTENTION_HEAD_DIM  # 512 -- an expand-then-project attention
TIME_EMBED_DIM = CHANNELS * 4  # 1024


# ---------------------------------------------------------------------------
# torch reference -- real nn.Module submodules (nn.Conv1d/nn.Linear/nn.LayerNorm),
# real torch.nn.functional ops for every primitive (mish, gelu,
# scaled_dot_product_attention) rather than a hand-derived reimplementation, so a
# comparison against this reference has no "shared bug" risk for those pieces --
# matching this package's existing TorchHiFTDecodeRef pattern (matcha-tts/diffusers
# are not installed in this environment, so importing the real classes directly is
# not available; this is a faithful line-by-line transcription of the real source
# read for this port, not a redesign).
# ---------------------------------------------------------------------------


def sinusoidal_pos_emb_torch(t: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    """`matcha.models.components.decoder.SinusoidalPosEmb.forward`, verbatim."""
    if t.ndim < 1:
        t = t.unsqueeze(0)
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device).float() * -emb)
    emb = scale * t.unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbeddingRef(nn.Module):
    """`matcha.models.components.decoder.TimestepEmbedding`, the `act_fn="silu"`
    path only (the only one CosyVoice uses): Linear -> SiLU -> Linear."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


class CausalConv1dRef(nn.Conv1d):
    """`cosyvoice.flow.decoder.CausalConv1d`: left-pad by `kernel_size - 1`, then a
    plain `padding=0` conv -- so position `i` of the output only ever depends on
    input positions `<= i`."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=bias)
        self.causal_padding = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.causal_padding, 0), value=0.0))


class CausalBlock1DRef(nn.Module):
    """`cosyvoice.flow.decoder.CausalBlock1D`: CausalConv1d -> LayerNorm(over
    channels) -> Mish. GroupNorm -> LayerNorm is the real, causality-motivated swap
    from matcha's plain `Block1D` (see module docstring)."""

    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.conv = CausalConv1dRef(dim, dim_out, 3)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.conv(x * mask)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.mish(x)
        return x * mask


class CausalResnetBlock1DRef(nn.Module):
    """`cosyvoice.flow.decoder.CausalResnetBlock1D` (via matcha's `ResnetBlock1D`
    structure, unchanged -- only its two `Block1D`s become causal)."""

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = CausalBlock1DRef(dim, dim_out)
        self.block2 = CausalBlock1DRef(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, mask)
        h = h + self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        return h + self.res_conv(x * mask)


class BasicTransformerBlockRef(nn.Module):
    """`matcha.models.components.transformer.BasicTransformerBlock` at the config
    CosyVoice actually instantiates it with: no cross-attention
    (`cross_attention_dim=None`, `double_self_attention=False`), no AdaLN
    (`norm_type` defaults to `"layer_norm"`, so `timestep` is accepted but never
    used), `activation_fn="gelu"` (plain GELU FFN, not GEGLU). Attention is
    `diffusers.models.attention_processor.Attention`'s standard scaled dot-product
    form: `inner_dim = num_heads * attention_head_dim` (512), which need not equal
    `dim` (256) -- an expand-then-project-back attention, not equal-width. Q/K/V
    have no bias (`attention_bias=False`, diffusers' default); the output
    projection does.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads, self.head_dim = num_heads, head_dim
        self.norm1 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, dim * 4)
        self.ff_out = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        """x: [B, T, dim]. attn_bias: additive, broadcastable to [B, 1, T, T]."""
        b, t, _ = x.shape
        h = self.norm1(x)

        def _heads(m):
            return m.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = _heads(self.to_q(h)), _heads(self.to_k(h)), _heads(self.to_v(h))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        attn = attn.transpose(1, 2).reshape(b, t, self.num_heads * self.head_dim)
        x = x + self.to_out(attn)

        h = self.norm3(x)
        h = self.ff_out(F.gelu(self.ff_in(h)))
        return x + h


class CausalConditionalDecoderRef(nn.Module):
    """`cosyvoice.flow.decoder.CausalConditionalDecoder` at the real checkpoint's
    1-stage config (see module docstring), both `streaming` modes.
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        out_channels: int = OUT_CHANNELS,
        channels: int = CHANNELS,
        n_blocks: int = N_BLOCKS,
        num_mid_blocks: int = NUM_MID_BLOCKS,
        num_heads: int = NUM_HEADS,
        head_dim: int = ATTENTION_HEAD_DIM,
    ):
        super().__init__()
        self.channels = channels
        time_embed_dim = channels * 4
        self.time_embeddings_dim = in_channels
        self.time_mlp = TimestepEmbeddingRef(in_channels, time_embed_dim)

        self.down_resnet = CausalResnetBlock1DRef(in_channels, channels, time_embed_dim)
        self.down_tbs = nn.ModuleList(
            [BasicTransformerBlockRef(channels, num_heads, head_dim) for _ in range(n_blocks)]
        )
        self.down_conv = CausalConv1dRef(channels, channels, 3)

        self.mid_resnets = nn.ModuleList(
            [CausalResnetBlock1DRef(channels, channels, time_embed_dim) for _ in range(num_mid_blocks)]
        )
        self.mid_tbs = nn.ModuleList(
            [
                nn.ModuleList([BasicTransformerBlockRef(channels, num_heads, head_dim) for _ in range(n_blocks)])
                for _ in range(num_mid_blocks)
            ]
        )

        self.up_resnet = CausalResnetBlock1DRef(channels * 2, channels, time_embed_dim)
        self.up_tbs = nn.ModuleList([BasicTransformerBlockRef(channels, num_heads, head_dim) for _ in range(n_blocks)])
        self.up_conv = CausalConv1dRef(channels, channels, 3)

        self.final_block = CausalBlock1DRef(channels, channels)
        self.final_proj = nn.Conv1d(channels, out_channels, 1)
        self.initialize_weights()

    def initialize_weights(self):
        """`cosyvoice.flow.decoder.ConditionalDecoder.initialize_weights`, verbatim."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @classmethod
    def from_checkpoint(cls, estimator_state_dict: dict, **kwargs) -> "CausalConditionalDecoderRef":
        """Real weights from `flow.pt`'s `decoder.estimator.*` keys (strip that
        prefix first -- see `tt/checkpoint.py`'s `sub_state_dict`).

        Real upstream wraps this class's `down_resnet`/`down_tbs`/`down_conv`
        in one `nn.ModuleList` (`down_blocks[0] = [resnet, ModuleList(tbs),
        conv]`, confirmed directly against the real checkpoint -- `channels:
        [256]` in `cosyvoice2.yaml` is a 1-element list, so `down_blocks`/
        `up_blocks` each have exactly one index, `0`); `mid_resnets`/`mid_tbs`
        similarly under `mid_blocks[i] = [resnet_i, ModuleList(tbs_i)]` for
        each of the 12 mid stages. This is a real, verified container/naming
        difference, NOT a missing architecture -- every real tensor shape at
        every real index matches this class's own modules exactly once
        unwrapped (confirmed empirically: 910/910 keys match after remapping,
        zero missing, zero shape mismatches).

        Two more real (non-obvious) unwrappings, also confirmed directly
        against the real checkpoint's tensor shapes, not assumed from names:
        - `CausalBlock1DRef` (`block1`/`block2`/`final_block`): real upstream
          wraps `conv`/`norm` in `self.block = nn.Sequential(conv, Mish(),
          norm)` -- `block.0.*` -> `conv.*`, `block.2.*` -> `norm.*` (index 1
          is `Mish`, parameter-free, nothing to load).
        - `BasicTransformerBlockRef`: real upstream nests `to_q`/`to_k`/`to_v`/
          `to_out` under `self.attn1` (an `Attention` submodule) and
          `ff_in`/`ff_out` under `self.ff.net` (a `FeedForward` submodule,
          `net.0.proj`/`net.2`) -- this class keeps all four flat, matching
          `dim*4 == 1024` shape-for-shape with the real `ff.net.0.proj`/
          `ff.net.2` tensors, confirming it's the same computation, just
          unwrapped.
        """
        import re

        remapped = {}
        for k, v in estimator_state_dict.items():
            nk = re.sub(r"^down_blocks\.0\.0\.", "down_resnet.", k)
            nk = re.sub(r"^down_blocks\.0\.1\.(\d+)\.", r"down_tbs.\1.", nk)
            nk = re.sub(r"^down_blocks\.0\.2\.", "down_conv.", nk)
            nk = re.sub(r"^mid_blocks\.(\d+)\.0\.", r"mid_resnets.\1.", nk)
            nk = re.sub(r"^mid_blocks\.(\d+)\.1\.(\d+)\.", r"mid_tbs.\1.\2.", nk)
            nk = re.sub(r"^up_blocks\.0\.0\.", "up_resnet.", nk)
            nk = re.sub(r"^up_blocks\.0\.1\.(\d+)\.", r"up_tbs.\1.", nk)
            nk = re.sub(r"^up_blocks\.0\.2\.", "up_conv.", nk)
            nk = nk.replace(".block.0.", ".conv.")
            nk = nk.replace(".block.2.", ".norm.")
            nk = nk.replace(".attn1.to_out.0.", ".to_out.")
            nk = nk.replace(".attn1.", ".")
            nk = nk.replace(".ff.net.0.proj.", ".ff_in.")
            nk = nk.replace(".ff.net.2.", ".ff_out.")
            remapped[nk] = v
        ref = cls(**kwargs)
        ref.load_state_dict(remapped, strict=True)
        return ref

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        streaming: bool = False,
    ) -> torch.Tensor:
        """x/mu/cond: [B, T, 80]. mask: [B, T, 1] (1.0 = valid). spks: [B, 80]. t: [B]."""
        temb = sinusoidal_pos_emb_torch(t, self.time_embeddings_dim)
        temb = self.time_mlp(temb)

        h = torch.cat([x, mu], dim=-1)
        h = torch.cat([h, spks.unsqueeze(1).expand(-1, h.shape[1], -1)], dim=-1)
        h = torch.cat([h, cond], dim=-1)

        # channel-first for the resnet/conv legs (matches nn.Conv1d's own layout);
        # channel-last for LayerNorm/attention. Transposed at each boundary, exactly
        # like the real source's own `rearrange` calls.
        mask_cl = mask  # [B, T, 1]
        if streaming:
            # Real upstream, literally: `add_optional_chunk_mask` ANDs the per-key padding mask
            # with `subsequent_chunk_mask(T, static_chunk_size)`, then `mask_to_bias`. Built in
            # the boolean domain here (not as the additive sum the device uses) so the reference
            # does not share the device path's formulation. No row is ever all-False while at
            # least key 0 is valid, so upstream's "force all-False rows to True" branch is moot.
            t_len = mask.shape[1]
            valid = mask_cl.transpose(1, 2).bool() & subsequent_chunk_mask_torch(t_len, CHUNK_SIZE_UP).unsqueeze(0)
            attn_bias = ((1.0 - valid.float()) * -1.0e10).unsqueeze(1)  # [B, 1, T, T]
        else:
            attn_bias = (1.0 - mask_cl.transpose(1, 2)).float() * -1.0e10  # [B, 1, T] -> broadcasts to [B,1,Tq,Tk]
            attn_bias = attn_bias.unsqueeze(1)

        h = h.transpose(1, 2)  # [B, in_channels, T]
        mask_cf = mask.transpose(1, 2)  # [B, 1, T]
        h = self.down_resnet(h, mask_cf, temb)
        h = h.transpose(1, 2)
        for tb in self.down_tbs:
            h = tb(h, attn_bias)
        skip = h
        h = h.transpose(1, 2)
        h = self.down_conv(h * mask_cf)

        for resnet, tbs in zip(self.mid_resnets, self.mid_tbs):
            h = resnet(h, mask_cf, temb)
            h = h.transpose(1, 2)
            for tb in tbs:
                h = tb(h, attn_bias)
            h = h.transpose(1, 2)

        h = h.transpose(1, 2)  # [B, T, channels]
        h = torch.cat([h, skip], dim=-1)
        h = h.transpose(1, 2)
        h = self.up_resnet(h, mask_cf, temb)
        h = h.transpose(1, 2)
        for tb in self.up_tbs:
            h = tb(h, attn_bias)
        h = h.transpose(1, 2)
        h = self.up_conv(h * mask_cf)

        h = self.final_block(h, mask_cf)
        out = self.final_proj(h * mask_cf)
        out = out.transpose(1, 2)  # [B, T, 80]
        return out * mask


class CausalConditionalCFMRef:
    """`cosyvoice.flow.flow_matching.CausalConditionalCFM`: the fixed-noise-buffer
    Euler solver wrapped around `CausalConditionalDecoderRef`. `streaming` is only passed
    through to the estimator, exactly as upstream's `solve_euler` does.
    """

    def __init__(
        self,
        estimator: CausalConditionalDecoderRef,
        sigma_min: float = 1e-6,
        t_scheduler: str = "cosine",
        inference_cfg_rate: float = 0.7,
        seed: int = 0,
    ):
        self.estimator = estimator
        self.sigma_min = sigma_min
        self.t_scheduler = t_scheduler
        self.inference_cfg_rate = inference_cfg_rate
        # `set_all_random_seed(0)` then `torch.randn([1, 80, 50*300])`, verbatim --
        # a fixed buffer, not fresh-per-call noise. Stored channel-last here
        # ([1, 15000, 80]) to match this module's convention.
        torch.manual_seed(seed)
        self.rand_noise = torch.randn(1, 80, 50 * 300).transpose(1, 2).contiguous()

    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        n_timesteps: int,
        spks: torch.Tensor,
        cond: torch.Tensor,
        temperature: float = 1.0,
        streaming: bool = False,
    ) -> torch.Tensor:
        t_len = mu.shape[1]
        z = self.rand_noise[:, :t_len, :].to(mu.dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span, mu, mask, spks, cond, streaming=streaming)

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        streaming: bool = False,
    ) -> torch.Tensor:
        b, t_len, c = x.shape
        t = t_span[0].unsqueeze(0)
        dt = t_span[1] - t_span[0]

        x_in = torch.zeros(2 * b, t_len, c, dtype=x.dtype)
        mask_in = torch.zeros(2 * b, t_len, 1, dtype=x.dtype)
        mu_in = torch.zeros(2 * b, t_len, c, dtype=x.dtype)
        t_in = torch.zeros(2 * b, dtype=x.dtype)
        spks_in = torch.zeros(2 * b, spks.shape[-1], dtype=x.dtype)
        cond_in = torch.zeros(2 * b, t_len, c, dtype=x.dtype)

        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[:b] = mu
            t_in[:] = t
            spks_in[:b] = spks
            cond_in[:b] = cond
            dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming=streaming)
            dphi_dt, cfg_dphi_dt = dphi_dt[:b], dphi_dt[b:]
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x


# ---------------------------------------------------------------------------
# TTNN port. Channels-last ([N, L, C]) throughout, per this package's convention --
# unlike the torch reference above, which mirrors upstream's own [B, C, T] storage
# internally (with transposes at the attention/LayerNorm boundaries), the TT port
# never needs those transposes: `ttnn.conv1d` and `ttnn.layer_norm` both already
# operate channels-last natively.
# ---------------------------------------------------------------------------


def _linear_weight(device, weight: torch.Tensor, dtype):
    """torch Linear weight [out, in] -> ttnn [in, out], matching ttnn.linear's
    expected orientation (see TtLinearHead in tt/llm/qwen2lm.py, the same
    convention used there)."""
    return ttnn.from_torch(
        weight.detach().float().t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )


def _bias(device, bias: torch.Tensor, dtype):
    return ttnn.from_torch(bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class TtCausalConv1d(TtConv1d):
    """`CausalConv1dRef` on device: `ttnn.conv1d` with asymmetric `padding=(k-1, 0)`
    -- confirmed from `ttnn.conv1d`'s own docstring that `padding` accepts a
    `[pad_left, pad_right]` tuple directly, so this needs no manual pad step (which
    would hit `ttnn.pad`'s documented "front padding on device not supported in
    tile layout" restriction).

    A thin subclass of `hifigan.conv.TtConv1d` (ported 2026-09-22; used to be a
    self-contained, unprepared-weight class with its own `max|out|`-within-2%
    verification). `TtConv1d` already generalizes to asymmetric `padding` tuples --
    see its `_pad_pair`/`_prepared`/`_host_reference` -- so this class only fixes
    the padding this module always uses and restores the single-tensor return
    every call site here expects (`TtConv1d.__call__` returns `(out, out_length)`).
    Inherits prepared, per-geometry-cached weights (a trace no longer rejects this
    conv) and the relative-L2, float64-arbitrated resolver (`_verify_and_resolve`)
    in place of the old absolute `max|out|`-within-2% check, which could not tell a
    corruption that scales the whole output from one that doesn't.
    """

    def __init__(
        self, device, weight: torch.Tensor, bias: torch.Tensor, dtype=ttnn.bfloat16, weights_dtype=ttnn.bfloat16
    ):
        kernel_size = weight.shape[-1]
        super().__init__(
            device,
            weight,
            bias,
            stride=1,
            padding=(kernel_size - 1, 0),
            dilation=1,
            groups=1,
            dtype=dtype,
            weights_dtype=weights_dtype,
            high_fidelity=True,
        )

    @classmethod
    def from_module(cls, device, module: CausalConv1dRef, dtype=ttnn.bfloat16):
        return cls(device, module.weight, module.bias, dtype=dtype)

    def __call__(self, x, input_length: int, batch_size: int = 1):
        out, _ = super().__call__(x, input_length, batch_size)
        return out


class TtCausalBlock1D:
    def __init__(self, device, module: CausalBlock1DRef, dtype=ttnn.bfloat16):
        self.conv = TtCausalConv1d.from_module(device, module.conv, dtype=dtype)
        self.norm_weight = ttnn.from_torch(
            module.norm.weight.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.norm_bias = ttnn.from_torch(
            module.norm.bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, x, mask, length: int, batch_size: int = 1):
        h = ttnn.multiply(x, mask)
        h = self.conv(h, length, batch_size)
        h = ttnn.layer_norm(h, weight=self.norm_weight, bias=self.norm_bias, epsilon=1e-5)
        h = ttnn.mish(h)
        return ttnn.multiply(h, mask)


class TtCausalResnetBlock1D:
    def __init__(self, device, module: CausalResnetBlock1DRef, dtype=ttnn.bfloat16):
        self.block1 = TtCausalBlock1D(device, module.block1, dtype=dtype)
        self.block2 = TtCausalBlock1D(device, module.block2, dtype=dtype)
        self.cc = flow_matmul_config(device)
        mlp_linear = module.mlp[1]  # nn.Sequential(Mish, Linear)
        self.mlp_weight = _linear_weight(device, mlp_linear.weight, dtype)
        self.mlp_bias = _bias(device, mlp_linear.bias, dtype)
        self.res_weight = _linear_weight(device, module.res_conv.weight.squeeze(-1), dtype)
        self.res_bias = _bias(device, module.res_conv.bias, dtype)

    def __call__(self, x, mask, time_emb, length: int, batch_size: int = 1):
        """time_emb: `[2B, 1, time_embed_dim]` -- already shaped to broadcast
        against `h`'s `[2B, T, dim_out]` with no reshape needed."""
        h = self.block1(x, mask, length, batch_size)
        temb = ttnn.mish(time_emb)
        temb = ttnn.linear(temb, self.mlp_weight, bias=self.mlp_bias, compute_kernel_config=self.cc)
        h = ttnn.add(h, temb)
        h = self.block2(h, mask, length, batch_size)
        skip = ttnn.multiply(x, mask)
        skip = ttnn.linear(skip, self.res_weight, bias=self.res_bias, compute_kernel_config=self.cc)
        return ttnn.add(h, skip)


def flow_matmul_config(device):
    """Compute config for every linear / matmul / SDPA in the flow estimator.

    `COSYVOICE2_FLOW_MATMUL_CC=accurate` -> HiFi4 + fp32 destination accumulation + packer L1 accumulation (the config
    CosyVoice1's estimator uses, and this port's convs already use). Unset -> ttnn's default. The estimator stacks 56
    transformer blocks x 10 Euler steps, so bf16 accumulation drift compounds, and the later Euler steps are where
    it shows. Read at construction time."""
    if os.environ.get("COSYVOICE2_FLOW_MATMUL_CC", "") == "accurate":
        return accurate_compute_config(device)
    return None


def flow_fused_qkv() -> bool:
    """One `[dim, 3*inner]` Q/K/V linear plus `split_query_key_value_and_split_heads`, instead of three
    linears, three reshape+transpose head splits and a separate K transpose. Measured on device at
    `[2, 660, 256]`: ~0.22 ms for the fused linear vs ~2.06 ms for the three linears + head splits (fp32).
    Bit-identical to the three-linear form (mel max |diff| 0.0 on two sentences). On by default;
    `COSYVOICE2_FLOW_FUSED_QKV=0` turns it off. Read at construction time."""
    return os.environ.get("COSYVOICE2_FLOW_FUSED_QKV", "1") == "1"


def flow_cfm_trace() -> bool:
    """Traced Euler-step solve in `TtCausalConditionalCFM.forward` -- one estimator call
    (CFG concat, estimator, CFG blend, update) captured as a single trace, replayed once per
    Euler step from the host (see `TtCausalConditionalCFM._capture`). Off by default, matching
    `TtQwen2LM`'s `use_decode_trace` -- opt-in, since a trace needs the device opened with a
    nonzero `trace_region_size` and holds a persistent buffer per distinct mel length seen.
    `COSYVOICE2_FLOW_CFM_TRACE=1` turns it on; read at construction time."""
    return os.environ.get("COSYVOICE2_FLOW_CFM_TRACE", "0") == "1"


def flow_fused_sdpa() -> bool:
    """`ttnn.transformer.scaled_dot_product_attention` in place of the explicit
    matmul -> scale -> add-bias -> softmax -> matmul chain. Measured on device at `[2, 8, 660, 64]` (bf16):
    0.33 ms vs 0.90 ms, and slightly more accurate. bf16-only (fp32 inputs fail a TT_FATAL), so q/k/v are
    cast to bf16 for the call if they are not already, and the result cast back.

    Masking: `streaming=True` passes the full `[B, 1, T, T]` padding + chunk-causal bias as SDPA's
    `attn_mask` (correct, and faster than the explicit chain -- see `TtBasicTransformerBlock`).
    `streaming=False` passes `attn_mask=None`, i.e. assumes an all-ones padding mask, which holds for
    the whole non-streaming inference path; `TtCausalConditionalCFM.forward` refuses a partial mask in
    that one combination (non-streaming + fused SDPA) rather than silently ignoring it. On by default;
    `COSYVOICE2_FLOW_SDPA=0` turns it off. Read at construction time."""
    return os.environ.get("COSYVOICE2_FLOW_SDPA", "1") == "1"


# SDPA chunking. The op's DEFAULT program config gets slower with sequence length (measured on device,
# `[2, 8, T, 64]` bf16: 0.34 ms at T=660 and 1.46 ms at T=1500), while q_chunk=128 / k_chunk=256 measured 0.15 ms
# and 0.35 ms -- 2.3x / 4.2x -- with the same accuracy at every T in 300..1500 (error vs float64 ~2.7e-2 either way,
# which is bf16 input quantisation, not the op).
SDPA_Q_CHUNK, SDPA_K_CHUNK = 128, 256


class TtBasicTransformerBlock:
    def __init__(self, device, module: BasicTransformerBlockRef, dtype=ttnn.bfloat16):
        self.num_heads, self.head_dim = module.num_heads, module.head_dim
        self.fused_qkv = flow_fused_qkv()
        self.fused_sdpa = flow_fused_sdpa()
        self.cc = flow_matmul_config(device)
        self.sdpa_program_config = (
            ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                q_chunk_size=SDPA_Q_CHUNK,
                k_chunk_size=SDPA_K_CHUNK,
                exp_approx_mode=False,
            )
            if self.fused_sdpa
            else None
        )
        self.norm1_w = ttnn.from_torch(
            module.norm1.weight.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.norm1_b = ttnn.from_torch(
            module.norm1.bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.norm3_w = ttnn.from_torch(
            module.norm3.weight.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.norm3_b = ttnn.from_torch(
            module.norm3.bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        if self.fused_qkv:
            # [q | k | v] along the output dim, the order split_query_key_value_and_split_heads expects.
            self.wqkv = _linear_weight(
                device, torch.cat([module.to_q.weight, module.to_k.weight, module.to_v.weight], dim=0), dtype
            )
        else:
            self.wq = _linear_weight(device, module.to_q.weight, dtype)
            self.wk = _linear_weight(device, module.to_k.weight, dtype)
            self.wv = _linear_weight(device, module.to_v.weight, dtype)
        self.wo = _linear_weight(device, module.to_out.weight, dtype)
        self.bo = _bias(device, module.to_out.bias, dtype)
        self.w_ff_in = _linear_weight(device, module.ff_in.weight, dtype)
        self.b_ff_in = _bias(device, module.ff_in.bias, dtype)
        self.w_ff_out = _linear_weight(device, module.ff_out.weight, dtype)
        self.b_ff_out = _bias(device, module.ff_out.bias, dtype)
        self.scale = module.head_dim**-0.5

    def _heads(self, x, b, t):
        x = ttnn.reshape(x, (b, t, self.num_heads, self.head_dim))
        return ttnn.transpose(x, 1, 2)  # [B, heads, T, head_dim]

    def _sdpa(self, q, k, v, attn_mask=None):
        """Fused attention. bf16-only op: cast in (and the result back) only if the activations are not bf16.

        `attn_mask`: `None` (non-streaming: all keys valid) or the full `[B, 1, T, T]` bf16 DRAM bias the
        decoder builds in streaming mode -- the op validates exactly that form (a key-only `[B, 1, 1, T]`
        padding row is rejected). Measured at the real `[2, 8, T, 64]` shape, real bucket sizes
        (scripts/perf_2026_09_25/masked_sdpa_investigation.py): PCC 0.9997 vs float64 at chunk-aligned and
        non-aligned valid lengths alike, and 2.2x / 4.9x / 7.1x faster than the explicit chain with the same
        mask at T=384 / 768 / 1536. A mask costs ~2.1-3.5x over no mask at the same T, which is why the
        non-streaming path keeps passing `None` rather than an all-zero mask."""
        dt = q.dtype
        if dt != ttnn.bfloat16:
            q, k, v = (ttnn.typecast(x, ttnn.bfloat16) for x in (q, k, v))
        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=attn_mask,
            scale=self.scale,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.cc,
        )
        return out if dt == ttnn.bfloat16 else ttnn.typecast(out, dt)

    def __call__(self, x, attn_bias, streaming: bool = False):
        """`attn_bias`: non-streaming, the `[B, 1, 1, T]` padding row (used by the explicit chain only;
        fused SDPA gets no mask). Streaming, the full `[B, 1, T, T]` padding + chunk-causal bias, used by
        whichever attention path is active."""
        b, t, _ = x.shape
        h = ttnn.layer_norm(x, weight=self.norm1_w, bias=self.norm1_b, epsilon=1e-5)
        if self.fused_qkv:
            # The explicit chain wants K pre-transposed ([B, heads, head_dim, T]); SDPA wants it as [B, heads, T, head_dim].
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                ttnn.linear(h, self.wqkv, compute_kernel_config=self.cc),
                num_heads=self.num_heads,
                transpose_key=not self.fused_sdpa,
            )
        else:
            q = self._heads(ttnn.linear(h, self.wq, compute_kernel_config=self.cc), b, t)
            k = self._heads(ttnn.linear(h, self.wk, compute_kernel_config=self.cc), b, t)
            v = self._heads(ttnn.linear(h, self.wv, compute_kernel_config=self.cc), b, t)
        if self.fused_sdpa:
            out = self._sdpa(q, k, v, attn_mask=attn_bias if streaming else None)  # [B, heads, T, head_dim]
        else:
            k_t = k if self.fused_qkv else ttnn.transpose(k, -2, -1)
            scores = ttnn.matmul(q, k_t, compute_kernel_config=self.cc)
            scores = ttnn.multiply(scores, self.scale)
            scores = ttnn.add(scores, attn_bias)
            attn = ttnn.softmax(scores, dim=-1)
            out = ttnn.matmul(attn, v, compute_kernel_config=self.cc)  # [B, heads, T, head_dim]
        out = ttnn.transpose(out, 1, 2)
        out = ttnn.reshape(out, (b, t, self.num_heads * self.head_dim))
        out = ttnn.linear(out, self.wo, bias=self.bo, compute_kernel_config=self.cc)
        x = ttnn.add(x, out)

        h = ttnn.layer_norm(x, weight=self.norm3_w, bias=self.norm3_b, epsilon=1e-5)
        h = ttnn.linear(h, self.w_ff_in, bias=self.b_ff_in, compute_kernel_config=self.cc)
        h = ttnn.gelu(h)
        h = ttnn.linear(h, self.w_ff_out, bias=self.b_ff_out, compute_kernel_config=self.cc)
        return ttnn.add(x, h)


class TtCausalConditionalDecoder:
    """`CausalConditionalDecoderRef` on device -- see that class and the module
    docstring for the verified 1-stage topology this hardcodes."""

    def __init__(self, device, module: CausalConditionalDecoderRef, dtype=ttnn.bfloat16):
        self.device = device
        self.cc = flow_matmul_config(device)
        self.time_embeddings_dim = module.time_embeddings_dim
        self.channels = module.channels
        self.time_mlp_w1 = _linear_weight(device, module.time_mlp.linear_1.weight, dtype)
        self.time_mlp_b1 = _bias(device, module.time_mlp.linear_1.bias, dtype)
        self.time_mlp_w2 = _linear_weight(device, module.time_mlp.linear_2.weight, dtype)
        self.time_mlp_b2 = _bias(device, module.time_mlp.linear_2.bias, dtype)

        self.down_resnet = TtCausalResnetBlock1D(device, module.down_resnet, dtype=dtype)
        self.down_tbs = [TtBasicTransformerBlock(device, tb, dtype=dtype) for tb in module.down_tbs]
        self.down_conv = TtCausalConv1d.from_module(device, module.down_conv, dtype=dtype)

        self.mid_resnets = [TtCausalResnetBlock1D(device, m, dtype=dtype) for m in module.mid_resnets]
        self.mid_tbs = [[TtBasicTransformerBlock(device, tb, dtype=dtype) for tb in tbs] for tbs in module.mid_tbs]

        self.up_resnet = TtCausalResnetBlock1D(device, module.up_resnet, dtype=dtype)
        self.up_tbs = [TtBasicTransformerBlock(device, tb, dtype=dtype) for tb in module.up_tbs]
        self.up_conv = TtCausalConv1d.from_module(device, module.up_conv, dtype=dtype)

        self.final_block = TtCausalBlock1D(device, module.final_block, dtype=dtype)
        self.final_weight = _linear_weight(device, module.final_proj.weight.squeeze(-1), dtype)
        self.final_bias = _bias(device, module.final_proj.bias, dtype)

    def _sinusoidal_pos_emb(self, t: torch.Tensor) -> ttnn.Tensor:
        """Computed on host (cheap: t is a length-2*B vector, not a mel-length
        tensor) and uploaded directly at `[2B, 1, dim]` -- the shape every
        downstream use (broadcast-add against `[2B, T, dim_out]`) actually needs,
        so no on-device reshape ever has to touch this tensor's tile-critical last
        two dims (a `[N, 1, dim]` reshape of an already-tiled `[N, dim]` tensor hit
        `TT_FATAL: Invalid arguments to reshape` -- inserting a non-tile-aligned
        size-1 dim into an already-tiled tensor is not a valid in-place reshape;
        building the right shape on the host side, where it is free, avoids it
        entirely).
        """
        emb = sinusoidal_pos_emb_torch(t, self.time_embeddings_dim)
        return ttnn.from_torch(
            emb.reshape(emb.shape[0], 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    def chunk_bias_device(self, t_len: int):
        """`chunk_causal_bias_torch(t_len, CHUNK_SIZE_UP)` uploaded as `[1, 1, T, T]` bf16 in DRAM -- the
        streaming mode's chunk-causal term. A pure function of `t_len` (NOT of the valid length: the
        padding term is separate, see `_forward_from_raw_temb`), so callers can upload it once per
        geometry."""
        return ttnn.from_torch(
            chunk_causal_bias_torch(t_len, CHUNK_SIZE_UP),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x, mask, mu, t: torch.Tensor, spks, cond, length: int, batch_size: int, streaming: bool = False):
        temb_raw = self._sinusoidal_pos_emb(t)  # [2B, 1, time_embeddings_dim]
        chunk_bias = self.chunk_bias_device(length) if streaming else None
        out = self._forward_from_raw_temb(x, mask, mu, temb_raw, spks, cond, length, batch_size, chunk_bias=chunk_bias)
        if chunk_bias is not None:
            ttnn.deallocate(chunk_bias)
        return out

    def _forward_from_raw_temb(self, x, mask, mu, temb_raw, spks, cond, length: int, batch_size: int, chunk_bias=None):
        """Same computation as `__call__`, from an already-uploaded raw sinusoidal
        embedding (pre time_mlp) instead of a host `t`. Split out so the traced Euler-step
        solver (`TtCausalConditionalCFM`) can feed a persistent device buffer here directly:
        `_sinusoidal_pos_emb` computes on the host (see its docstring), which a trace
        capture cannot contain, so every schedule step's raw embedding is precomputed once
        before capture instead (see `TtCausalConditionalCFM._capture`) and swapped into that
        buffer via a device-to-device `ttnn.copy` per replay -- no host math in the loop.

        `chunk_bias` (`[1, 1, T, T]`, from `chunk_bias_device`) selects streaming mode. The padding
        term is ALWAYS rebuilt here from `mask`, on device, every call, and the two are summed per
        call -- so in the traced solver the padding term follows whatever `mask` currently holds
        (refreshed on trace reuse), while only the `t_len`-determined chunk term is a fixed buffer.
        """
        temb = ttnn.linear(temb_raw, self.time_mlp_w1, bias=self.time_mlp_b1, compute_kernel_config=self.cc)
        temb = ttnn.silu(temb)
        temb = ttnn.linear(
            temb, self.time_mlp_w2, bias=self.time_mlp_b2, compute_kernel_config=self.cc
        )  # [2B, 1, time_embed_dim]

        h = ttnn.concat([x, mu], dim=-1)
        # spks arrives pre-shaped [2B, 1, spk_dim] (see TtCausalConditionalCFM --
        # uploaded that way on the host to avoid the same tile-reshape pitfall
        # `_sinusoidal_pos_emb` documents).
        h = ttnn.concat([h, ttnn.repeat(spks, ttnn.Shape((1, h.shape[1], 1)))], dim=-1)
        h = ttnn.concat([h, cond], dim=-1)

        attn_bias = ttnn.multiply(ttnn.subtract(mask, 1.0), 1.0e10)  # (mask-1)*1e10 == (1-mask)*-1e10
        attn_bias = ttnn.transpose(attn_bias, 1, 2)  # [B, 1, T]
        attn_bias = ttnn.reshape(attn_bias, (attn_bias.shape[0], 1, 1, attn_bias.shape[-1]))  # [B,1,1,T]
        streaming = chunk_bias is not None
        if streaming:
            # Upstream's `padding_mask & chunk_mask`, as its additive form: [1,1,T,T] + [B,1,1,T] ->
            # [B,1,T,T] (device broadcast verified equal to the host sum, masked_sdpa_investigation.py).
            # bf16 + DRAM: the only form fused SDPA's `attn_mask` accepts.
            pad_bias = attn_bias if attn_bias.dtype == ttnn.bfloat16 else ttnn.typecast(attn_bias, ttnn.bfloat16)
            attn_bias = ttnn.add(chunk_bias, pad_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(pad_bias)

        h = self.down_resnet(h, mask, temb, length, batch_size)
        for tb in self.down_tbs:
            h = tb(h, attn_bias, streaming)
        skip = h
        h = self.down_conv(ttnn.multiply(h, mask), length, batch_size)

        for resnet, tbs in zip(self.mid_resnets, self.mid_tbs):
            h = resnet(h, mask, temb, length, batch_size)
            for tb in tbs:
                h = tb(h, attn_bias, streaming)

        h = ttnn.concat([h, skip], dim=-1)
        h = self.up_resnet(h, mask, temb, length, batch_size)
        for tb in self.up_tbs:
            h = tb(h, attn_bias, streaming)
        h = self.up_conv(ttnn.multiply(h, mask), length, batch_size)

        h = self.final_block(h, mask, length, batch_size)
        out = ttnn.linear(
            ttnn.multiply(h, mask), self.final_weight, bias=self.final_bias, compute_kernel_config=self.cc
        )
        return ttnn.multiply(out, mask)


def cfm_trace_cache_capacity() -> int:
    """How many captured CFM traces (distinct `(t_len, channels, streaming)` keys) `TtCausalConditionalCFM`
    keeps resident at once. `COSYVOICE2_CFM_TRACE_CACHE_CAPACITY`, default 1 -- the original single-slot
    behavior (a new key releases the old trace before capturing). Only 1 is accepted: see
    `TtCausalConditionalCFM`'s docstring for why capacity > 1 is refused. Read at construction time."""
    return int(os.environ.get("COSYVOICE2_CFM_TRACE_CACHE_CAPACITY", "1"))


@dataclass
class _CfmTraceSlot:
    """One captured Euler-step trace and the persistent device buffers its graph reads through. The
    trace bakes in these buffers' ADDRESSES, so they are refilled in place (`ttnn.copy`), never
    reassigned, for as long as the trace lives."""

    trace_id: object
    next_x: object  # allocated INSIDE the capture -- owned by the trace, see `release`
    x_buf: object
    temb_raw_buf: object
    dt_buf: object
    mu2_buf: object
    spks2_buf: object
    cond2_buf: object
    mask2_buf: object
    chunk_bias_buf: object  # streaming only: the fixed chunk-causal term for this t_len, else None

    def release(self, device) -> None:
        if self.trace_id is not None:
            ttnn.release_trace(device, self.trace_id)
            self.trace_id = None
        # `next_x` is allocated inside the capture, so `release_trace` reclaims it -- deallocating it
        # here would double-free.
        self.next_x = None
        for name in (
            "x_buf",
            "temb_raw_buf",
            "dt_buf",
            "mu2_buf",
            "spks2_buf",
            "cond2_buf",
            "mask2_buf",
            "chunk_bias_buf",
        ):
            t = getattr(self, name)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self, name, None)


class TtCausalConditionalCFM:
    """`CausalConditionalCFMRef` on device -- same fixed-noise-buffer, same Euler
    solver, same CFG-doubling batch-of-2 trick, built directly against the ttnn
    estimator above rather than round-tripping through torch for the solve.

    **Traced solve (`COSYVOICE2_FLOW_CFM_TRACE=1` / `use_trace=True`), added 2026-09-22,
    ported from the CosyVoice1 reference repo's `tt/flow/cfm.py` recipe.** The eager path
    below (`use_trace=False`, still the default) round-trips through the host on every
    Euler step: builds `x_in` via `torch.cat` on the CPU, re-uploads it, reads `dphi` back
    with `ttnn.to_torch`, and does the CFG blend + Euler update in torch. Ten steps means
    ten such round trips, each paying full host<->device dispatch latency for work that is
    otherwise entirely on-device.

    The traced path captures **exactly one Euler step** -- CFG-doubling concat, one
    estimator call, CFG blend, the `x + dt*dphi_dt` update -- as a single ttnn trace, and
    replays that SAME captured graph once per Euler step from the host. The step count is
    never baked into the capture: nothing about `begin_trace_capture`/`end_trace_capture`
    or the body it records depends on how many times the caller later calls
    `execute_trace`, and `_forward_traced`'s replay loop is the only place step count
    appears, entirely on the host, entirely between replays -- see `_capture`'s docstring.

    Two conditions this depends on, ported directly from the CosyVoice1 recipe:

    * **Nothing may be allocated during a replay.** `x_buf` (the ODE state) is refreshed
      in place from the trace's own output (`ttnn.copy(slot.next_x, slot.x_buf)`,
      device-to-device, no host round trip) rather than reassigned -- reassigning it would
      leave the next replay reading stale data with no error, since the trace has already
      baked in `x_buf`'s address.
    * **`t` and `dt` are device tensors refreshed per step, never Python floats baked into
      the graph.** Our port's `_sinusoidal_pos_emb` computes the raw sinusoidal embedding
      on the HOST (see `TtCausalConditionalDecoder._sinusoidal_pos_emb`'s docstring) --
      unlike CosyVoice1's `time_embedding`, which does the sin/cos on-device and so can
      read a raw scalar `t` straight from a trace-captured buffer. Because the 10-point
      cosine schedule is fixed and utterance-independent, every step's `(t, dt)` is
      computed on the host once, before the replay loop starts, and each step's raw
      embedding / `dt` is swapped into the persistent `temb_raw_buf`/`dt_buf` via a
      device-to-device `ttnn.copy` just before its replay. No host math happens inside the
      graph; it only ever reads device tensors.

    **Streaming (`streaming=True`).** The trace key is `(t_len, channels, streaming)`: a
    streaming trace's captured body adds the chunk-causal term and passes a mask to fused
    SDPA, so it must never be reused for a non-streaming solve at the same shape (or vice
    versa). What each part of the streaming attention bias is, and when it is refreshed:

    * chunk-causal term -- `slot.chunk_bias_buf`, `[1, 1, T, T]`, uploaded ONCE at capture.
      A pure function of `t_len` (upstream's `subsequent_chunk_mask` depends only on the
      query index and the chunk size), and `t_len` is part of the key, so it is identical
      for every solve that can ever reuse this slot.
    * padding term -- NOT a buffer. Rebuilt inside the traced body, every replay, from
      `slot.mask2_buf`, which `_reuse_trace` refills from the caller's mask on every solve.
      So a reused trace at the same bucket but a DIFFERENT valid length (e.g. valid 310
      after valid 300, both in the 384 bucket) masks the new padding correctly.
    * the sum (`[2, 1, T, T]`, what SDPA's `attn_mask` receives) -- computed inside the
      traced body every replay, never stored.

    **Trace cache: exactly one resident trace (`COSYVOICE2_CFM_TRACE_CACHE_CAPACITY` /
    `trace_cache_capacity`, only 1 accepted; anything else raises).** A new key releases the
    resident trace BEFORE the new capture allocates anything, so no capture ever happens with
    another trace live. `_traces` is keyed and LRU-ordered so a safe multi-trace design could
    slot in later, but a LAZY multi-slot cache was built and refused on 2026-09-25 by
    `TT_METAL_TRACE_ALLOC_TRACKING=1`: capturing key B while trace A is resident allocates B's
    slot buffers -- and, for a bucket never seen before, B's prepared conv weights and
    program-cache entries (warm-up) -- while A is live, and A's next replay may overwrite any
    of them. B's slot buffers are refilled before B's own replays (except the streaming
    `chunk_bias_buf`), but conv weights / program-cache entries never are. Holding several
    traces safely needs every geometry warmed and every slot captured up front, before any
    persistent allocation that must survive a replay -- not built. What it would buy, per
    scripts/perf_2026_09_25/cfm_trace_cache_thrashing_simulation.py (real hop schedule,
    measured costs): every chunk of an utterance lands in a distinct bucket, any capacity
    below an utterance's full bucket range gives zero hits, and the full range would cut
    CFM time by ~19% over back-to-back utterances.

    **A real, live hazard this class does NOT yet protect against**: this trace's safety
    today relies on the CFM trace being captured/replayed/released strictly within one
    `TtCausalMaskedDiffWithXvec.inference` call, which itself runs strictly after
    `TtQwen2LM`'s own decode trace has released (see `qwen2lm.py`'s `_decode_step_traced`
    module note on trace scope -- a trace kept alive across stages hung the device once
    already, observed directly). Streaming (bounty Stages 2/3) will interleave LLM decode,
    flow encode and CFM solve across chunks rather than running them strictly in sequence,
    which breaks that non-overlap assumption for BOTH traces at once. Flagged here in
    writing, per instruction, not solved -- solving it needs either per-stage trace region
    partitioning or a documented ordering constraint the streaming scheduler enforces.
    """

    def __init__(
        self,
        device,
        estimator: TtCausalConditionalDecoder,
        rand_noise: torch.Tensor,
        cfm_ref: CausalConditionalCFMRef,
        use_trace: bool | None = None,
        trace_cache_capacity: int | None = None,
    ):
        self.device = device
        self.estimator = estimator
        self.rand_noise = rand_noise  # host tensor, [1, 15000, 80] -- see CausalConditionalCFMRef
        self.t_scheduler = cfm_ref.t_scheduler
        self.inference_cfg_rate = cfm_ref.inference_cfg_rate
        self.use_trace = flow_cfm_trace() if use_trace is None else use_trace
        # Keep captured traces across solves of the same key -- same reasoning and same env-var
        # name as the CosyVoice1 reference's `COSYVOICE_CFM_TRACE_CACHE`: a solve that captured and
        # released every call would spend a large fraction of the stage recording a graph it
        # immediately threw away.
        self._cache_trace = os.environ.get("COSYVOICE2_CFM_TRACE_CACHE", "1") != "0"
        self._trace_capacity = cfm_trace_cache_capacity() if trace_cache_capacity is None else trace_cache_capacity
        if self._trace_capacity != 1:
            raise ValueError(
                f"CFM trace cache capacity must be 1, got {self._trace_capacity}: a capacity > 1 captures while other "
                "traces are resident, which TT_METAL_TRACE_ALLOC_TRACKING refuses as unsafe (see "
                "TtCausalConditionalCFM's docstring)."
            )
        self._traces: OrderedDict[tuple, _CfmTraceSlot] = OrderedDict()  # LRU order: oldest first

    def forward(
        self,
        mu_t: torch.Tensor,
        mask_t: torch.Tensor,
        n_timesteps: int,
        spks_t: torch.Tensor,
        cond_t: torch.Tensor,
        use_trace: bool | None = None,
        streaming: bool = False,
    ):
        """All *_t args are torch tensors (host), batch=1: mu/cond [1,T,80], mask
        [1,T,1], spks [1,80]. Returns a torch tensor [1,T,80] -- the solve loop's
        conditioning tensors (mu/spks/cond/mask) are uploaded once, CFG-doubled, and
        reused every Euler step; only x and t change per step.

        A partial (padded) mask is refused in exactly one case: non-streaming with fused SDPA on.
        That is the only combination whose attention never sees the mask (`attn_mask=None`, see
        `flow_fused_sdpa`). Streaming always hands the full padding + chunk-causal bias to whichever
        attention path is active, and the non-streaming explicit chain adds the padding row, so both
        honor a partial mask.
        """
        if flow_fused_sdpa() and not streaming and not bool((mask_t != 0).all()):
            raise ValueError(
                "Non-streaming fused flow SDPA (COSYVOICE2_FLOW_SDPA, on by default) runs attention without a mask, "
                "so a padded/partial mask would be silently ignored. Use streaming=True (masked SDPA) or "
                "COSYVOICE2_FLOW_SDPA=0 for padded inputs."
            )
        if use_trace is None:
            use_trace = self.use_trace
        t_len = mu_t.shape[1]
        z = self.rand_noise[:, :t_len, :].to(mu_t.dtype)
        t_span = torch.linspace(0, 1, n_timesteps + 1, dtype=mu_t.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        zero_mu = torch.zeros_like(mu_t)
        zero_spks = torch.zeros_like(spks_t)
        zero_cond = torch.zeros_like(cond_t)
        mu_in = torch.cat([mu_t, zero_mu], dim=0)
        spks_in = torch.cat([spks_t, zero_spks], dim=0)
        cond_in = torch.cat([cond_t, zero_cond], dim=0)
        mask_in = torch.cat([mask_t, mask_t], dim=0)

        # Explicit DRAM, not the default memory config: when `use_trace` is set, `_capture`
        # ADOPTS these four tensors directly as the trace's own persistent conditioning
        # buffers (see `_capture`'s docstring) -- a trace bakes in the ADDRESS it reads
        # from, and a non-DRAM default (measured: reusing a captured trace at a NEW
        # `n_timesteps` corrupted the replay, PCC 0.58, traced back to exactly this) is not
        # guaranteed stable across the later `ttnn.copy`-based refresh `_reuse_trace` does.
        # Ported directly from the CosyVoice1 reference's own note on this ("allocated
        # explicitly in DRAM rather than inheriting a memory config ... since a trace bakes
        # in addresses").
        def up(x):
            return ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        mu_dev = up(mu_in)
        # [2B, 1, spk_dim], not [2B, spk_dim] -- see TtCausalConditionalDecoder's
        # spks handling / _sinusoidal_pos_emb's docstring for why.
        spks_dev = up(spks_in.unsqueeze(1))
        cond_dev = up(cond_in)
        mask_dev = up(mask_in)

        if use_trace:
            return self._forward_traced(mu_dev, spks_dev, cond_dev, mask_dev, z, t_span, t_len, streaming)
        result = self._solve_eager(mu_dev, spks_dev, cond_dev, mask_dev, z, t_span, t_len, streaming)
        for t in (mu_dev, spks_dev, cond_dev, mask_dev):
            ttnn.deallocate(t)
        return result

    def _solve_eager(self, mu_dev, spks_dev, cond_dev, mask_dev, z, t_span, t_len: int, streaming: bool):
        """The untraced Euler loop (host CFG blend + update) on already-uploaded conditioning. Used by
        `use_trace=False` and as the traced path's fallback. The streaming chunk-causal term is uploaded
        once per solve, not once per step."""
        chunk_bias = self.estimator.chunk_bias_device(t_len) if streaming else None
        t = t_span[0].unsqueeze(0)
        dt = t_span[1] - t_span[0]
        x = z
        for step in range(1, len(t_span)):
            x_in = torch.cat([x, x], dim=0)
            t_in = t.repeat(2)
            x_dev = ttnn.from_torch(x_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            temb_raw = self.estimator._sinusoidal_pos_emb(t_in)
            dphi_dev = self.estimator._forward_from_raw_temb(
                x_dev, mask_dev, mu_dev, temb_raw, spks_dev, cond_dev, t_len, batch_size=2, chunk_bias=chunk_bias
            )
            dphi = ttnn.to_torch(dphi_dev).float()
            dphi_dt, cfg_dphi_dt = dphi[:1], dphi[1:2]
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt.to(x.dtype)
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        if chunk_bias is not None:
            ttnn.deallocate(chunk_bias)
        return x

    # ------------------------------------------------------------------
    # Traced path. See the class docstring for the recipe and its provenance.
    # ------------------------------------------------------------------

    @staticmethod
    def _euler_schedule(t_span: torch.Tensor) -> list[tuple[float, float]]:
        """`(t, dt)` per step as plain floats, same update order as the eager loop above
        (and CosyVoice1's `cfm.py::euler_steps`) -- computed once, host-side, so the
        replay loop below does no per-step arithmetic at all."""
        t = float(t_span[0])
        dt = float(t_span[1] - t_span[0])
        out = []
        for step in range(1, len(t_span)):
            out.append((t, dt))
            t = t + dt
            if step < len(t_span) - 1:
                dt = float(t_span[step + 1]) - t
        return out

    @staticmethod
    def _trace_key_for(t_len: int, ch: int, streaming: bool):
        return (t_len, ch, streaming)

    def _release_trace(self) -> None:
        """Free every captured trace and the persistent device tensors each owns. Safe to call any
        time, including when nothing is captured."""
        while self._traces:
            _, slot = self._traces.popitem(last=False)
            slot.release(self.device)

    def _reuse_trace(self, slot: _CfmTraceSlot, mu2_dev, spks2_dev, cond2_dev, mask2_dev, z_dev) -> None:
        """Refill an already-captured trace's conditioning + initial state in place, for a new solve
        at the same key.

        Copied into the buffers rather than reassigned, same reasoning as `x_buf`: the trace holds
        these buffers' addresses, so reassigning the attribute would leave the replay reading the
        PREVIOUS utterance's conditioning with no error at all. `mask2_buf` is refreshed here along
        with the rest -- which is what carries a new valid length's padding term into a reused
        streaming trace (see the class docstring). `chunk_bias_buf` is not: it depends only on
        `t_len`, which the key fixes.
        """
        ttnn.copy(mu2_dev, slot.mu2_buf)
        ttnn.copy(spks2_dev, slot.spks2_buf)
        ttnn.copy(cond2_dev, slot.cond2_buf)
        ttnn.copy(mask2_dev, slot.mask2_buf)
        ttnn.copy(z_dev, slot.x_buf)
        # `_capture` syncs before its first `execute_trace` (after its own warm-up writes);
        # a reuse has no equivalent sync anywhere on its path to the replay loop's first
        # `execute_trace` otherwise -- measured effect of omitting this: a second solve on
        # a cached trace (same geometry, either the same or a different `n_timesteps`)
        # replayed against stale/partial buffer contents, PCC ~0.6 against the eager
        # reference despite the first solve on the same trace scoring PCC 0.9996+.
        ttnn.synchronize_device(self.device)

    def _capture(
        self,
        mu2_dev,
        spks2_dev,
        cond2_dev,
        mask2_dev,
        z_dev,
        temb_raw0_dev,
        dt0: float,
        t_len: int,
        ch: int,
        streaming: bool,
    ) -> _CfmTraceSlot:
        """Trace exactly ONE Euler step and return the slot holding the buffers it reads through.

        The traced body is CFG-doubling concat -> estimator -> CFG blend -> update, i.e.
        `_forward_from_raw_temb`'s whole cost plus a handful of elementwise ops -- nothing
        about how many times it will later be replayed is recorded here at all; that lives
        entirely in `_forward_traced`'s replay loop, on the host, between calls to
        `ttnn.execute_trace`.

        `mu2_dev`/`spks2_dev`/`cond2_dev`/`mask2_dev` are ADOPTED as the slot's persistent
        conditioning buffers. On a failure they are NOT freed here (the caller still owns them
        and falls back to the eager loop on them); only what this method itself allocated is.
        """

        def dram_zeros(shape):
            return ttnn.from_torch(
                torch.zeros(*shape),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        slot = _CfmTraceSlot(
            trace_id=None,
            next_x=None,
            x_buf=None,
            temb_raw_buf=None,
            dt_buf=None,
            mu2_buf=mu2_dev,
            spks2_buf=spks2_dev,
            cond2_buf=cond2_dev,
            mask2_buf=mask2_dev,
            chunk_bias_buf=None,
        )
        try:
            # Single-row buffer; the CFG doubling happens INSIDE the traced body from this
            # plain device tensor, not from a stored dim-0-concat result -- see the class
            # docstring's CosyVoice1-ported note on why (a concat *output* used as a `ttnn.copy`
            # source was measured at PCC 0.768 there; a plain tensor was bit-exact).
            slot.x_buf = dram_zeros((1, t_len, ch))
            ttnn.copy(z_dev, slot.x_buf)
            slot.temb_raw_buf = dram_zeros((2, 1, self.estimator.time_embeddings_dim))
            ttnn.copy(temb_raw0_dev, slot.temb_raw_buf)
            # dt varies per step, so (like the LLM decode trace's position/rope-index buffers)
            # it is a device tensor rather than a Python float -- otherwise its value would be
            # baked into the trace and every replay would use the first step's dt.
            slot.dt_buf = dram_zeros((1, 1, 1))
            dt0_dev = self._dt_device(dt0)
            ttnn.copy(dt0_dev, slot.dt_buf)
            ttnn.deallocate(dt0_dev)
            if streaming:
                slot.chunk_bias_buf = self.estimator.chunk_bias_device(t_len)
            # Compile, BEFORE capture, every `ttnn.copy` program the replay loop and `_reuse_trace`
            # will run into this slot's buffers (temb/dt/x above; the four conditioning refills here).
            # A copy program first compiled after capture allocates a program-cache device buffer
            # while the trace is live -- flagged by TT_METAL_TRACE_ALLOC_TRACKING=1 as liable to be
            # overwritten by the next replay (found 2026-09-25: the per-step `dt` copy, pre-existing).
            for buf in (slot.mu2_buf, slot.spks2_buf, slot.cond2_buf, slot.mask2_buf):
                tmp = ttnn.clone(buf, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.copy(tmp, buf)
                ttnn.deallocate(tmp)

            def body():
                x2 = ttnn.concat([slot.x_buf, slot.x_buf], dim=0)
                d = self.estimator._forward_from_raw_temb(
                    x2,
                    slot.mask2_buf,
                    slot.mu2_buf,
                    slot.temb_raw_buf,
                    slot.spks2_buf,
                    slot.cond2_buf,
                    t_len,
                    batch_size=2,
                    chunk_bias=slot.chunk_bias_buf,
                )
                ttnn.deallocate(x2)
                c = ttnn.slice(d, [0, 0, 0], [1, t_len, ch])
                u = ttnn.slice(d, [1, 0, 0], [2, t_len, ch])
                ttnn.deallocate(d)
                guided = ttnn.subtract(
                    ttnn.multiply(c, 1.0 + self.inference_cfg_rate), ttnn.multiply(u, self.inference_cfg_rate)
                )
                ttnn.deallocate(c)
                ttnn.deallocate(u)
                step = ttnn.multiply(guided, slot.dt_buf)
                ttnn.deallocate(guided)
                nxt = ttnn.add(slot.x_buf, step)
                ttnn.deallocate(step)
                return nxt

            # Warm the program cache AND every conv's prepared-weight / verified-config cache
            # inside the estimator (TtCausalConv1d / TtConvTranspose1d, see tt/hifigan/conv.py
            # and tt/hifigan/upsample.py) before capture -- both are host work a trace cannot
            # contain, and both are already keyed by (input_length, batch_size), so two full
            # eager passes at this exact geometry populate every cache capture will need (same
            # "warm up twice" requirement, and the same reason, as the CosyVoice1 recipe).
            for _ in range(2):
                ttnn.deallocate(body())
            ttnn.synchronize_device(self.device)

            slot.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            try:
                # Output allocated INSIDE the capture: its address is baked into the trace, so
                # every replay writes to this exact tensor -- no copy-out-of-the-graph step to
                # get silently dropped (see the CosyVoice1 recipe's note on why a
                # pre-allocated-buffer-plus-`ttnn.copy` version replayed to PCC 0.0017: the
                # copy never landed, `x` never advanced, and the "output" was the untouched
                # initial noise).
                slot.next_x = body()
            finally:
                ttnn.end_trace_capture(self.device, slot.trace_id, cq_id=0)
        except Exception:
            # Hand the adopted conditioning tensors back to the caller before releasing the rest --
            # the eager fallback still needs them.
            slot.mu2_buf = slot.spks2_buf = slot.cond2_buf = slot.mask2_buf = None
            slot.release(self.device)
            raise
        return slot

    def _temb_device(self, t_val: float):
        """Upload one schedule step's raw sinusoidal embedding as a fresh device tensor.
        Called per-replay (not pre-built as a list, and paired with `_dt_device` only where
        both are actually needed) -- measured effect of pre-building the whole schedule's
        temb/dt tensors upfront, before the reuse/capture dispatch below: a second solve on
        a cached trace replayed against corrupted buffer contents (PCC ~0.6 against the
        eager reference); building and freeing one step's tensors at a time, immediately
        around their own `execute_trace` call, removed the corruption. `_capture` still
        only runs ONCE per new key (see `_reuse_trace`), so this does not add a
        meaningful per-step cost of its own -- one small host-side upload per replay either
        way, pre-built or not."""
        t_in = torch.full((2,), t_val, dtype=torch.float32)
        emb = sinusoidal_pos_emb_torch(t_in, self.estimator.time_embeddings_dim)
        return ttnn.from_torch(
            emb.reshape(2, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _dt_device(self, dt_val: float):
        return ttnn.from_torch(
            torch.full((1, 1, 1), dt_val, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _forward_traced(
        self, mu_dev, spks_dev, cond_dev, mask_dev, z: torch.Tensor, t_span: torch.Tensor, t_len: int, streaming: bool
    ):
        ch = z.shape[-1]
        key = self._trace_key_for(t_len, ch, streaming)
        schedule = self._euler_schedule(t_span)
        z_dev = ttnn.from_torch(
            z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        slot = self._traces.get(key) if self._cache_trace else None
        reused = slot is not None
        try:
            if reused:
                self._reuse_trace(slot, mu_dev, spks_dev, cond_dev, mask_dev, z_dev)
                self._traces.move_to_end(key)
            else:
                # Evict least-recently-used traces down to capacity-1 BEFORE this capture allocates
                # anything: at capacity 1 that releases every trace first, so the new capture's
                # warm-up/buffer allocations happen with no trace live -- the original single-slot
                # ordering, unchanged.
                while len(self._traces) >= self._trace_capacity:
                    _, old = self._traces.popitem(last=False)
                    old.release(self.device)
                temb0_dev = self._temb_device(schedule[0][0])
                try:
                    slot = self._capture(
                        mu_dev, spks_dev, cond_dev, mask_dev, z_dev, temb0_dev, schedule[0][1], t_len, ch, streaming
                    )
                finally:
                    ttnn.deallocate(temb0_dev)
                self._traces[key] = slot
        except Exception as e:  # noqa: BLE001
            logger.warning(f"CFM trace capture unavailable, falling back to eager: {e}")
            slot = None

        if slot is None:
            # Fell back before capturing anything durable -- run the untraced eager loop on the
            # already-uploaded conditioning tensors instead of failing the solve.
            result = self._solve_eager(mu_dev, spks_dev, cond_dev, mask_dev, z, t_span, t_len, streaming)
            for t in (z_dev, mu_dev, spks_dev, cond_dev, mask_dev):
                ttnn.deallocate(t)
            return result

        if reused:
            # mu_dev/spks_dev/cond_dev/mask_dev were only a COPY SOURCE here -- `_reuse_trace`
            # copied their content into the slot's own persistent buffers (from an earlier
            # capture at this key), so these are now redundant. On a fresh `_capture`, the
            # opposite is true: the slot ADOPTS these exact tensors as its persistent buffers, so
            # they must NOT be freed here -- `_CfmTraceSlot.release` owns them from then on.
            for t in (mu_dev, spks_dev, cond_dev, mask_dev):
                ttnn.deallocate(t)
        # `z_dev` is only ever a copy source (into `x_buf`, by `_reuse_trace` or `_capture`), so it is
        # freed BEFORE the first replay: on a reuse it was allocated while this trace is live, and a
        # buffer from that window still alive at `execute_trace` is one the replay may overwrite
        # (TT_METAL_TRACE_ALLOC_TRACKING=1 flags it; found 2026-09-25, pre-existing).
        ttnn.deallocate(z_dev)

        # The traced body ITSELF allocates nothing during replay -- every tensor `body()`
        # touches is a persistent buffer. The per-step temb/dt UPLOAD (host -> device,
        # `_temb_device`/`_dt_device`) is not part of that graph at all; seeing PCC ~0.6
        # corruption specifically when 2x/10x of these were all held alive at once (see
        # `_temb_device`'s docstring) is why each pair is built, used, and freed before
        # the next one is even allocated.
        for t_val, dt_val in schedule:
            temb_dev = self._temb_device(t_val)
            dt_dev = self._dt_device(dt_val)
            ttnn.copy(temb_dev, slot.temb_raw_buf)
            ttnn.copy(dt_dev, slot.dt_buf)
            ttnn.deallocate(temb_dev)
            ttnn.deallocate(dt_dev)
            ttnn.execute_trace(self.device, slot.trace_id, cq_id=0, blocking=True)
            ttnn.copy(slot.next_x, slot.x_buf)
        result = ttnn.to_torch(slot.x_buf).float().reshape(1, t_len, ch)
        if not self._cache_trace:
            del self._traces[key]
            slot.release(self.device)
        return result

    def release_cfm_trace(self) -> None:
        """Public wrapper, matching `TtQwen2LM.release_decode_trace`'s naming -- callers
        (e.g. `TtCausalMaskedDiffWithXvec`) should call this when they are done with the
        CFM for good, same trace-scope discipline the LLM decode trace uses."""
        self._release_trace()
